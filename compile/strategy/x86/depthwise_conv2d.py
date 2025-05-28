import tvm
from tvm import autotvm, te, topi
from tvm.topi.nn.utils import get_pad_tuple
from tvm.topi.nn.pad import pad
from tvm.topi.utils import get_const_tuple, traverse_inline
from tvm.topi.x86.depthwise_conv2d import _get_workload, _fallback_schedule, _pack_data
from tvm.relay.op.strategy.generic import wrap_compute_conv2d, wrap_topi_schedule
from tvm.relay import op as _op


@autotvm.register_topi_compute("custom_depthwise_conv2d_NCHWc.x86")
def custom_depthwise_conv2d_NCHWc(
    cfg, data, kernel, strides, padding, dilation, layout, out_layout, out_dtype=None
):
    """Compute depthwise conv2d with NCHWc layout"""
    out_dtype = data.dtype if out_dtype is None else out_dtype

    if len(data.shape) == 5:
        batch, in_channel_chunk, in_height, in_width, in_channel_block = get_const_tuple(data.shape)
        (
            out_channel_chunk,
            cm_chunk,
            filter_height,
            filter_width,
            cm_block,
            out_channel_block,
        ) = get_const_tuple(kernel.shape)
        in_channel = in_channel_chunk * in_channel_block
        out_channel = out_channel_chunk * out_channel_block
        channel_multiplier = cm_chunk * cm_block
        assert channel_multiplier * in_channel == out_channel
    else:
        batch, in_channel, in_height, in_width = get_const_tuple(data.shape)
        out_channel, channel_multiplier, filter_height, filter_width = get_const_tuple(kernel.shape)
    assert channel_multiplier == 1

    strides = strides if isinstance(strides, (tuple, list)) else (strides, strides)
    HSTR, WSTR = strides

    dh, dw = dilation if isinstance(dilation, (tuple, list)) else (dilation, dilation)

    dilated_kernel_h = (filter_height - 1) * dh + 1
    dilated_kernel_w = (filter_width - 1) * dw + 1
    pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
        padding, (dilated_kernel_h, dilated_kernel_w)
    )
    HPAD = pad_top + pad_down
    WPAD = pad_left + pad_right

    out_height = (in_height + HPAD - dilated_kernel_h) // HSTR + 1
    out_width = (in_width + WPAD - dilated_kernel_w) // WSTR + 1

    cfg.define_split("tile_ic", in_channel, num_outputs=2)
    cfg.define_split("tile_oc", out_channel, num_outputs=2)
    cfg.define_split("tile_ow", out_width, num_outputs=2, filter=lambda y: y.size[-1] <= 64)
    cfg.define_knob("unroll_kw", [True, False])

    # get workload and related schedule config
    wkl = _get_workload(
        te.placeholder((batch, in_channel, in_height, in_width), dtype=data.dtype),
        te.placeholder(
            (out_channel, channel_multiplier, filter_height, filter_width), dtype=kernel.dtype
        ),
        strides,
        (pad_top, pad_down),
        dilation,
        out_dtype,
    )
    if cfg.is_fallback:
        _fallback_schedule(cfg, wkl)

    # Pack data if raw 4-D data is provided.
    # This can only happen when autotuning.
    if len(data.shape) == 4:
        if autotvm.GLOBAL_SCOPE.in_tuning:
            # Directly use modified data layout placeholder.
            in_channel_block = cfg["tile_ic"].size[-1]
            in_channel_chunk = in_channel // in_channel_block
            out_channel_block = cfg["tile_oc"].size[-1]
            out_channel_chunk = out_channel // out_channel_block
            dshape = (batch, in_channel_chunk, in_height, in_width, in_channel_block)
            data = tvm.te.placeholder(dshape, data.dtype, name="data")
            kshape = (out_channel_chunk, 1, filter_height, filter_width, 1, out_channel_block)
            kernel = tvm.te.placeholder(kshape, kernel.dtype, name="kernel")
        else:
            data, kernel = _pack_data(cfg, data, kernel)
            _, _, _, _, in_channel_block = get_const_tuple(data.shape)
            out_channel_chunk, _, _, _, _, out_channel_block = get_const_tuple(kernel.shape)

    # padding stage
    DOPAD = pad_top != 0 or pad_left != 0 or pad_down != 0 or pad_right != 0
    if DOPAD:
        pad_before = [0, 0, pad_top, pad_left, 0]
        pad_after = [0, 0, pad_down, pad_right, 0]
        data_pad = pad(data, pad_before, pad_after, name="PaddedInput")
    else:
        data_pad = data

    # depthconv stage
    idxdiv = tvm.tir.indexdiv
    idxmod = tvm.tir.indexmod

    kh = te.reduce_axis((0, filter_height), name="kh")
    kw = te.reduce_axis((0, filter_width), name="kw")
    Output = te.compute(
        (batch, out_channel_chunk, out_height, out_width, out_channel_block),
        lambda b, oco, oh, ow, oci: te.sum(
            (
                data_pad[
                    b,
                    idxdiv(
                        idxdiv(oco * out_channel_block + oci, channel_multiplier), in_channel_block
                    ),
                    oh * HSTR + kh * dh,
                    ow * WSTR + kw * dw,
                    idxmod(
                        idxdiv(oco * out_channel_block + oci, channel_multiplier), in_channel_block
                    ),
                ].astype(out_dtype)
                * kernel[oco, 0, kh, kw, 0, oci].astype(out_dtype)
            ),
            axis=[kh, kw],
        ),
        name="DepthwiseConv2d",
        tag="depthwise_conv2d_NCHWc",
    )
    return Output


@autotvm.register_topi_schedule("custom_depthwise_conv2d_NCHWc.x86")
def custom_schedule_depthwise_conv2d_NCHWc(cfg, outs):
    """CPU schedule for depthwise conv2d in NCHW[x]c layout"""
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        """Traverse operators from computation graph"""
        if "depthwise_conv2d_NCHWc" in op.tag:
            conv_out = op.output(0)
            data = conv_out.op.input_tensors[0]
            kernel = conv_out.op.input_tensors[1]
            _custom_schedule_depthwise_conv2d_NCHWc_impl(s, cfg, data, kernel, conv_out, outs[0])

    traverse_inline(s, outs[0].op, _callback)
    return s


def _custom_schedule_depthwise_conv2d_NCHWc_impl(s, cfg, data_vec, kernel_vec, conv_out, output):
    tile_ow, oc_bn = cfg["tile_ow"].size[-1], cfg["tile_oc"].size[-1]
    unroll_kw = cfg["unroll_kw"].val

    # schedule pad
    if isinstance(s[data_vec].op, tvm.te.ComputeOp) and "pad" in data_vec.op.tag:
        batch, ic_chunk, ih, iw, ic_block = s[data_vec].op.axis
        s[data_vec].vectorize(ic_block)
        parallel_axis = s[data_vec].fuse(batch, ic_chunk, ih)
        s[data_vec].parallel(parallel_axis)

    C, O = conv_out, output
    CC = s.cache_write(C, "global")

    _, ic_chunk, oh, ow, ic_block = s[C].op.axis
    ow_chunk, ow_block = s[C].split(ow, factor=tile_ow)
    s[C].reorder(ic_chunk, oh, ow_chunk, ow_block, ic_block)
    s[C].vectorize(ic_block)
    parallel_axis = s[C].fuse(ic_chunk, oh)
    s[C].parallel(parallel_axis)
    s[CC].compute_at(s[C], ow_chunk)

    # the ow axis in the cached block CC is the ow_block in C
    _, ic_chunk, oh, ow, ic_block = s[CC].op.axis
    kh, kw = s[CC].op.reduce_axis
    s[CC].reorder(ic_chunk, oh, kh, kw, ow, ic_block)
    if unroll_kw:
        s[CC].unroll(kw)
    s[CC].vectorize(ic_block)
    s[CC].unroll(ow)

    if C != O:
        out_ndim = len(s[O].op.axis)
        if out_ndim == 5:
            batch, oc_chunk, oh, ow, oc_block = s[O].op.axis
            ow_chunk, ow_block = s[O].split(ow, factor=tile_ow)
            s[O].reorder(oc_chunk, oh, ow_chunk, ow_block, oc_block)
            parallel_axis = s[O].fuse(oc_chunk, oh)
            s[C].compute_at(s[O], parallel_axis)
            s[O].vectorize(oc_block)
            s[O].parallel(parallel_axis)
        elif out_ndim == 4:
            batch, oc, oh, ow = s[O].op.axis
            ow_chunk, ow_block = s[O].split(ow, factor=tile_ow)
            oc_chunk, oc_block = s[O].split(oc, factor=oc_bn)
            s[O].reorder(oc_chunk, oh, ow_chunk, ow_block, oc_block)
            parallel_axis = s[O].fuse(oc_chunk, oh)
            s[C].compute_at(s[O], parallel_axis)
            s[O].vectorize(oc_block)
            s[O].parallel(parallel_axis)
        else:
            raise ValueError(f"Unsupported output ndim: {out_ndim}")

        s[O].pragma(parallel_axis, "outerloop")

    return s


@tvm.relay.op.strategy.generic.depthwise_conv2d_NCHWc_strategy.register("cpu", override=True)
def depthwise_conv2d_NCHWc_strategy_cpu(attrs, inputs, out_type, target):
    """depthwise_conv2d x86 strategy"""
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_conv2d(
            custom_depthwise_conv2d_NCHWc, need_data_layout=True, need_out_layout=True
        ),
        wrap_topi_schedule(custom_schedule_depthwise_conv2d_NCHWc),
        name="custom_depthwise_conv2d_NCHWc.x86",
    )
    return strategy
