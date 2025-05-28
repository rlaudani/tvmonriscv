import tvm
from tvm import autotvm, te, topi
from tvm.topi import nn
from tvm.topi.nn.utils import get_pad_tuple
from tvm.topi.utils import get_const_tuple, traverse_inline
from tvm.topi.x86 import conv2d_avx_1x1
from tvm.topi.x86.conv2d import _pack_data, _get_default_config
from tvm.relay.op.strategy.generic import wrap_compute_conv2d, wrap_topi_schedule
from tvm.relay import op as _op

from strategy.x86.conv2d_avx_common import _custom_schedule_conv_NCHWc


@autotvm.register_topi_compute("custom_conv2d_NCHWc.x86")
def custom_conv2d_NCHWc(cfg, data, kernel, strides, padding, dilation, layout, out_layout, out_dtype):
    """Compute conv2d with NCHWc layout."""
    # layout and out_layout are not used here,
    # we keep them for debug convenience when dumping autotvm workload
    if len(data.shape) == 5:
        n, ic_chunk, ih, iw, ic_bn = get_const_tuple(data.shape)
        oc_chunk, ic_chunk_group, kernel_height, kernel_width, _, oc_bn = get_const_tuple(
            kernel.shape
        )
        in_channel = ic_chunk * ic_bn
        num_filter = oc_chunk * oc_bn
    else:
        n, in_channel, ih, iw = get_const_tuple(data.shape)
        num_filter, _, kernel_height, kernel_width = get_const_tuple(kernel.shape)

    # Define autotvm tuning space
    is_kernel_1x1 = kernel_height == 1 and kernel_width == 1
    pt, pl, pb, pr = get_pad_tuple(padding, (kernel_height, kernel_width))
    sh, sw = strides if isinstance(strides, (tuple, list)) else (strides, strides)
    oh = (ih - kernel_height + pt + pb) // sh + 1
    ow = (iw - kernel_width + pl + pr) // sw + 1

    cfg.define_split("tile_ic", in_channel, num_outputs=2)
    cfg.define_split("tile_oc", num_filter, num_outputs=2)
    if isinstance(ow, (tvm.tir.IntImm, int)):
        cfg.define_split(
            "tile_ow", ow, num_outputs=2, filter=lambda y: y.size[-1] <= 64, policy="verbose"
        )
    if is_kernel_1x1:
        if isinstance(oh, (tvm.tir.IntImm, int)):
            cfg.define_knob("tile_oh", [1, 2] if oh > 1 else [1])
    else:
        cfg.define_knob("unroll_kw", [True, False])

    # If no config was set, we can fallback to default config.
    if cfg.is_fallback:
        _get_default_config(
            cfg,
            te.placeholder((n, in_channel, ih, iw), dtype=data.dtype),
            te.placeholder(
                (num_filter, in_channel, kernel_height, kernel_width), dtype=kernel.dtype
            ),
            strides,
            padding,
            dilation,
            out_dtype,
        )

    # Pack data if raw 4-D data is provided.
    # This can only happen when autotuning.
    if len(data.shape) == 4:
        if autotvm.GLOBAL_SCOPE.in_tuning:
            # Directly use modified data layout placeholder.
            dshape = (n, in_channel // cfg["tile_ic"].size[-1], ih, iw, cfg["tile_ic"].size[-1])
            data = tvm.te.placeholder(dshape, data.dtype, name="data")
            kshape = (
                num_filter // cfg["tile_oc"].size[-1],
                in_channel // cfg["tile_ic"].size[-1],
                kernel_height,
                kernel_width,
                cfg["tile_ic"].size[-1],
                cfg["tile_oc"].size[-1],
            )
            kernel = tvm.te.placeholder(kshape, kernel.dtype, name="kernel")
        else:
            data, kernel = _pack_data(cfg, data, kernel)

    return nn.conv2d_NCHWc(data, kernel, strides, padding, dilation, layout, out_layout, out_dtype)


@autotvm.register_topi_schedule("custom_conv2d_NCHWc.x86")
def custom_schedule_conv2d_NCHWc(cfg, outs):
    """Create schedule for tensors"""
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if "conv2d_NCHWc" in op.tag:
            conv_out = op.output(0)
            kernel_vec = conv_out.op.input_tensors[1]
            data_vec = conv_out.op.input_tensors[0]

            args = [s, cfg, data_vec, kernel_vec, conv_out, outs[0]]
            (_, _, kh, kw, _, _) = get_const_tuple(kernel_vec.shape)
            if kh == 1 and kw == 1:
                conv2d_avx_1x1._schedule_conv_NCHWc(*args)
            else:
                _custom_schedule_conv_NCHWc(*args)

    traverse_inline(s, outs[0].op, _callback)
    return s


@tvm.relay.op.strategy.generic.conv2d_NCHWc_strategy.register("cpu", override=True)
def conv2d_NCHWc_strategy_cpu(attrs, inputs, out_type, target):
    """conv2d_NCHWc x86 strategy"""
    strategy = _op.OpStrategy()
    data, kernel = inputs
    if topi.x86.is_int8_hw_support(data.dtype, kernel.dtype):
        strategy.add_implementation(
            wrap_compute_conv2d(
                topi.x86.conv2d_NCHWc_int8, need_data_layout=True, need_out_layout=True
            ),
            wrap_topi_schedule(topi.x86.schedule_conv2d_NCHWc_int8),
            name="conv2d_NCHWc_int8.x86",
        )
    else:
        strategy.add_implementation(
            wrap_compute_conv2d(custom_conv2d_NCHWc, need_data_layout=True, need_out_layout=True),
            wrap_topi_schedule(custom_schedule_conv2d_NCHWc),
            name="custom_conv2d_NCHWc.x86",
        )
    return strategy
