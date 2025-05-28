"""Conv2D schedule on for Intel CPU"""
import tvm
from tvm.topi.utils import get_const_tuple


def _custom_schedule_conv_NCHWc(s, cfg, data_vec, kernel_vec, conv_out, last):
    # fetch schedule
    reg_n, unroll_kw = cfg["tile_ow"].size[-1], cfg["unroll_kw"].val
    _, _, _, _, ic_bn = get_const_tuple(data_vec.shape)

    # schedule pad
    if isinstance(s[data_vec].op, tvm.te.ComputeOp) and "pad" in data_vec.op.tag:
        batch, ic_chunk, ih, iw, ic_block = s[data_vec].op.axis
        s[data_vec].vectorize(ic_block)
        parallel_axis = s[data_vec].fuse(batch, ic_chunk, ih)
        s[data_vec].parallel(parallel_axis)
        data_vec = data_vec.op.input_tensors[0]

    oc_bn = cfg["tile_oc"].size[-1]
    if isinstance(kernel_vec.op, tvm.te.ComputeOp) and kernel_vec.name == "kernel_vec":
        # data and kernel are not pre-computed, schedule layout transform here.
        # this should only be used by x86 conv2d_nchw, which is for
        # testing purpose.
        batch, ic_chunk, ih, ic_block, iw = s[data_vec].op.axis
        parallel_axis = s[data_vec].fuse(batch, ic_chunk, ih)
        s[data_vec].parallel(parallel_axis)

        oc_chunk, ic_chunk, oh, ow, ic_block, oc_block = s[kernel_vec].op.axis
        s[kernel_vec].reorder(oc_chunk, oh, ic_chunk, ow, ic_block, oc_block)
        if oc_bn > 1:
            s[kernel_vec].vectorize(oc_block)
        parallel_axis = s[kernel_vec].fuse(oc_chunk, oh)
        s[kernel_vec].parallel(parallel_axis)

    # schedule 5-D NCHW[x]c conv
    C, O = conv_out, last
    CC = s.cache_write(C, "global")

    batch, oc_chunk, oh, ow, oc_block = s[C].op.axis
    ow_chunk, ow_block = s[C].split(ow, factor=reg_n)
    s[C].reorder(oc_chunk, oh, ow_chunk, ow_block, oc_block)
    parallel_axis = s[C].fuse(batch, oc_chunk, oh)
    s[C].vectorize(oc_block)
    if C == O:
        s[C].parallel(parallel_axis)

    s[CC].compute_at(s[C], ow_chunk)
    _, oc_chunk, oh, ow, oc_block = s[CC].op.axis
    ic, kh, kw = s[CC].op.reduce_axis

    ow_chunk, ow_block = s[CC].split(ow, factor=reg_n)
    ic_chunk, ic_block = s[CC].split(ic, factor=ic_bn)

    if unroll_kw:
        s[CC].reorder(oc_chunk, oh, ow_chunk, ic_chunk, kh, ic_block, kw, ow_block, oc_block)
        s[CC].unroll(kw)
    else:
        s[CC].reorder(oc_chunk, oh, ow_chunk, ic_chunk, kh, kw, ic_block, ow_block, oc_block)

    s[CC].vectorize(oc_block)
    s[CC].unroll(ow_block)

    if C != O:
        out_ndim = len(s[O].op.axis)
        if out_ndim == 5:
            batch, oc_chunk, oh, ow, oc_block = s[O].op.axis
            ow_chunk, ow_block = s[O].split(ow, factor=reg_n)
            s[O].reorder(oc_chunk, oh, ow_chunk, ow_block, oc_block)
            parallel_axis = s[O].fuse(batch, oc_chunk, oh)
            s[C].compute_at(s[O], parallel_axis)
            s[O].vectorize(oc_block)
            s[O].parallel(parallel_axis)
        elif out_ndim == 4:
            batch, oc, oh, ow = s[O].op.axis
            ow_chunk, ow_block = s[O].split(ow, factor=reg_n)
            oc_chunk, oc_block = s[O].split(oc, factor=oc_bn)
            s[O].reorder(oc_chunk, oh, ow_chunk, ow_block, oc_block)
            parallel_axis = s[O].fuse(batch, oc_chunk, oh)
            s[C].compute_at(s[O], parallel_axis)
            s[O].vectorize(oc_block)
            s[O].parallel(parallel_axis)
        else:
            raise ValueError(f"Unsupported output ndim: {out_ndim}")

        print("CONV2D: YES")
        s[O].pragma(parallel_axis, "outerloop")

    return s
