import tvm
from tvm import autotvm, te, tir, topi
from tvm.te import SpecializedCondition
from tvm.topi.utils import get_const_tuple, traverse_inline
from tvm.topi.x86.dense import _default_dense_pack_config
from tvm.auto_scheduler import is_auto_scheduler_enabled
from tvm.meta_schedule import is_meta_schedule_enabled
from tvm.relay.op.strategy.generic import wrap_compute_dense, wrap_topi_schedule, naive_schedule
from tvm.topi import tag
from tvm.relay import op as _op


def _custom_schedule_dense_pack_template(cfg, s, C, O):
    A, packedB = s[C].op.input_tensors

    CC = s.cache_write(C, "global")
    y, x = s[C].op.axis
    (k,) = s[CC].op.reduce_axis

    yt, yo, yi = cfg["tile_y"].apply(s, C, y)
    xt, xo, xi = cfg["tile_x"].apply(s, C, x)
    s[C].reorder(xt, yt, yo, xo, yi, xi)
    xyt = s[C].fuse(xt, yt)
    if C == O:
        s[C].parallel(xyt)
    xyo = s[C].fuse(yo, xo)
    s[C].unroll(yi)
    s[C].vectorize(xi)

    s[CC].compute_at(s[C], xyo)
    y, x = s[CC].op.axis
    ko, ki = cfg["tile_k"].apply(s, CC, k)
    s[CC].reorder(ko, ki, y, x)
    s[CC].vectorize(x)

    tile_inner = cfg["tile_inner"].size[-1]
    if tile_inner > 1:
        yo, yi = s[CC].split(y, tile_inner)
        s[CC].reorder(ko, yo, ki, yi, x)
        s[CC].unroll(yo)
        s[CC].unroll(ki)
        s[CC].unroll(yi)
    else:
        s[CC].unroll(ki)
        s[CC].unroll(y)

    if C != O:
        y, x = s[O].op.axis
        yt, yo, yi = cfg["tile_y"].apply(s, O, y)
        xt, xo, xi = cfg["tile_x"].apply(s, O, x)
        s[O].reorder(xt, yt, yo, xo, yi, xi)
        xyt = s[O].fuse(xt, yt)
        s[C].compute_at(s[O], xyt)
        s[O].vectorize(xi)
        s[O].parallel(xyt)

    print("DENSE: YES")
    s[O].pragma(xyt, "outerloop")

    return s


@autotvm.register_topi_compute("custom_dense_pack.x86")
def custom_dense_pack(cfg, data, weight, bias=None, out_dtype=None):
    """Compute dense with transformed weight."""
    if out_dtype is None:
        out_dtype = data.dtype
    M, K = get_const_tuple(data.shape)  # batch, in_dim
    if len(weight.shape) == 3:
        N, _, packw_bn = get_const_tuple(weight.shape)  # out_dim
        N = N * packw_bn
    else:
        N, _ = get_const_tuple(weight.shape)  # out_dim
    # create tuning space
    cfg.define_split(
        "tile_y", 32 if isinstance(M, (tvm.tir.Var, tvm.tir.Any)) else M, num_outputs=3
    )
    cfg.define_split(
        "tile_x", 32 if isinstance(N, (tvm.tir.Var, tvm.tir.Any)) else N, num_outputs=3
    )
    cfg.define_split(
        "tile_k", 32 if isinstance(K, (tvm.tir.Var, tvm.tir.Any)) else K, num_outputs=2
    )
    cfg.define_split(
        "tile_inner",
        32 if isinstance(M, (tvm.tir.Var, tvm.tir.Any)) else M,
        num_outputs=2,
        filter=lambda y: y.size[-1] <= 16,
    )
    if cfg.is_fallback:
        _default_dense_pack_config(cfg, M, N, K)

    if len(weight.shape) == 2:
        packw_bn = cfg["tile_x"].size[-1]
        packw_shape = (N // packw_bn, K, packw_bn)
        if autotvm.GLOBAL_SCOPE.in_tuning:
            # Directly use modified data layout placeholder.
            packw = tvm.te.placeholder(packw_shape, weight.dtype, name="packed_weight")
        else:
            packw = te.compute(
                packw_shape, lambda z, y, x: weight[z * packw_bn + x, y], name="packed_weight"
            )
    else:
        packw = weight

    idxdiv = tvm.tir.indexdiv
    idxmod = tvm.tir.indexmod
    k = te.reduce_axis((0, K), name="k")
    C = te.compute(
        (M, N),
        lambda y, x: te.sum(
            data[y, k].astype(out_dtype)
            * packw[idxdiv(x, packw_bn), k, idxmod(x, packw_bn)].astype(out_dtype),
            axis=k,
        ),
        tag="dense_pack",
    )
    if bias is not None:
        C = te.compute((M, N), lambda i, j: C[i, j] + bias[j].astype(out_dtype), tag=tag.BROADCAST)
    return C


@autotvm.register_topi_schedule("custom_dense_pack.x86")
def custom_schedule_dense_pack(cfg, outs):
    """Create the schedule for dense_pack"""
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if "dense_pack" in op.tag:
            _custom_schedule_dense_pack_template(cfg, s, op.output(0), outs[0])

    traverse_inline(s, outs[0].op, _callback)
    return s


@tvm.relay.op.strategy.generic.dense_strategy.register("cpu", override=True)
def dense_strategy_cpu(attrs, inputs, out_type, target):
    """dense x86 strategy"""

    strategy = _op.OpStrategy()
    # For dynamic matrix-vector multiply we use a hand written kernel.
    if (
        isinstance(inputs[0].shape[0], (int, tir.IntImm))
        and inputs[0].shape[0] == 1
        and (
            topi.utils.is_dynamic_shape(inputs[0].shape)
            or topi.utils.is_dynamic_shape(inputs[1].shape)
        )
    ):
        strategy.add_implementation(
            wrap_compute_dense(topi.x86.dense_dynamic),
            wrap_topi_schedule(topi.x86.schedule_dense_dynamic),
            name="dense_dynamic.x86",
            plevel=20,
        )
        return strategy

    same_type = inputs[0].dtype == inputs[1].dtype == out_type.dtype
    dtype = inputs[0].dtype
    u8s8s32 = dtype == "uint8" and inputs[1].dtype == "int8" and out_type.dtype == "int32"
    strategy.add_implementation(
        wrap_compute_dense(topi.x86.dense_nopack),
        wrap_topi_schedule(topi.x86.schedule_dense_nopack),
        name="dense_nopack.x86",
        plevel=5,
    )

    strategy.add_implementation(
        wrap_compute_dense(custom_dense_pack),
        wrap_topi_schedule(custom_schedule_dense_pack),
        name="custom_dense_pack.x86",
        plevel=10,
    )

    need_auto_scheduler_layout = is_auto_scheduler_enabled()
    need_meta_schedule_layout = is_meta_schedule_enabled()

    if need_auto_scheduler_layout or need_meta_schedule_layout:
        strategy.add_implementation(
            wrap_compute_dense(
                topi.nn.dense,
                need_auto_scheduler_layout=need_auto_scheduler_layout,
                need_meta_schedule_layout=need_meta_schedule_layout,
            ),
            naive_schedule,
            name="dense.generic",
            plevel=11,
        )

    if "cblas" in target.libs:
        with SpecializedCondition(same_type and dtype in ["float32", "float64"]):
            strategy.add_implementation(
                wrap_compute_dense(topi.x86.dense_cblas),
                wrap_topi_schedule(topi.x86.schedule_dense_cblas),
                name="dense_cblas.x86",
                plevel=13,
            )
    if "mkl" in target.libs:
        with SpecializedCondition(same_type and dtype in ["float32", "float64"] or u8s8s32):
            strategy.add_implementation(
                wrap_compute_dense(topi.x86.dense_mkl),
                wrap_topi_schedule(topi.x86.schedule_dense_mkl),
                name="dense_mkl.x86",
                plevel=14,
            )
    if "dnnl" in target.libs:
        with SpecializedCondition(same_type and dtype == "float32"):
            strategy.add_implementation(
                wrap_compute_dense(topi.x86.dense_dnnl),
                wrap_topi_schedule(topi.x86.schedule_dense_dnnl),
                name="dense_dnnl.x86",
                plevel=15,
            )
    return strategy


@tvm.relay.op.strategy.generic.dense_pack_strategy.register("cpu", override=True)
def dense_pack_strategy_cpu(attrs, inputs, out_type, target):
    """dense_pack x86 strategy"""
    strategy = _op.OpStrategy()
    if (
        inputs[0].dtype == "uint8"
        and inputs[1].dtype == "int8"
        and out_type.dtype == "int32"
        and attrs["weight_layout"] == "NC16n4c"
    ):
        strategy.add_implementation(
            wrap_compute_dense(topi.x86.dense_int8),
            wrap_topi_schedule(topi.x86.schedule_dense_int8),
            name="dense_int8.x86",
            plevel=13,
        )
    else:
        strategy.add_implementation(
            wrap_compute_dense(custom_dense_pack),
            wrap_topi_schedule(custom_schedule_dense_pack),
            name="custom_dense_pack.x86",
            plevel=10,
        )

    return strategy
