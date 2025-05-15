import tvm
from tvm import te
from tvm import topi
import tvm.relay.op as _op
import tvm.relay.op.strategy.x86 as _strategy


def wrap_topi_schedule(topi_schedule):
    """Wrap TOPI schedule which doesn't use attrs"""
    def wrapper(attrs, outs, target):
        with target:
            
            # s = te.create_schedule([x.op for x in outs])

            # print("OUTS\n", outs)
            # print("te.create_schedule\n")
            # print(s)
            # print(topi_schedule(outs))


            # conv_op = [stge for stge in s.stages if stge.op.tag == 'conv2d_NCHWc']
            # conv_op = conv_op[0]
            # n, *_ = conv_op.all_iter_vars
            # conv_op.pragma(var=n, pragma_type="outerloop")
            # return s


            # s = te.create_schedule([x.op for x in outs])
            # conv_op = [stge for stge in s.stages if stge.op.tag == 'conv2d_NCHWc']
            # conv_op = conv_op[0]
            # print(len(conv_op.all_iter_vars))
            # n, *x = conv_op.all_iter_vars
            # conv_op.pragma(var=n, pragma_type="outerloop")
            # return s

            # print(conv_op[0].all_iter_vars)
            # if len(conv_op) != 1 :
            #     raise Exception("Expected one conv2d operation in 'stages'.")
            # conv_op = conv_op[0]
            
            # if len(conv_op.all_iter_vars) == 7 :
            #     n, oc, oh, ow, kh, kw, ki = conv_op.all_iter_vars
            #     conv_op.pragma(var=n, pragma_type="outerloop")
            # else :
            #     print("Conv2D NCHW has not enough axes due to optimization.")
            # return s

            return topi_schedule(outs)

    return wrapper


@_strategy.conv2d_strategy.register(["cpu"])
def conv2d_strategy_cpu(attrs, inputs, out_type, target):
    """conv2d x86 strategy"""
    strategy = _op.OpStrategy()
    data, kernel = inputs
    stride_h, stride_w = _strategy.get_const_tuple(attrs.strides)
    dilation_h, dilation_w = _strategy.get_const_tuple(attrs.dilation)
    groups = attrs.groups
    layout = attrs.data_layout
    kernel_layout = attrs.kernel_layout
    if dilation_h < 1 or dilation_w < 1:
        raise ValueError("dilation should be positive value")

    need_auto_scheduler_layout = _strategy.is_auto_scheduler_enabled()
    need_meta_schedule_layout = _strategy.is_meta_schedule_enabled()

    print("STRATEGY")

    if groups == 1:
        if layout == "NCHW":
            assert kernel_layout == "OIHW"
            strategy.add_implementation(
                _strategy.wrap_compute_conv2d(topi.x86.conv2d_nchw),
                wrap_topi_schedule(topi.x86.schedule_conv2d_nchw),
                name="conv2d_nchw.x86"
            )
        else:
            print("STRATEGY: ELSE")
            return _strategy.conv2d_strategy_cpu(attrs, inputs, out_type, target)

    return strategy









    # # Depthwise Conv2D
    # elif _strategy.is_depthwise_conv2d(data.shape, layout, kernel.shape, kernel_layout, groups):
    #     if layout == "NCHW":
    #         assert kernel_layout == "OIHW"
    #         channel_multiplier = _strategy.get_const_tuple(inputs[1].shape)[1]
    #         if channel_multiplier == 1 and dilation_h == 1 and dilation_w == 1:
    #             strategy.add_implementation(
    #                 _strategy.wrap_compute_conv2d(topi.x86.depthwise_conv2d_nchw),
    #                 _strategy.wrap_topi_schedule(topi.x86.schedule_depthwise_conv2d_nchw),
    #                 name="depthwise_conv2d_nchw.x86",
    #             )
    #         else:
    #             _strategy.logger.warning(
    #                 "For x86 target, depthwise_conv2d with channel "
    #                 "multiplier greater than 1 is not optimized"
    #             )
    #             strategy.add_implementation(
    #                 _strategy.wrap_compute_conv2d(topi.nn.depthwise_conv2d_nchw),
    #                 _strategy.wrap_topi_schedule(topi.generic.schedule_depthwise_conv2d_nchw),
    #                 name="depthwise_conv2d_nchw.generic",
    #             )
    #     else:
    #         return _strategy.conv2d_strategy_cpu(attrs, inputs, out_type, target)

    # else:  # group_conv2d
    #     if layout == "NCHW":
    #         assert kernel_layout == "OIHW"
    #         strategy.add_implementation(
    #             _strategy.wrap_compute_conv2d(topi.x86.group_conv2d_nchw, has_groups=True),
    #             _strategy.wrap_topi_schedule(topi.x86.schedule_group_conv2d_nchw),
    #             name="group_conv2d_nchw.x86",
    #         )
    #     else:
    #         return _strategy.conv2d_strategy_cpu(attrs, inputs, out_type, target)
