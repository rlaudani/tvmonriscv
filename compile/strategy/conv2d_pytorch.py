import tvm
from tvm import te
from tvm import topi
import tvm.relay.op as _op
import tvm.relay.op.strategy as _strategy
from tvm.relay.op.strategy.generic import is_depthwise_conv2d


def default_compute(op) :
    return _strategy.generic.wrap_compute_conv2d(op)
def default_topi_schedule(sched) :
    return _strategy.generic.wrap_topi_schedule(sched)
def default_depthwise_compute(op) :
    return _strategy.generic.wrap_compute_conv2d(op, need_kernel_layout=True)

def compute_conv2d_nchw(op) :
    return default_compute(op)
def compute_depthwise_conv2d_nchw(op) :
    return default_depthwise_compute(op)


def topi_schedule_conv2d_nchw(topi_schedule) :
    def wrapper(attrs, outs, target):
        with target:
            s = te.create_schedule([x.op for x in outs])
            
            conv_op = [stge for stge in s.stages if stge.op.tag == 'conv2d_nchw']
            if len(conv_op) != 1 :
                raise Exception("Expected one conv2d operation in 'stages'.")
            conv_op = conv_op[0]
            
            if len(conv_op.all_iter_vars) == 7 :
                n, oc, oh, ow, kh, kw, ki = conv_op.all_iter_vars
                conv_op.pragma(var=n, pragma_type="outerloop")
            else :
                print("Conv2D NCHW has not enough axes due to optimization.")
            return s
    return wrapper


@_strategy.generic.conv2d_strategy.register(["cpu"])
def conv2d_strategy_(attrs, inputs, out_type, target):
    strategy = _op.OpStrategy()
    layout = attrs.data_layout
    groups = attrs.groups
    kernel_layout = attrs.kernel_layout
    data, kernel = inputs

    if (groups == 1) and (layout == "NCHW"):
        assert kernel_layout == "OIHW"

        strategy.add_implementation(
            compute_conv2d_nchw(topi.nn.conv2d_nchw),
            topi_schedule_conv2d_nchw(topi.generic.schedule_conv2d_nchw),
            name="conv2d_nchw.generic",
        )
    else :
        return _strategy.x86.conv2d_strategy_cpu(attrs, inputs, out_type, target)
    return strategy
