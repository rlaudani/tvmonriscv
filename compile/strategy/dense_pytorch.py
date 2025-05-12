import tvm
from tvm import te
from tvm import topi
import tvm.relay.op as _op
import tvm.relay.op.strategy as _strategy

def default_compute(op) :
    return _strategy.generic.wrap_compute_dense(op)
def default_topi_schedule(sched) :
    return _strategy.generic.wrap_topi_schedule(sched)


def compute_dense(op) :
    return default_compute(op)


def topi_schedule_dense(topi_schedule) :
    def wrapper(attrs, outs, target):
        with target:
            s = te.create_schedule([x.op for x in outs])
            
            dense_op = [stge for stge in s.stages if stge.op.tag == 'dense']
            if len(dense_op) != 1 :
                raise Exception("Expected one conv2d operation in 'stages'.")
            dense_op = dense_op[0]
            
            if len(dense_op.all_iter_vars) == 3 :
                n, i, o = dense_op.all_iter_vars
                dense_op.pragma(var=n, pragma_type="outerloop")
            else :
                print("Conv2D NHWC has not enough axes due to optimization.")
            return s
    return wrapper


@_strategy.generic.dense_strategy.register(["cpu"])
def dense_strategy_xbar(attrs, inputs, out_type, target):
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        compute_dense(topi.nn.dense),
        topi_schedule_dense(topi.generic.schedule_dense),
        name="dense.generic",
    )
    return strategy

