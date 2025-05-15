import tvm

class CallCounter:
    count = 0

# Match attribute with layer-specific string
def _match_pragma(key:str, stmt: tvm.tir.stmt.AttrStmt) -> bool:
    print("MATCH PRAGMA") 
    return (stmt.attr_key.startswith("pragma_" + key)) \
        or (stmt.attr_key == "pragma_scope" and stmt.value.value.startswith(key))


# Inject one funtion call before and one after the statement
def inject_mvm_call(stmt):
    print("MVM CALL")
    args = [tvm.tir.const(CallCounter.count, "int")]
    CallCounter.count += 1
    start_call = tvm.tir.call_extern("void", "my_test_start", *args)
    end_call = tvm.tir.call_extern("void", "my_test_end", *args)
    return tvm.tir.stmt_seq(start_call, stmt, end_call)


def inject_tracing_conv2d(): 
    print("INJECT") 
    def _postorder(stmt):
        # Find the statement with the "outerloop" pragma
        if _match_pragma(key="outerloop", stmt=stmt):
            return inject_mvm_call(stmt)
        return None

    def _ftransform(f, mod, ctx):
        stmt_in = f.body
        stmt = tvm.tir.stmt_functor.ir_transform(
            stmt_in, None, _postorder, ["tir.AttrStmt"])
        return f.with_body(stmt)

    return tvm.tir.transform.prim_func_pass(
        _ftransform,
        opt_level=2,
        name="outerloop")
