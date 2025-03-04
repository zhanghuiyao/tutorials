import time
import numpy as np

import mindspore
from mindspore import jit, ops, nn, Tensor


f_input = [Tensor(np.full((2, 3), i), mindspore.float32) for i in range(3)]

print(f"function input: {f_input}")

def f(a, b, c):
    return a * b + c

jitted_defult_f = jit(f)
# jitted_by_ast_and_levelO0_f = jit(f, capture_mode="ast", jit_level="O0")
jitted_by_ast_and_levelO1_f = jit(f, capture_mode="ast", jit_level="O1")
jitted_by_ast_and_ge_f = jit(f, capture_mode="ast", backend="GE")


jitted_by_bytecode_and_levelO0_f = jit(f, capture_mode="bytecode", jit_level="O0")
jitted_by_bytecode_and_levelO1_f = jit(f, capture_mode="bytecode", jit_level="O1")
jitted_by_bytecode_and_ge_f = jit(f, capture_mode="bytecode", backend="GE")


######################################################################################################
# !!! jit by trace must be use the registry method !!!

# jitted_by_trace_and_levelO0_f = jit(f, capture_mode="trace", jit_level="O0")
# jitted_by_trace_and_levelO1_f = jit(f, capture_mode="trace", jit_level="O1")
# jitted_by_trace_and_ge_f = jit(f, capture_mode="trace", backend="GE")


# 
# @jit(capture_mode="trace", jit_level="O0")
# def jitted_by_trace_and_levelO0_f(*input):
#     return f(*input)

@jit(capture_mode="trace", jit_level="O0")
def jitted_by_trace_and_levelO0_f(a, b, c):
    return a * b + c

@jit(capture_mode="trace", jit_level="O1")
def jitted_by_trace_and_levelO1_f(a, b, c):
    return a * b + c

@jit(capture_mode="trace", backend="GE")
def jitted_by_trace_and_ge_f(a, b, c):
    return a * b + c
######################################################################################################


jitted_by_ast_and_levelO0_fullgraph_f = jit(f, capture_mode="ast", jit_level="O0", fullgraph=True)
jitted_by_ast_and_levelO1_fullgraph_f = jit(f, capture_mode="ast", jit_level="O1", fullgraph=True)
jitted_by_ast_and_ge_fullgraph_f = jit(f, capture_mode="ast", backend="GE", fullgraph=True)



function_dict = {
    "function": f,
    
    "function jitted by ast and levelO0": jitted_defult_f,
    "function jitted by ast and levelO1": jitted_by_ast_and_levelO1_f,
    "function jitted by ast and ge": jitted_by_ast_and_ge_f,
    
    "function jitted by bytecode and levelO0": jitted_by_bytecode_and_levelO0_f,
    "function jitted by bytecode and levelO1": jitted_by_bytecode_and_levelO1_f,
    "function jitted by bytecode and ge": jitted_by_bytecode_and_ge_f,

    "function jitted by trace and levelO0": jitted_by_trace_and_levelO0_f,
    "function jitted by trace and levelO1": jitted_by_trace_and_levelO1_f,
    "function jitted by trace and ge": jitted_by_trace_and_ge_f,

    "function jitted by ast and levelO0 fullgraph": jitted_by_ast_and_levelO0_fullgraph_f,
    "function jitted by ast and levelO1 fullgraph": jitted_by_ast_and_levelO1_fullgraph_f,
    "function jitted by ast and ge fullgraph": jitted_by_ast_and_ge_fullgraph_f
}



# compare time cost
for s, f in function_dict.items():
    s_time = time.time()
    
    out = f(*f_input)
    
    time_to_compile = time.time() - s_time
    s_time = time.time()

    for _ in range(1000):
        out = f(*f_input)
    
    time_to_run_thousand_times = time.time() - s_time

    print(f"{s}, time to compile: {time_to_compile:.2f}s, time to run thousand times: {time_to_run_thousand_times:.2f}s, time end to end(a thousand times): {time_to_compile+time_to_run_thousand_times:.2f}")

