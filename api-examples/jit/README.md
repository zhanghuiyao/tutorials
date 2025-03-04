
# JIT

## test simple function examples:

we define `funtion(a,b,c)=a*b+c`, and warp it by `mindspore.jit`, run by `GLOG_v=3 python -u simple_funtion.py`, result is follow:

| enable jit | jit level | capture mode | backend | fullgraph | *time to compile | *time to run thousand times | *time end to end(a thousand times) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| false | -     | -          | -             | -     | ~4.00s | ~0.09    | ~4.09     |
||
| true  | O0    | ast        | ms_backend    | false | ~0.28s | ~0.46s   | ~0.74s    |
| true  | O1    | ast        | ms_backend    | false | ~0.03s | ~0.45s   | ~0.48s    |
| true  | -     | ast        | ge            | false | ~0.90s | ~0.92s   | ~1.81s    |
||
| true  | O0    | bytecode   | ms_backend    | false | ~0.13s | ~0.64s   | ~0.76s    |
| true  | O1    | bytecode   | ms_backend    | false | ~0.00s | ~0.64s   | ~0.64s    |
| true  | -     | bytecode   | ge            | false | ~0.00s | ~0.65s   | ~0.65s    |
||
| true  | O0    | trace      | ms_backend    | false | ~0.18s | ~3.02s   | ~3.20s    |
| true  | O1    | trace      | ms_backend    | false | ~0.17s | ~2.94s   | ~3.11s    |
| true  | -     | trace      | ge            | false | ~0.19s | ~2.94s   | ~3.13s    |
||
| true  | O0    | ast        | ms_backend    | true  | ~0.03s | ~0.45s   | ~0.47s    |
| true  | O1    | ast        | ms_backend    | true  | ~0.02s | ~0.45s   | ~0.47s    |
| true  | -     | ast        | ge            | true  | ~0.13s | ~0.89s   | ~1.03s    |


> ⚠️ Note:
> 
> 0. the above results vary greatly for different devices and device states, data only for reference.
> 1. *time to compile, potential jitted object reuse may lead to inaccurate comparison.
> 2. *time to run thousand times, potential asynchronous execution operations may lead to inaccurate testing times.
> 3. *time end to end, due to the first and second points, the time may not be accurate.

