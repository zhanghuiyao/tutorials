
simple function examples:

we define `funtion(a,b,c)=a*b+c`, and warp it by mindspore.jit, result is follow:

| enable jit | jit level | capture mode | backend | fullgraph | time to compile | time to run thousand times | time end to end(a thousand times) | 
| --- | --- | --- | --- | --- | --- | --- | --- | 
| false | - | -  
| true | O0 | ast | ms_backend | false | 0.25s | 0.45s | 