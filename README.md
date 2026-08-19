# GEMM learning project

In this project I built a from-scratch GEMM algorithm specifically tailored for my (quite unconventional) orion CPU.
I learned a lot about how the bottleneck of this algorithm (and many others) isn't really the arithmetic operations but the low arithmetic density - 
moving a lot of data very inefficiently, starving ALU's.

First time directly building an algorithm to exploit L1, L2, L3 and registers in a way that makes it fast.
My slight advantage here was that even though the CPU i have has a small L3 (3mb per core) i have tons of incredibly fast xDDR5 mem.

I wanted to beat Eigen, and I did so Im happy bout that. 
I do acknowledge that if i spent more time tailoring Eigen itself for my architecture it probably would have performed better, but i just stuck to vanilla performance.

Current GFLOPS/s : 104

Eigen GFLOPS/S : 67 

target GFLOPS/S : ~80 


Beat Eigen on single thread performance.
Theoretical multi thread performance is ~633 GFLOPS/s, maybe i'll try doing that one day idk.
