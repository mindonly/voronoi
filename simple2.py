# simpleV2.py
# simple code showing alternative syntax

import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy

# kernel function (in 'C'); executes on device
mod = SourceModule("""
  __global__ void fill(int *a_d)
  {
    int idx = threadIdx.x;
    a_d[idx] = idx;
  }
  """)

# create array to work on
a = numpy.zeros(10, dtype=numpy.int32)
print 'Original array:'
print a

# find compiled 'C' function
func_d = mod.get_function("fill")

# call it with data argument and size parameters
func_d(cuda.InOut(a), block=(10, 1, 1))

print 'Processed array:'
print a

