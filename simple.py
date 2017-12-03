simple.py
# simple code showing Python-CUDA binding

import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy

# kernel function (in 'C'); executes on device
mod = SourceModule("""
  __global__ void fill(int *dest_d, int *a_d)
  {
    int idx = threadIdx.x;
    dest_d[idx] = idx;
  }
  """)

# create array to work on
a = numpy.zeros(10, dtype=numpy.int32)
dest = numpy.empty_like(a)
print 'Original array:'
print dest

# allocate M on device
a_d = cuda.mem_alloc(a.nbytes)
dest_d = cuda.mem_alloc(dest.nbytes)
 
# copy input array from host to device (a_d <-- a)
cuda.memcpy_htod(a_d, a)

# find compiled 'C' function
func = mod.get_function("fill")

# call it with data argument(s) and size parameters
func(dest_d, a_d, block=(10,1,1))

# copy results from device to host (dest <-- dest_d)
cuda.memcpy_dtoh(dest, dest_d)
print 'Processed array:'
print dest


