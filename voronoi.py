#!/usr/bin/python

# voronoi.py
# Rob Sanchez
# PyCUDA version of Voronoi
# Programming assignment #3
# CIS 677, F2017

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from timeit import default_timer as timer
from PIL import Image
import numpy as np
import sys

mod = SourceModule("""
    __global__ void closest_euc(int *x1_d, int *y1_d, int nseeds,
                                int *x2_d, int *y2_d, int img_width, int img_height,
                                int *csx_d, int *csy_d)
    {
        const int BLOCK_SIZE = 32;
        const float MAX_DIST = 99999;
        const int x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
        float closest_dist = MAX_DIST; 

        for (int i=0; i < nseeds; i++) {
            int x_diff = x2_d[x] - x1_d[i] * x2_d[x] - x1_d[i];
            int y_diff = y2_d[x] - y1_d[i] * y2_d[x] - y1_d[i];
            float cur_dist = sqrtf( x_diff + y_diff );
            if (cur_dist < closest_dist) {
                closest_dist = cur_dist;
                csx_d[x] = x1_d[i];
                csy_d[x] = y1_d[i];
            }
        }
    }
    """)

BLOCK_SIZE = 32

def main():
    if len(sys.argv) != 4:
        print "incorrect number of arguments! \n\tusage: ./voronoi.py <seeds> <width> <height>\n" 
        sys.exit(1);

        # process command-line arguments
    num_seeds  = int(sys.argv[1])
    IMG_WIDTH  = int(sys.argv[2])
    IMG_HEIGHT = int(sys.argv[3])

        # instantiate the timer

        # create Euclidean and Manhattan bitmaps
    voronoi_euclidean = Image.new('RGB', (IMG_WIDTH, IMG_HEIGHT), "rgb(0, 0, 0)")
    voronoi_manhattan = Image.new('RGB', (IMG_WIDTH, IMG_HEIGHT), "rgb(0, 0, 0)")

        # generate and sort the image seeds
    seed_x_coords = np.random.randint(IMG_WIDTH,  size=num_seeds, dtype=np.int32)
    seed_y_coords = np.random.randint(IMG_HEIGHT, size=num_seeds, dtype=np.int32)

        # set up vector of image pixels
    img_x_coords = np.arange(IMG_WIDTH,  dtype=np.int32)
    img_y_coords = np.arange(IMG_HEIGHT, dtype=np.int32)

        # set up color map (dictionary)
    color_map = dict()
    for i in range(len(seed_x_coords)):
        color_map[(seed_x_coords[i], seed_y_coords[i])] = (np.random.randint(256), 
                                                           np.random.randint(256), 
                                                           np.random.randint(256))

        # create target vectors for closest seed results
    closest_seed_x = np.zeros(IMG_WIDTH * IMG_HEIGHT,  dtype=np.int32)
    closest_seed_y = np.zeros(IMG_HEIGHT * IMG_WIDTH,  dtype=np.int32)

        # find compiled C function
    func_d = mod.get_function("closest_euc")

        # call with data argument and size params
    func_d(cuda.InOut(seed_x_coords, seed_y_coords, num_seeds, 
                      img_x_coords, img_y_coords, IMG_WIDTH, IMG_HEIGHT,
                      closest_seed_x, closest_seed_y), block=(BLOCK_SIZE, 1, 1), 
                                                       grid=(IMG_WIDTH * IMG_HEIGHT / BLOCKSIZE, 1, 1))
        
        # print the new arrays
    print closest_seed_x
    print closest_seed_y
        


if __name__ == '__main__':
    main()
