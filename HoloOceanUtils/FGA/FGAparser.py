import math
import numpy as np
import linecache
from scipy.interpolate import RegularGridInterpolator

def interpolator_from_fga(filename):
    # get the dimensions
    dims = linecache.getline(filename, 1).split(",")
    dim_x = int(float(dims[0]))
    dim_y = int(float(dims[1]))
    dim_z = int(float(dims[2]))

    # get the min and max locations
    mins = linecache.getline(filename, 2).split(",")
    min_x = float(mins[0])
    min_y = float(mins[1])
    min_z = float(mins[2])

    maxs = linecache.getline(filename, 3).split(",")
    max_x = float(maxs[0])
    max_y = float(maxs[1])
    max_z = float(maxs[2])

    # create the numpy array
    fga_array = np.zeros((dim_x, dim_y, dim_z, 3))
    #print(fga_array)

    # fill the numpy array
    for x in range(dim_x):
        for y in range(dim_y):
            for z in range(dim_z):
                index = xyz_to_fga_index(x, y, z, dim_x, dim_y)
                line = linecache.getline(filename, index + 1).split(",")
                line = np.float_(line[0:-1])

                fga_array[x][y][z] = line

    x = np.linspace(min_x, max_x, dim_x)
    y = np.linspace(min_y, max_y, dim_y)
    z = np.linspace(min_z, max_z, dim_z)

    return RegularGridInterpolator((x, y, z), fga_array, bounds_error=False, fill_value=np.array([0.0, 0.0, 0.0]))

def xyz_to_fga_index(x, y, z, dim_x, dim_y):
    # index of the position in the fga file
    index = x + y * dim_x + z * dim_x * dim_y

    #add 3 to get past file header
    return index + 3

if __name__ == "__main__":
    #stuff = interpolator_from_fga("VF_Turbulence.fga")
    #print(interpolated_vector([0.0,0.0,0.0], stuff))
    interp = interpolator_from_fga("VF_Turbulence.fga")
    print(interp([80.0,79.0,80.0]))