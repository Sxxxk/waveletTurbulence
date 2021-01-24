import numpy
import openvdb as vdb

file_in = "explosion.vdb"
file_out = "expOut.vdb"

# Access to the grids
grids = vdb.readAllGridMetadata(file_in)
grid_density = [x for x in grids if x.name == "density"][0]
grid_temp = [x for x in grids if x.name == "temperature"][0]
grid_velocity = [x for x in grids if x.name == "v"][0]

density_accessor = grid_density.getConstAccessor()
velocity_accessor = grid_velocity.getConstAccessor()

# Fill new grids
for i in range(264):
    for j in range(310):
        for k in range(270):
            ijk = (i, j, k)
            value = density_accessor.probeValue(ijk)[0]
            if value != 0.0:
                print(ijk, " - ", value)

del density_accessor
del velocity_accessor

# New grids
# new_grid_density = vdb.FloatGrid()
# new_grid_velocity = vdb.Vec3SGrid()

# new_density_accessor = grid_density.getAccessor()
# new_velocity_accessor = grid_velocity.getAccessor()

voxels = tiles = 0
N = 5
for item in grid_density.citerOffValues():  # read-only iterator
    if voxels == N and tiles == N:
        break
    if item.count == 1:
        if voxels < N:
            voxels += 1
            print('voxel', item.min)
    else:
        if tiles < N:
            tiles += 1
            print('tile', item.min, item.max)