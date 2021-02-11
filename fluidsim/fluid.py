import pyopenvdb as vdb
from functools import reduce

grids, metadata = vdb.readAll('fluid_data_0190.vdb')

# Print grids
print("Grids :")
print(reduce(lambda a, b: a+b, ["{} : {}\n".format(i, grid.name) for i, grid in enumerate(grids)]))


dAccessor = grids[0].getConstAccessor()
vAccessor = grids[4].getConstAccessor()

# Test value
pv = (10, 10, 10)
print("valocity (10, 10, 10) : ", vAccessor.probeValue(pv))
print("density (10, 10, 10) : ", dAccessor.probeValue(pv))