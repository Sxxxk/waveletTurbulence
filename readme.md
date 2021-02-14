# Wavelet turbulence

## Requirements
### OpenVDB
Refer to instructions found there : https://pypi.org/project/pyopenvdb/ (it is not only pip install pyopenvdb)

### numpy
pip install numpy

### pywt
pip install pyWavelets

## Start the project

./smoke.py
(will transform fluid_data_0190.vdb to make it 3times its base resolution)

You can view the files in blender for example (drag & drop the file in blender).

## others
 - Pyopenvdb seems to not yet support all .vdb files. It is however able to open the ones created by blender, like for example fluid_data_0190.vdb.
 - noiseTile is a noise tile (thanks captain obvious) generated using the author's implementation. However, its size limits us to upgrade simulations
 to a resolution up to 128x128x128. We took this size because it is enough for the purpose of this project, and is already long enough to compute.
