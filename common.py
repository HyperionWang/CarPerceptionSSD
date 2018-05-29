BG_ID = 0
CAR_ID = 1
ROAD_ID = 2
n_classes = 3

input_height = 600
input_width = 800
image_shape = (input_height, input_width)

nw_height = 600
nw_width = 800
nw_shape = (nw_height, nw_width)

OFFSET_HIGH = 0
OFFSET_LOW = OFFSET_HIGH + nw_height

visualize = True
enable_profiling = True

model_path = 'ep-078-val_loss-0.9916.hdf5'
