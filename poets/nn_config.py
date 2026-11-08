message_len = 768
PATIENCE = 20
MIN_DELTA = 0.005
MAX_EPOCHS = 200
MAX_POST_REGIONS = 8
EPOCH_SIZE = 960


EMBEDDING_LENGTH = 64


CONV1_KERNEL = 9
CONV1_OUT = 128
CONV1_STRIDE = 1  # Padding calculation is now hardcoded for stride 1

DO_CONV2 = False
CONV2_KERNEL = 9
CONV2_OUT = 128
CONV2_STRIDE = 1  # Padding calculation is now hardcoded for stride 1

if DO_CONV2:
    FC_IN = CONV2_OUT
else:
    FC_IN = CONV1_OUT

POOL_REGIONS = 2
DROPOUT_RATIO = 0.5

SEED = range(111,556,111)
