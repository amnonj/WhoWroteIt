message_len = 384
patience = 20
min_delta = 0.005
max_epochs = 200

EMBEDDING_LENGTH = 64


CONV1_KERNEL = 9
CONV1_OUT = 256
CONV1_STRIDE = 1  # Padding calculation is now hardcoded for stride 1

DO_CONV2 = False
CONV2_KERNEL = 1
CONV2_OUT = 1
CONV2_STRIDE = 1  # Padding calculation is now hardcoded for stride 1

if DO_CONV2:
    FC_IN = CONV2_OUT
else:
    FC_IN = CONV1_OUT

POOL_REGIONS = 1
DROPOUT_RATIO = 0.5

SEED = 555
