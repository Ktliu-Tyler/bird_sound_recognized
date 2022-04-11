SOUND_PATH = r"C:\Users\Liu Ty\Desktop\2022_Bird\japanese_white_eye\dataset\raw"
DATASET_PATH = r"C:\Users\Liu Ty\Desktop\2022_Bird\japanese_white_eye\dataset\Tyler_data"

MODEL_NAME = "Model_500Epochs_v2"

TIME_STEP = 20
SOUND_LENGTH = 128
TEST_DATA_RATIO = 0.2
INPUT_SHAPE = (TIME_STEP, SOUND_LENGTH)
CLASSES = ["before_net", "build_net", "fail", "hug_egg", "out_net", "spawn", "yu_zhu_gi"]

MODEL_CONV1DS = [32, 32]
MODEL_LAYERS = [64, 128, 0, 32, 64]

BATCH_SIZE = 50
EPOCHS = 100
