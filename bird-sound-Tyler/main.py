import numpy as np
import tool
import train_value as value
import os
import model_creater

def main_test_wav(wav_path):
    model = model_creater.load(os.path.join(value.DATASET_PATH, "MODEL", value.MODEL_NAME))
    soundTool = tool.SoundProcessor()
    data = soundTool.load_sound(wav_path)
    mfccData = soundTool.instant_transform(data)
    if mfccData.shape[0] != np.array([]).shape[0]:
        predictions, classes = model_creater.predict_classes(model, mfccData)
        print("Prediction: ", max(classes))

def main_wav_process(mode):
    soundTool = tool.SoundProcessor()
    if mode == "npy":
        soundTool.bird_sound_all_classes_npy(value.SOUND_PATH, os.path.join(value.DATASET_PATH, "NPY"))
    elif mode == "json":
        soundTool.bird_sound_all_classes_json(value.SOUND_PATH, os.path.join(value.DATASET_PATH, "JSON"))

def data_loader(mode):
    soundTool = tool.SoundProcessor()
    if mode == "npy":
        x_train, y_train, x_test, y_test = soundTool.load_npy(value.DATASET_PATH)
    elif mode == "json":
        x_train, y_train, x_test, y_test = soundTool.load_json(value.DATASET_PATH)
    else:
        print("Please choice data type.")
        return
    return x_train, y_train, x_test, y_test


def plot_all(model_training_history):
  model_creater.plot_metric(model_training_history, "loss", "val_loss", "Train Loss Vs Train Val Loss",
        True, os.path.join(value.DATASET_PATH, "MODEL", value.MODEL_NAME + "_loss"))
  model_creater.plot_metric(model_training_history, "acc", "val_acc", "Train Acc Vs Train Val Acc",
        True, os.path.join(value.DATASET_PATH, "MODEL", value.MODEL_NAME + "_acc"))

def main_training(x_train, y_train, x_test, y_test):
    print("x_train: ", x_train.shape)
    print("y_train: ", y_train.shape)
    print("x_test: ", x_test.shape)
    print("y_test: ", y_test.shape)
    model = model_creater.get_model()
    model = model_creater.set(model)
    model, history = model_creater.fit(model,
                                       x_train, y_train,
                                       x_test, y_test,
                                       value.BATCH_SIZE,
                                       value.EPOCHS,
                                       )
    model_creater.save(model, os.path.join(value.DATASET_PATH, "MODEL"), value.MODEL_NAME)
    plot_all(history)

def main_testing(x_test, y_test):
    model = model_creater.load(os.path.join(value.DATASET_PATH, "MODEL", value.MODEL_NAME))
    model_creater.test(model, x_test, y_test, 150)



if __name__ == "__main__":
    # main_wav_process("npy")
    # main_wav_process("json")
    # x_train, y_train, x_test, y_test = data_loader("json")
    # main_training(x_train, y_train, x_test, y_test)
    # main_testing(x_test, y_test)
    wav_path = r"C:\Users\Liu Ty\Desktop\2022_Bird\japanese_white_eye\dataset\raw\hug_egg\0000.wav"
    main_test_wav(wav_path)



