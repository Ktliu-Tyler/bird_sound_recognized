# Bird Sound Recognized
#### Author Tyler, Hding
##
## Introduce
<h6/>
In this project, we want to use python to build a model that can recognize bird sound's differient between each term of a life cycle of a
Japanese White Eye.
<br />
<br />
We prepared 7 classes of Japanese White Eye's sound.
They are being labeled as before net, build net, fail, hug egg, out net, spawn and yu zhu gi.<br />

In order to concrete the bird sound recognize, we use python package such as librosa, tensorflow to proccess wav video and build our training model. Remember to install all package we need in your environment.<br />

Also we had built some useful tool to make the sound proccess easier,
you can use them not only to transform our bird audio but also use other .wav audio to train other kind of sound recognition model.<br />

Last but not last, we provide differience kind of code for notebook user and pycharm user. In the folder bird-sound-Tyler, there are .py file for pycharm or vsCode user, and in the colab folder are .ipynb file for notebook user, youcan run them on goolge colab or jupyter notebook.


## Install yuor package

|package|version|
|:----- |:--- |
|tensorflow|1.15.5|
|numpy|1.21.5|
|librosa|0.9.1|
|matplotlib|3.5.1|
|json||
|random||
|pickle||

## Instruction

<h6>
We have set up all train value for you in train_value.py. So you don't have to revise too much code to let it on track.
Below are some important  

To run all the code successfully, remember to change the .wav folder path and dataset path, or the system won't find the directory and report an error.

## In colab

<h6>
Running all the code, and Remember to check your tensorflow version is 1.15.5.
If you got the error in testing model, follow the debug cell to reinstall your h5py. And restart your notebook.

## In pycharm

<h6>
Running the code in main.py, and change the value in train_value.py if you want to create a new model with differient structure.Remember to check your tensorflow version is 1.15.5. Or you may got a over fitting model that always stock at 49% val_accuracy.


## All train value you can change
<h6>

* SOUND_PATH = ""
* DATASET_PATH = ""
* MODEL_NAME = "Model_500Epochs_v2"
* TIME_STEP = 20
* SOUND_LENGTH = 128
* TEST_DATA_RATIO = 0.2
* INPUT_SHAPE = (TIME_STEP, SOUND_LENGTH)
* CLASSES:
["before_net", "build_net", "fail", "hug_egg", "out_net", "spawn", "yu_zhu_gi"]

* MODEL_CONV1DS = [32, 32]
* MODEL_LAYERS = [64, 128, 0, 32, 64]

* BATCH_SIZE = 50
* EPOCHS = 100
