from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from keras.preprocessing.image import ImageDataGenerator
import cv2
import os
import numpy as np
from random import Random,shuffle
from tqdm import tqdm

from keras import backend as K
import tensorflow as tf
from tensorflow.python.tools import freeze_graph, optimize_for_inference_lib

'''Setting up the env'''

### Edit Disini
DATA_DIR = 'data'            # Lokasi data gambar
IMG_SIZE = 64       # Ukuran gambar (u/ diperkecil)
LR = 0.07          # Learning Rate
no_batch = 1        # Banyaknya gambar u/ sekali proses
k = 10
chosen_test_k = 9   # Data test k yang terpilih (yang memiliki akurasi terbesar)
no_epoch = 30      # Banyaknya epoch
LABELS = {"100celeng", "25s75c", "50s50c", "75s25c", "100sapi"} # Nama-nama kelas
#################
rnd = Random(143)   # random seed
'''Setting up the model which will help with tensorflow models'''
MODEL_NAME = 'daging-{}-e{}-k{}.model'.format(LR, no_epoch, chosen_test_k)   # Nama model

'''Labelling the dataset'''
def label_img(img):
    word_label = img.split(' ')[-2]
    # DIY One hot encoder
    if word_label == '100celeng':
        return [1, 0, 0, 0, 0]
    elif word_label == '25s75c':
        return [0, 1, 0, 0, 0]
    elif word_label == '50s50c':
        return [0, 0, 1, 0, 0]
    elif word_label == '75s25c':
        return [0, 0, 0, 1, 0]
    elif word_label == '100sapi':
        return [0, 0, 0, 0, 1]

'''Export model to make it compatible with Android systems'''
def export_model(saver, model, input_node_names, output_node_name):
    tf.train.write_graph(K.get_session().graph_def, 'out', MODEL_NAME + '_graph.pbtxt')
    saver.save(K.get_session(), 'out/' + MODEL_NAME)

    with tf.Session() as session:
        my_saver = tf.train.import_meta_graph('out/' + MODEL_NAME + '.meta')
        my_saver.restore(session, tf.train.latest_checkpoint('out/'))
        frozen_graph = tf.graph_util.convert_variables_to_constants(
                            session,
                            session.graph_def,
                            ['dense_2/Sigmoid'])

        with open('frozen_model.pb', 'wb') as f:
            f.write(frozen_graph.SerializeToString())
    # freeze_graph.freeze_graph('out/' + MODEL_NAME + '_graph.pbtxt', None,
    #                             False, 'out/' + MODEL_NAME + '.chkp.index', output_node_name,
    #                             "save/restore_all", "save/Const:0",
    #                             'out/frozen_' + MODEL_NAME + '.pb', True, "")
    #
    # input_graph_def = tf.GraphDef()
    # with tf.gfile.Open('out/frozen_' + MODEL_NAME + '.pb', "rb") as f:
    #     input_graph_def.ParseFromString(f.read())
    #
    # output_graph_def = optimize_for_inference_lib.optimize_for_inference(
    #         input_graph_def, input_node_names, [output_node_name],
    #         tf.float32.as_datatype_enum)
    # with tf.gfile.FastGFile('out/opt_' + MODEL_NAME + '.pb', "wb") as f:
    #     f.write(output_graph_def.SerializeToString())
    print("graph saved!")

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        # Graph -> GraphDef ProtoBuf
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph

'''Creating the training data'''
def create_data():
    # Creating an empty list where we should the store the training data
    # after a little preprocessing of the data
    data = []
    labels = []

    # tqdm is only used for interactive loading
    # loading the training data
    for img in tqdm(os.listdir(DATA_DIR)):
        # labeling the images
        label = label_img(img)
        path = os.path.join(DATA_DIR, img)

        # loading the image from the path and then converting them into
        # greyscale for easier covnet prob
        img = cv2.imread(path)

        # resizing the image for processing them in the covnet
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        # final step-forming the training data list with numpy array of the images
        # training_data.append([np.array(img), np.array(label)])
        data.append(img)
        labels.append(label)

    # shuffling of the training data to preserve the random state of our data
    data = list(zip(data, labels))
    rnd.shuffle(data)
    data[:], labels[:] = zip(*data)

    # saving our trained data for further uses if required
    np.save('data.npy', data)
    np.save('labels.npy', labels)
    return data, labels

'''Split data into train and test (80 train-20 test)'''
# data, labels = create_data()         # Harus ada, baris ini atau baris dibawahnya
data, labels = np.load('data.npy'), np.load('labels.npy')

'''Split data using K-Fold Cross Validation'''
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
cv = KFold(n_splits=k, random_state=143, shuffle=False)

'''Creating the neural network using tensorflow'''
import keras
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, AveragePooling2D

## Kode CNN
classifier = Sequential()
# Conv2D('jumlah kernel', '(size kernel)', 'bentuk input', 'fungsi aktifasi')
classifier.add(Conv2D(32, (3,3), input_shape=(IMG_SIZE, IMG_SIZE, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Flatten())  # Proses Flattening
classifier.add(Dense(units = 128, activation = 'relu'))  # Jumlah node pada hidden layer
classifier.add(Dense(units = 5, activation = 'sigmoid')) # Jumlah node output (Jumlah kelas data)

i = 0
for train_index, test_index in cv.split(data): #[chosen_test_k-1]
    if i != chosen_test_k-1:
        # print(i)
        i = i+1
    else:
        trainX, testX, trainY, testY = data[train_index], data[test_index], labels[train_index], labels[test_index]
        # ### construct the training image generator for data augmentation
        # aug = ImageDataGenerator(rescale = 1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
        # train_aug = aug.flow(trainX, trainY, batch_size=no_batch)
        # aug = ImageDataGenerator(rescale = 1./255)
        # test_aug = aug.flow(testX, testY, batch_size=no_batch)
        #
        # classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
        # print("[INFO] compiling model..."+str(len(trainX)))
        # classifier.fit_generator(train_aug, steps_per_epoch = len(trainX)//no_batch, epochs = no_epoch, validation_data = test_aug, validation_steps=len(testX))
        #
        # classifier.save(MODEL_NAME)             # Proses untuk men-save model
        ############
        K.set_learning_phase(0)
        classifier = load_model(MODEL_NAME)   # Proses untuk me-load model
        ############
        print(classifier.inputs)
        print(classifier.outputs)
        export_model(tf.train.Saver(), classifier, ["conv2d_1_input"], "dense_2/Sigmoid")
        # frozen_graph = freeze_session(K.get_session(), output_names=[out.op.name for out in classifier.outputs])
        # tf.train.write_graph(frozen_graph, "model", MODEL_NAME + ".tf_model.pb", as_text=False)
        ## Making Predictions (PROSES UNTUK AKURASI DATA TESTING) ##
        predictions = classifier.predict(testX, batch_size=no_batch)
        # print(predictions)      # Hasil kelas dari Prediksi
        # if predictions[0][3] == 1:
        #     print("yea")
        # print(testY)            # Kelas aslinya (Data test)
        print(confusion_matrix(testY.argmax(axis=1),predictions.argmax(axis=1)))
        # ^ Kesimpulan hasil dari prediksi

        print("Accuracy score: ")
        print(accuracy_score(testY.argmax(axis=1),predictions.argmax(axis=1)))

        i = i+1

