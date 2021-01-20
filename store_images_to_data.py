from PIL import Image
import glob
import cv2
from random import Random
import numpy as np
from tqdm import tqdm

'''Labelling the dataset'''
def label_img(word_label):
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

IMG_SIZE = 128               # Ukuran gambar (u/ diperkecil)
LABELS = ["100celeng", "25s75c", "50s50c", "75s25c", "100sapi"] # Nama-nama kelas
rnd = Random(777)

# image_list = []
data = []
labels = []
j=0 ; k=0
for i in range(len(LABELS)):
    print("Loading images from folder \""+LABELS[i]+"\"... Please wait")
    for filename in tqdm(glob.glob('test\\'+LABELS[i]+'/*.jpg')):
        label = label_img(LABELS[i])
        # greyscale for easier covnet prob
        img = cv2.imread(filename)
        # resizing the image for processing them in the covnet
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        # final step-forming the training data list with numpy array of the images
        data.append(img)
        labels.append(label)

        ### Rotating images (90 - 180 - 270)
        center = (IMG_SIZE/2, IMG_SIZE/2)           # calculate the center of the image for rotation
        for j in range(3):
            # rotated_img = cv2.rotate(img, cv2)
            img_tr = cv2.getRotationMatrix2D(center, 90*(i+1), 1.0)
            rotated_img = cv2.warpAffine(img, img_tr, (IMG_SIZE, IMG_SIZE))
            data.append(rotated_img)
            labels.append(label)
            # print(j)

        ### Flipping images (Flip Vertically - Flip Horizontally)
        for k in range(2):
            flipped_img = cv2.flip(img, k)
            data.append(flipped_img)
            labels.append(label)
            # print(k)

print(len(data))

# shuffling of the training data to preserve the random state of our data
data = list(zip(data, labels))
rnd.shuffle(data)
data[:], labels[:] = zip(*data)

# saving our trained data for further uses if required
np.save('test_data_128.npy', data)
np.save('test_labels_128.npy', labels)


