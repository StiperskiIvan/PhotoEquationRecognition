import random
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.utils import np_utils
import os
import cv2
from sklearn import preprocessing


def prepare_data(training_path, eval_path):
    # Loading the folder paths of all testing and training images
    symbols_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-', '+', '=', 'div', 'times', '(', ')']
    train_image = []
    train_label = []
    number_of_images = []
    list_of_symbols = []

    for symbols_dir in os.listdir(training_path):
        if symbols_dir.split()[0] in symbols_list:
            number_of_classes = 0
            for image in os.listdir(training_path + "/" + symbols_dir):
                train_label.append(symbols_dir.split()[0])
                train_image.append(training_path + "/" + symbols_dir + "/" + image)
                number_of_classes += 1
        list_of_symbols.append(symbols_dir.split()[0])
        number_of_images.append(number_of_classes)

    print('Number of Training dataset classes:')
    for i in range(len(list_of_symbols)):
        print(list_of_symbols[i] + ': ' + str(number_of_images[i]))

    test_image = []
    test_label = []
    number_of_images = []
    number_of_classes = 0
    list_of_symbols = []

    for symbols_dir in os.listdir(eval_path):
        if symbols_dir.split()[0] in symbols_list:
            number_of_classes = 0
            for image in os.listdir(eval_path + "/" + symbols_dir):
                test_label.append(symbols_dir.split()[0])
                test_image.append(eval_path + "/" + symbols_dir + "/" + image)
                number_of_classes += 1
                list_of_symbols.append(symbols_dir.split()[0])
                number_of_images.append(number_of_classes)

            print('Number of Validation dataset classes:')
            for i in range(len(list_of_symbols)):
                print(list_of_symbols[i] + ': ' + str(number_of_images[i]))

    print("Length of train_image : ", len(train_image), " , length of labels list : ", len(train_label))
    print("Length of test_image : ", len(test_image), " , length of labels list : ", len(test_label))

    # Verifying the data
    # Let's see that we have 17 unique labels for both test and train

    unique_test = list(set(test_label))
    unique_train = list(set(train_label))
    print("Length of test unique labels: ", len(unique_test), " : ", unique_test)
    print("Length of train unique labels: ", len(unique_train), " : ", unique_train)


    # Loading the images and label and checking correctness
    random_number = random.randint(0, len(train_label))
    image = cv2.imread(train_image[random_number])
    plt.imshow(image)
    plt.title("Label: " + train_label[random_number])
    plt.show()

    random_number = random.randint(0, len(train_label))
    image = cv2.imread(train_image[random_number])
    plt.imshow(image)
    plt.title("Label: " + train_label[random_number])
    plt.show()

    # Creating train test and validation set

    test = np.array(cv2.imread(train_image[20]))
    print(test.shape)

    # Creating the X_train and X_test

    X_train = []
    X_test = []

    # laoding the images from the path
    for path in train_image:
        img = cv2.imread(path)
        img = cv2.resize(img, (100, 100))
        img = np.array(img)
        X_train.append(img)

    for path in test_image:
        img = cv2.imread(path)
        img = cv2.resize(img, (100, 100))
        img = np.array(img)
        X_test.append(img)

    # creating numpy array from the images
    X_train = np.array(X_train)
    X_test = np.array(X_test)


    # Verifying the shape

    print(X_train.shape)

    # normalizing the data
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    print("X_train shape: ", X_train.shape)
    print("X_test shape: ", X_test.shape)


    # Creating the y_train and y_test

    # label encoding the 17 symbols
    label_encoder = preprocessing.LabelEncoder()
    y_train_temp = label_encoder.fit_transform(train_label)
    y_test_temp = label_encoder.fit_transform(test_label)

    print("y_train_temp shape: ", y_train_temp.shape)
    print("y_test_temp shape: ", y_test_temp.shape)

    # creating matrix labels list
    y_train = keras.utils.np_utils.to_categorical(y_train_temp, 17)
    y_test = keras.utils.np_utils.to_categorical(y_test_temp, 17)


    print("y_train shape: ", y_train.shape)
    print("y_test shape: ", y_test.shape)
    return X_train, y_train, X_test, y_test, label_encoder


