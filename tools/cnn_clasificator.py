import numpy as np

import cv2
from keras.models import model_from_json

import prepare_dataset
import character_detection
import calculator

# Training dataset
dataset_path = "C:/Users/Stipe/PycharmProjects/PhotoMathTask/data/training"
# Evaluation dataset
eval_path = "C:/Users/Stipe/PycharmProjects/PhotoMathTask/data/evaluation"
# Add option to user define input image --parser
image_path = "C:/Users/Stipe/Downloads/img7.jpg"
TRAIN_MODEL = False
# Prepare dataset for training and testing
X_train, y_train, X_test, y_test, label_encoder = prepare_dataset.prepare_data(dataset_path, eval_path)

if TRAIN_MODEL:
    import train_model
    model = train_model.train_model_1(X_train, y_train, X_test, y_test)
    if model:
        print('Model successfully loaded')
    else:
        print("Error loading model")
else:
    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights("model.h5")
    print("Loaded model from disk")

original_img, binary_img = character_detection.read_img_and_convert_to_binary(image_path)
# Extract 1. cropped images [i]['src'] which can be viewed with plt.imshow(symbols[i]['src_img'], cmap='gray')
# of symbols and their 2. bounding boxes [i]['location']
symbols = character_detection.binary_img_segment(binary_img, original_img=original_img)
# Process images for the network and export a list of characters in form of tensors shape (1, 100, 100, 3)
list_of_characters = character_detection.preprocess_image_for_network(original_img, symbols)

predict = []
for character in list_of_characters:
    prediction = model.predict(character)
    result = np.argsort(prediction)
    result = result[0][::-1]
    final_label = label_encoder.inverse_transform(np.array(result))
    final_label = final_label[0]
    predict.append(final_label)
cv2.imshow('Original image:', original_img)
print('Prediction od elements: ', predict)
result = calculator.crazy_looping_calculus(predict)
if len(result) != 1:
    print('Error in calculation, please try taking another picture or rewrite the equation and try again')
print('Result of the equation is: ', result[0])



