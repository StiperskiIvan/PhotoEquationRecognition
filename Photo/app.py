import os

from keras.models import model_from_json
import operator

from flask import Flask, request, render_template

import cv2
import numpy as np
from skimage.morphology import skeletonize
from werkzeug.utils import secure_filename

IMG_SIZE = 100
LARGEST_NUMBER_OF_SYMBOLS = 20
SCALESIZE = 1
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def read_img_and_convert_to_binary(filename):
    original_img = cv2.imread(filename)
    original_img = cv2.resize(original_img,
                              (np.int(original_img.shape[1] / SCALESIZE), np.int(original_img.shape[0] / SCALESIZE)),
                              interpolation=cv2.INTER_AREA)
    blur = cv2.GaussianBlur(original_img, (5, 5), 0)
    img_gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(img_gray, cv2.MORPH_OPEN, kernel)

    kernel2 = np.ones((3, 3), np.uint8)
    opening = cv2.dilate(opening, kernel2, iterations=1)
    # Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(opening, (13, 13), 0)
    ret, binary_img = cv2.threshold(blur, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return original_img, binary_img


def extract_img(location, img, contour=None):
    x, y, w, h = location
    if contour is None:
        extracted_img = img[y:y + h, x:x + w]
    else:
        mask = np.zeros(img.shape, np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, cv2.FILLED)
        img_after_masked = cv2.bitwise_and(mask, img)
        extracted_img = img_after_masked[y:y + h, x:x + w]
    black = np.zeros((IMG_SIZE, IMG_SIZE), np.uint8)
    if w > h:
        res = cv2.resize(extracted_img, (IMG_SIZE, int(h * IMG_SIZE / w)), interpolation=cv2.INTER_AREA)
        d = int(abs(res.shape[0] - res.shape[1]) / 2)
        black[d:res.shape[0] + d, 0:res.shape[1]] = res
    else:
        res = cv2.resize(extracted_img, ((int)(w * IMG_SIZE / h), IMG_SIZE), interpolation=cv2.INTER_AREA)
        d = int(abs(res.shape[0] - res.shape[1]) / 2)
        black[0:res.shape[0], d:res.shape[1] + d] = res
    extracted_img = skeletonize(black)
    extracted_img = np.logical_not(extracted_img)
    return extracted_img


def binary_img_segment(binary_img, original_img=None):
    img, contours, hierarchy = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > LARGEST_NUMBER_OF_SYMBOLS:
        raise ValueError('something went wrong try writing the equation again or the number of characters is > 20 ')
    symbol_segment_location = []
    symbol_segment_list = []

    for contour in contours:
        location = cv2.boundingRect(contour)
        x, y, w, h = location
        if w * h < 100:
            continue
        symbol_segment_location.append(location)
        extracted_img = extract_img(location, img, contour)
        symbol_segment_list.append(extracted_img)
        if len(original_img):
            cv2.rectangle(original_img, (x, y), (x + w, y + h), (0, 0, 255), 3)
        symbols = []
        for i in range(len(symbol_segment_location)):
            symbols.append({'location': symbol_segment_location[i], 'src_img': symbol_segment_list[i]})
        symbols.sort(key=lambda k: k['location'][0])

    return symbols


def preprocess_image_for_network(original_img, symbols):
    """
    Prepares image for network input
    :param original_img: original equation image
    :param symbols: list containing extracted character location and binary images
    :return: tensor shape (1,100,100,3)
    """
    img = []
    for i in range(len(symbols)):
        x, y, w, h = symbols[i]['location']
        img_temp = original_img[y:y + h, x:x + w]
        img_temp = cv2.resize(img_temp, (100, 100))
        img_temp = np.array(img_temp)
        img_temp = np.expand_dims(img_temp, axis=0)
        img_temp = img_temp.astype('float32')
        img_temp /= 255.
        img.append(img_temp)
    return img


def merge_numbers(list_of_numbers):
    return ''.join(list_of_numbers)


def use_operators(op1, oper, op2):
    ops = {
        '+': operator.add,
        '-': operator.sub,
        'times': operator.mul,
        'div': operator.truediv
    }
    op1, op2 = int(op1), int(op2)
    return ops[oper](op1, op2)


def add_negative_numbers(expression):
    for j in range(len(expression)):
        if expression[j] == '-':
            if expression[j + 1] == '(':
                continue
            else:
                expression[j + 1] = merge_numbers([expression[j], expression[j + 1]])
                expression[j] = '+'
    return expression


def calculate(expression):
    # check the symbols in the expression and calculate until the len of expression is 1
    high_order_operators = []
    low_order_operators = []

    for i in range(len(expression)):
        # assume there is a digit before and after the expression
        if expression[i] == 'div' or expression[i] == 'times':
            high_order_operators.append(i)
        if expression[i] == '+' or expression[i] == '-':
            low_order_operators.append(i)

    new_list_of_expressions_high = expression
    high = False

    if not len(high_order_operators) and not len(low_order_operators):
        result = expression
        return result
    else:
        if len(high_order_operators):
            high = True
            for j in high_order_operators:
                list_to_delete_high = []
                new_result = str(use_operators(expression[j - 1], expression[j], expression[j + 1]))
                new_list_of_expressions_high[j] = new_result
                list_to_delete_high.append(j - 1)
                list_to_delete_high.append(j + 1)
                for item in reversed(list_to_delete_high):
                    del new_list_of_expressions_high[item]

        if len(low_order_operators):
            if high:
                low_order_operators = []
                expression = new_list_of_expressions_high
                new_list_of_expressions_low = new_list_of_expressions_high
                for i in range(len(expression)):
                    if expression[i] == '+' or expression[i] == '-':
                        low_order_operators.append(i)
                for j in reversed(low_order_operators):
                    list_to_delete_low = []
                    new_result = str(use_operators(expression[j - 1], expression[j], expression[j + 1]))
                    new_list_of_expressions_low[j] = new_result
                    list_to_delete_low.append(j - 1)
                    list_to_delete_low.append(j + 1)
                    for item in reversed(list_to_delete_low):
                        del new_list_of_expressions_low[item]
                result = new_list_of_expressions_low
                return result
            else:
                new_list_of_expressions_low = expression
                for j in reversed(low_order_operators):
                    list_to_delete_low = []
                    new_result = str(use_operators(expression[j - 1], expression[j], expression[j + 1]))
                    new_list_of_expressions_low[j] = new_result
                    list_to_delete_low.append(j - 1)
                    list_to_delete_low.append(j + 1)
                    for item in reversed(list_to_delete_low):
                        del new_list_of_expressions_low[item]

                result = new_list_of_expressions_low
                return result

        else:
            result = new_list_of_expressions_high
            return result


def create_big_numbers(expression):
    """
    find if elements contain numbers bigger than 9
    :param expression: original expression
    :return: new expression
    """
    new_expression = []
    temp = []
    for i in range(len(expression)):
        if expression[i].isdigit():
            temp.append(expression[i])
            if i + 1 == len(expression):
                if len(temp) > 1:
                    new_expression.append(merge_numbers(temp))
                else:
                    new_expression.append(temp[0])
        else:
            # if list is not empty
            if len(temp):
                # needs to be merged first
                if len(temp) > 1:
                    new_expression.append(merge_numbers(temp))
                else:
                    new_expression.append(temp[0])
                temp = []
                new_expression.append(expression[i])
            else:
                new_expression.append(expression[i])
    new_expression = add_negative_numbers(new_expression)
    return new_expression


def find_parentheses(expression):
    """
    find if elements contain parentheses, if they do save their indexes and extract that expression, if not continue.
    if the size of '(' is different from size od ')' print out warning that user should either enter newly written
    expression or take a better picture of the equation.
    break from calculation with an error
    :param expression: original expression
    :return: new expression
    """
    new_expression = []
    index_left = []
    index_right = []
    for i in range(len(expression)):
        if expression[i] == '(':
            index_left.append(i)
        elif expression[i] == ')':
            index_right.append(i)
    # Check if parentheses were found
    if not len(index_left) and not len(index_right):
        new_expression = expression
        if new_expression[0] in ['+', 'times', 'div']:
            del new_expression[0]
        new_expression = calculate(new_expression)
    elif len(index_left) == len(index_right):
        inner_expression = expression[index_left[0] + 1:index_right[0]]
        if inner_expression[0] in ['+', 'times', 'div']:
            del inner_expression[0]
            index_left -= 1
            index_right -= 1
        result_inner_expression = calculate(inner_expression)
        new_expression = expression
        new_expression[index_left[0]] = result_inner_expression[0]
        del new_expression[index_left[0] + 1:index_right[0] + 1]
        if new_expression[index_left[0] - 1].isdigit():
            new_expression.insert(index_left[0], 'times')
        if new_expression[0] in ['+', 'times', 'div']:
            del new_expression[0]
        new_expression = calculate(new_expression)

    elif len(index_left) != len(index_right):
        print('Error in the equation, odd number of parentheses, take another picture, '
              'or write the equation again please')
    return new_expression


def crazy_looping_calculus(expression):
    try:
        exp = create_big_numbers(expression)
        exp = find_parentheses(exp)
        return exp
    except:
        exp = []
        return exp


def get_predicted_list(list, model):
    predict = []
    list_of_classes = ['(', ')', '+', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '=', 'div', 'times']
    for character in list:
        prediction = model.predict(character)
        result = np.argsort(prediction)
        result = result[0][::-1]
        final_label = list_of_classes[result[0]]
        predict.append(final_label)
    return predict


app = Flask(__name__)

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("model.h5")


@app.route('/')
def main():
    return render_template("index.html")


@app.route("/uploader", methods=['POST'])
def classify_image():
    predict = []
    result = ''
    if request.method == 'POST':
        f = request.files['file']
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'static', 'uploads', secure_filename(f.filename))
        f.save(file_path)
        original_img, binary_img = read_img_and_convert_to_binary(str(file_path))
        symbols = binary_img_segment(binary_img, original_img=original_img)
        list_of_characters = preprocess_image_for_network(original_img, symbols)

        predict = get_predicted_list(list_of_characters, model)
        result = crazy_looping_calculus(predict)
        if len(result) != 1:
            result = ['Error in calculation, please try taking another picture ' \
                      'or rewrite the equation and try again']

    return render_template("upload.html", predictions=result[0], display_image=f.filename)


if __name__ == ' __main__ ':
    app.run(host="0.0.0.0", port="5000", ssl_context='adhoc', debug=False)
