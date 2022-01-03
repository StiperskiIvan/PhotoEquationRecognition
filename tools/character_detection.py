import cv2
import numpy as np
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt

IMG_SIZE = 100
LARGEST_NUMBER_OF_SYMBOLS = 20
SCALSIZE = 1


def read_img_and_convert_to_binary(filename):
    original_img = cv2.imread(filename)
    original_img = cv2.resize(original_img,
                              (np.int(original_img.shape[1]/SCALSIZE), np.int(original_img.shape[0]/SCALSIZE)),
                              interpolation=cv2.INTER_AREA)
    blur = cv2.GaussianBlur(original_img, (5, 5), 0)
    img_gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(img_gray, cv2.MORPH_OPEN, kernel)

    kernel2 = np.ones((3, 3), np.uint8)
    opening = cv2.dilate(opening, kernel2, iterations=1)
    # Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(opening, (13, 13), 0)
    ret, binary_img = cv2.threshold(blur, 0, 1, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
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
        if w*h < 100:
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


def test_image_extraction(file_path="C:/Users/Stipe/Downloads/img1.jpg"):
    original_img, binary_img = read_img_and_convert_to_binary(file_path)
    symbols = binary_img_segment(binary_img, original_img=original_img)
    list_of_ROI = []
    for i in range(len(symbols)):
        list_of_ROI.append(symbols[i]['location'])
    print(len(symbols))
    print(list_of_ROI[0])
    ROI_number = 0
    for item in list_of_ROI:
        x, y, w, h = item
        ROI = 255 - original_img[y:y+h, x:x+w]
        cv2.imwrite('ROI_{}.png'.format(ROI_number), ROI)
        ROI_number += 1

    plt.imshow(symbols[3]['src_img'], cmap='gray')
    plt.show()


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
        img_temp = original_img[y:y+h, x:x+w]
        img_temp = cv2.resize(img_temp, (100, 100))
        img_temp = np.array(img_temp)
        img_temp = np.expand_dims(img_temp, axis=0)
        img_temp = img_temp.astype('float32')
        img_temp /= 255.
        img.append(img_temp)
    return img


# test_image_extraction()
