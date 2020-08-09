
# -------------------------
# needed imports
# ---------------------------

import cv2
import scipy.spatial.distance as dist
from matplotlib import pyplot as plt
import numpy as np
import copy
from myfunctions import show_image


# -------------------------
# custom functions
# ---------------------------

def check_number_of_channels(image, template):
    color = ('b', 'g', 'r')
    # plotting the template with it's histogram - if image is in grayscale with only have one channel histogram
    # if image is rgb we have 3 different histograms

    if len(image.shape) < 3:
        channels = 1
    elif len(image.shape) == 3:
        channels = 3

    plt.figure()
    plt.subplot(211),

    for channel in range(channels):
        hist = cv2.calcHist([template], [channel], None, [256], [0, 256]) / (template.shape[0] * template.shape[1])
        if channel == 0:
            histogram_1 = hist
        if channel == 1:
            histogram_2 = hist
        if channel == 2:
            histogram_3 = hist
        plt.plot(hist, color=color[channel])
        plt.xlim([0, 256])

    plt.subplot(212)

    if channels == 1:
        histogram = [histogram_1]
        plt.imshow(template, cmap=plt.get_cmap('gray'))
    else:
        histogram = [histogram_1, histogram_2, histogram_3]
        plt.imshow(cv2.cvtColor(template, cv2.COLOR_BGR2RGB))
    plt.show()

    return histogram, channels


def image_patch_size(template):
    (template_height, template_width) = template.shape[0:2]
    # getting the two halves of the template image
    left_side_width = template_width // 2
    right_side_width = template_width // 2
    left_side_height = template_height // 2
    right_side_height = template_height // 2

    # in case of the width or height being even take one pixel of the right side of the template f0or it to
    # be the center pixel
    if np.mod(template_width, 2) == 0:
        right_side_width = right_side_width - 1
    if np.mod(template_height, 2) == 0:
        right_side_height = right_side_height - 1

    return left_side_width, right_side_width, left_side_height, right_side_height

# pattern matching using histogram similarity of image patches function
def color_histogram_similarity(image, template, threshold, method):
    image_rectangles = copy.copy(image)  # image needs to be copied so we won't draw on the original
    rect = np.zeros(image.shape, np.uint8)  # temporary black image used for the drawn rectangles on matches
    (image_height, image_width) = image.shape[0:2]

    histogram, channels = check_number_of_channels(image, template)
    left_side_width, right_side_width, left_side_height, right_side_height = image_patch_size(template)

    # scan the whole image starting from the pixel that is left_side_width and left_side_height away from the first pixel
    # till the right_size_width and wight_size_height pixel away from the last one of the image
    for j in range(left_side_width, image_width - right_side_width + 1):
        for i in range(left_side_height, image_height - right_side_height + 1):
            # depending on the number of channels of the image get a slice of an image
            if channels == 1:
                image_slice = image[i - left_side_height:i + right_side_height,
                              j - left_side_width:j + right_side_width]
            else:
                image_slice = image[i - left_side_height:i + right_side_height,
                              j - left_side_width:j + right_side_width, :]
            distance = []
            (slice_height, slice_width) = image_slice.shape[0:2]
            size_slice = slice_height * slice_width
            # for all the channels 1 or 3
            for channel in range(channels):
                # calculate the histogram of the image slice - normalized
                temp_histogram = cv2.calcHist([image_slice], [channel], None, [256], [0, 256]) / size_slice
                # depending on the method used
                if method[0] == "Manhattan":  # if method to use is Cityblock
                    dist = np.tanh(method[1](histogram[channel], temp_histogram))  # scale 0 to 1 (use of tanh for that)
                elif method[0] == "Intersection":  # if method to use is Intersection
                    dist = cv2.compareHist(histogram[channel], temp_histogram, method[1])  # scale 0 to 1
                distance = np.append(distance, dist)
            distance_flag = 0
            # check all channels distances
            for d in range(len(distance)):
                if method[0] == "Manhattan":  # if method used was Cityblock
                    if distance[d] <= threshold:  # if the distance wasn't greater than the threshold add one
                        distance_flag += 1
                elif method[0] == "Intersection":  # if method to use was Intersection
                    if distance[d] >= threshold:  # if the distance was greater or equal to the threshold add one
                        distance_flag += 1

            if distance_flag == len(distance):  # if all channels were better than the threshold draw rectangle
                cv2.rectangle(rect, (j - left_side_width, i - left_side_height),
                              (j + right_side_width, i + right_side_height), (191, 74, 211), 2)
    # on the image add the image with the rectangles with opacity 0.45
    image_rectangles = cv2.addWeighted(image_rectangles, 1.0, rect, 0.45, 1)
    return image_rectangles


# either an rgb or an grayscale image - works for both - use 1 for rgb and 0 for greyscale
original_image = cv2.imread("images/photo2.jpg", 1)  # read whole image
# load the template image we look for
template_image = cv2.imread("images/frog.jpg", 1)  # read pattern to find

methods = (("Intersection", cv2.HISTCMP_INTERSECT),
           ("Manhattan", dist.cityblock))
# for each method
for sim_method in range(len(methods)):
    thres = float(input("Enter threshold : "))
    image_rectangles = color_histogram_similarity(original_image, template_image, thres, methods[sim_method])
    show_image("Image-" + methods[sim_method][0], image_rectangles)
cv2.destroyAllWindows()
