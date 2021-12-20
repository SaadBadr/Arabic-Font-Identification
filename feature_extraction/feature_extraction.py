#!/usr/bin/env python
# coding: utf-8

# Hello everyone,
# 
# In this module, it is required to implement the feature extractor for Arabic Font Identification System.
# 
# At first, what are the main steps that we should go through in this module?
# 
# 
# # TODOs:
# 
# 1. Understand the problem
# 2. ..
# 3. Testing

# # 1. Understand the problem
# 
# Calligraphers use some specific features, The main aim of this step is to transform these features into values that can be fed to a machine so it can decide on the style of a given text image. It should be mentioned that a feature might specify one or more styles. Features designated for each style or set of styles might be used in a sequential manner (sequential
# decision) or parallel manner.
# 
# 
# 
# - Input: --
# - Output: --

# In[14]:


##################################################### imports #####################################################
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import hog


# # 2. Text thickness (Tth)
# Stroke thickness plays an important role in defining the style. Some styles use a flat pen, whereas some others use a pointed one. In some styles, calligraphers alter the thickness while writing (via pushing down the pen or the opposite), whereas in others, the thickness is always preserved. Modeling such a feature in form of a descriptor will help the machine to understand more specificities of each style. Text thickness (Tth) descriptor codifies the appearance frequency of different line thicknesses in a text image. To extract this descriptor, we employ both the skeleton and edge image, and thickness is determined by the distance between skeleton and edges.

# In[15]:


def Tth(skeleton_img, edge_img, bins=5):
    """Gets normalized histogram of text thickness 

    Args:
        skeleton_img: the image extracted skeleton
        edge_img: the image extracted edges
        
    Returns:
        hist : 1d array contains the normalized histogram values
        bin_edges : 1d array contains the bin edges values
    """

    skeleton = skeleton_img.copy()
    edge = edge_img.copy()

    # convert img to uint8 with 255:white 0:black

    if skeleton.max() == 1:
        skeleton = skeleton * 255
        skeleton = skeleton.astype(np.uint8)

    if edge.max() == 1:
        edge = edge * 255
        edge = edge.astype(np.uint8)

    # flipping edge => black text on white background
    edge = 255 - edge

    distance = cv2.distanceTransform(edge+skeleton, distanceType=cv2.DIST_L2, maskSize=5)
    # keeping only skeleton distances
    distance[skeleton==0] = 0
    
    h, h_bins = np.histogram(distance, range=(1, distance.max()), density=False, bins=bins)

    # normalizing the histogram
    h_sum = h.sum()
    if h_sum > 1:
        h = h/h_sum

    return h


# # 3. Special Diacritics (SDs)
# Thuluth and Mohakik have a similar writing style that is decorated with diacritics having special shapes, as shown in Figure 9:
# <img src="sds.png" alt="Figure 9" width="100"/>
# An SDs descriptor will be used to inspect the existence of such diacritics in a given text image. To this end, HuMoments is calculated for the contours of the two diacritics in Figure 9.
# Thereafter,  a score is calculated for a test image by calculating the distance between the two pre-calculated HuMoments and the test image diacritics contours HuMoments.
# 

# In[16]:



def precalculate_hu(path="sds.png", save=False, save_filename="sds_hue_moments"):
    """Utility function used by SDs to calculate hu moments for each contour, from one image contains diactritics
        
    Args:
        path: the path for the diactritics image
        save: if true, it will store the calculated values in sds_hue_moments
        save_filename: the filename in case of save=true, (default = sds_hue_moments)
    Returns:
        hu_moments: the calculated hu moments
    """

    # importing binarize from preprocessing module
    import sys
    sys.path.insert(1, "./../preprocessing/")
    from preprocessing import binarize

    sds_img = binarize(cv2.imread(path, 0))
    
    contours, _ = cv2.findContours(image=sds_img.copy(), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
    hu_moments = []
    for contour in contours:
        moments = cv2.moments(contour)
        hu = cv2.HuMoments(moments)
        hu = np.array(hu).flatten()
        hu_moments.append(hu)

    hu_moments = np.array(hu_moments)
    
    if save:
        np.save(save_filename, hu_moments)

    return hu_moments


# In[17]:


def SDs(img, recalculate=False, hu_file="sds_hue_moments.npy", path="sds.png"):
    """Gets an input image of the diacritics image and return a score determines how those diacritics are similar to mohakek and thuluth
        
    Args:
        diacritics_image: A binary image containing only diacritics (white text on black bg)
        recalculate: recalculate precalculate humoments values (default = False)
        path: path of image to recalculate humoments from (default = "sds.png")
        hu_file: file to read precalculated hu_moments from it or write to it in case of recalculate = True (default = "sds_hue_moments.npy")

    Returns:
        score: determine how those diacritics are similar to mohakek and thuluth (more similar => larger score)
    """
    
    # sd1 = np.array([2.40895486e-01, 4.84043539e-03, 1.38924440e-03, 3.48145819e-04, -6.47404757e-08, 6.90885170e-06, -2.33304273e-07])
    # sd2 = np.array([3.54057491e-01, 3.05011527e-03, 3.44288395e-02, 7.77674538e-04, 3.14350358e-06, 4.13909592e-05, -2.51216325e-06])


    if recalculate:
        sd = precalculate_hu(path, True, hu_file)
    else:
        sd = np.load(hu_file)

    contours, _ = cv2.findContours(image=img.copy(), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
    
    hu_moments1 = []
    hu_moments2 = []
    
    for contour in contours:
        moments = cv2.moments(contour)
        hu = cv2.HuMoments(moments)
        hu = np.array(hu).flatten()
        d1 = np.linalg.norm(sd[0] - hu)
        d2 = np.linalg.norm(sd[1] - hu)
        hu_moments1.append(d1)
        hu_moments2.append(d2)
        # hu_moments.append(hu)

    # moments = cv2.moments(np.array(hu_moments))
    # hu = cv2.HuMoments(moments)

    # return hu.flatten()
    min_hu_moments = np.minimum(hu_moments1, hu_moments2)

    if len(min_hu_moments) > 0:
        max_hu = 1/min_hu_moments.sum()
    else:
        max_hu = 0

    return [max_hu]


# # 4. Words Orientation (WOr)
# One of the salient features of the Diwani style is the words written in a slanted format. WOr descriptor mainly specifies how, on average, words in text are oriented. the orientation of each word contour (without diacritics) is calculated and then a mean oriatation is calculated weighted by the area of each word contour.
# WOr algorithm was used to distinguish Diwani from other styles. Diwani style yieldan orientation average of about 45 degrees compared to 0 degrees by other styles.
# 

# In[18]:


def WOr(img, debug=False):
    """Gets an input image of the text without diacritics and return the mean orientation of words
        
    Args:
        text_only_image: A binary image containing only text wihtout diacritics (white text on black bg)
        debug: if true, show the bounding box of words (default False)
    Returns:
        orientation_mean: the weighted mean of words orientation
    """

    contours, _ = cv2.findContours(image=img.copy(), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
    
    orientations = []
    areas = []

    if debug:
        rgb_img = cv2.cvtColor(img*255, cv2.COLOR_GRAY2RGB)

    for contour in contours:
        rect = cv2.minAreaRect(contour)
        p, dimensions, orientation = rect
        # orientations.append(np.abs(orientation) % 60)
        orientations.append(np.abs(orientation))
        x,y,w,h = cv2.boundingRect(contour)
        # areas.append(h*w)
        areas.append(dimensions[0]*dimensions[1])
        # areas.append(w*h)
        # areas.append(w)
        
        if debug:
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(rgb_img,[box],0,(255,255,255),1)
    

    if debug:
        plt.imshow(rgb_img)

    sum_areas = np.sum(areas)
    if sum_areas > 0:
        weighted_mean = np.dot(areas, orientations) / sum_areas
    else:
        weighted_mean = 0

    return [weighted_mean]
    


# In[19]:


def HVSL(edge_image):
    """(Horizontal & Vertical Straight Lines Descriptor)
    Gets the edge image as input and returns a vector of length 2 containing the features:
        a- The appearance frequency of vertical and horizontal lines.
        b- The ratio between the number of the pixels that constitute the texts' edges and the V/H lines' pixels.

    Params:
        edge_image: An image containing only the edges.
        
    Returns:
        HVSL_features: a vector containing the two features mentioned above 
    """

    # A constant defining the minimum percentage of the row/column that a line should have
    # The number of consecutive pixels divided by the total height/width should be at least this ratio to be considered a line
    MINIMUM_LINE_LENGTH_PERCENTAGE = 0.2

    vertical_horizontal_lines_freq = 0
    vertical_horizontal_lines_pixels = 0
    height, width = edge_image.shape

    for row in range(height):
        # Copy all pixels of this row to an array (for better readability)
        rowPixels = edge_image[row, :]
      
        line_pixels = 0
        # Loop through all pixels in row to calculate the number of horizontal lines
        # We only increment the counter when:
        #       1. There's a transition from white to black (to calc the line only once at its beginning)
        #       2. The ratio of consecutive pixels to the whole row length (width of image) is greater than (or equal) the threshold 
        for i in range(len(rowPixels) - 1):
            if rowPixels[i] == 1 and rowPixels[i+1] == 0:
                if line_pixels / width >= MINIMUM_LINE_LENGTH_PERCENTAGE:
                    vertical_horizontal_lines_freq += 1
                    vertical_horizontal_lines_pixels += line_pixels
                line_pixels = 0
            if rowPixels[i] == 1:
                line_pixels += 1

    for col in range(width):
        # Copy all pixels of this column to an array (for better readability)
        colPixels = edge_image[:, col]

        line_pixels = 0
        # Loop through all pixels in column to calculate the number of vertical lines
        # We only increment the counter when:
        #       1. There's a transition from white to black (to calc the line only once at its beginning)
        #       2. The ratio of consecutive pixels to the whole column length (height of image) is greater than (or equal) the threshold 
        for i in range(len(colPixels) - 1):
            if colPixels[i] == 1 and colPixels[i+1] == 0:
                if line_pixels / height >= MINIMUM_LINE_LENGTH_PERCENTAGE:
                    vertical_horizontal_lines_freq += 1
                    vertical_horizontal_lines_pixels += line_pixels
                line_pixels = 0
            if colPixels[i] == 1:
                line_pixels += 1

    # Get the number of pixels in the whole image that are white (edges)
    #edge_pixels = (edge_image[:, :] == 1).sum()
    edge_pixels = edge_image.sum()

    lines_edges_ratio = vertical_horizontal_lines_pixels / edge_pixels

    HVSL_features = [vertical_horizontal_lines_freq, lines_edges_ratio]

    return HVSL_features


# In[20]:


def LVL(skeleton_image):
    """(Long Vertical Lines Descriptor)
    Gets the skeleton image as input and returns a vector of length 5 containing the features:
        a- The text height from the bottom to top.
        b- The number of detected vertical lines.
        c- The length of the highest detected vertical line.
        d- The ratio between the text height and the highest vertical line.
        e- The variance among the vertical lines.

    Params:
        skeleton_image: An image with the skeletonized version of the original input image.
        
    Returns:
        LVL_features: a vector containing the five features mentioned above 
    """
    # TODO Crop the input image to remove the spaces between the text and the border

    # A constant defining the minimum percentage of the column that a line should have
    # The number of consecutive pixels divided by the total height should be at least this ratio to be considered a line
    MINIMUM_LINE_HEIGHT_PERCENTAGE = 0.1

    height, width = skeleton_image.shape

    text_top = 0
    for row in range(height):
        if skeleton_image[row, :].sum() != 0:
            break
        else:
            text_top += 1
        
    text_bottom = height
    for row in range(height - 1, 0, -1):
        if skeleton_image[row, :].sum() != 0:
            break
        else:
            text_bottom -= 1
    
    text_height = text_bottom - text_top

    vertical_lines = []

    for col in range(width):
        # Copy all pixels of this column to an array (for better readability)
        colPixels = skeleton_image[:, col]

        line_pixels = 0
        # Loop through all pixels in column to calculate the number of vertical lines
        # We only increment the counter when:
        #       1. There's a transition from white to black (to calc the line only once at its beginning)
        #       2. The ratio of consecutive pixels to the whole column length (height of image) is greater than (or equal) the threshold 
        for i in range(len(colPixels) - 1):
            if colPixels[i] == 1 and colPixels[i+1] == 0:
                if line_pixels / height >= MINIMUM_LINE_HEIGHT_PERCENTAGE:
                    vertical_lines.append(line_pixels)
                line_pixels = 0
            if colPixels[i] == 1:
                line_pixels += 1

    vertical_lines_freq = len(vertical_lines)

    if vertical_lines_freq < 1:
        highest_vertical_line_length = 0
        vertical_lines_variance = 0
    else:
        highest_vertical_line_length = max(vertical_lines)
        vertical_lines_variance = np.var(vertical_lines)


    if text_height < 1:
        highest_vertical_line_to_text_height_ratio = 0
    else:   
        highest_vertical_line_to_text_height_ratio = highest_vertical_line_length / text_height
    

    LVL = [text_height, vertical_lines_freq, highest_vertical_line_length, highest_vertical_line_to_text_height_ratio, vertical_lines_variance]

    return LVL


# In[21]:


def ToE_ToS(image, bins=9):
    fd, hog_image = hog(image.astype(np.uint8), orientations=bins, pixels_per_cell=(4, 4),
                    cells_per_block=(1, 1), visualize=True)
    w, h = image.shape
    # print("ToE", np.round(np.histogram(fd, bins=10)[0][2:] / (w * h), 3))
    # h, _ = np.histogram(fd, bins=10)[0][2:] / (w * h)

    h, _ = np.histogram(fd, range=(1, max(fd.max(),1)), bins=bins)

    # normalizing the histogram
    h_sum = h.sum()
    if h_sum > 1:
        h = h/h_sum

    return h


# In[22]:


# # importing io module
# import sys
# sys.path.insert(1, "./../io_utils/")
# from io_utils import read_data, read_classes

# # reading data and class names
# classes_names = read_classes('../ACdata_base/names.txt')
# dataset_images, dataset_labels = read_data('../ACdata_base/')


# In[23]:


def HPP(image, bins=10, croped=True):
    
    if not croped:
        image = crop(image)

    white_pixels_in_row = image.sum(axis=1)

    h, h_bins = np.histogram(white_pixels_in_row, range=(1, white_pixels_in_row.max()), density=False, bins=bins)

    # normalizing the histogram
    h_sum = h.sum()
    if h_sum > 1:
        h = h/h_sum

    return [h.var()]


# # 3. Testing

# In[24]:


def testing():
    # import randrange
    from random import randrange
    # importing io module
    import sys
    sys.path.insert(1, "./../io_utils/")
    from io_utils import read_data, read_classes

    # importing preprocessing module
    sys.path.insert(1, "./../preprocessing/")
    from preprocessing import binarize, extract_edges, extract_skeleton, separate_diacritics_and_text, crop

    # reading data and class names
    classes_names = read_classes('../ACdata_base/names.txt')
    dataset_images, dataset_labels = read_data('../ACdata_base/')
    
    assert len(dataset_images) == len(dataset_labels)

    # get the range of each class in the dataset
    ranges = [0]
    tmp = dataset_labels[0]
    for idx, num in enumerate(dataset_labels):
        if num != tmp:
            tmp = num
            ranges.append(idx)
    ranges.append(len(dataset_labels))

    # test_image = cv2.imread("../ACdata_base/7/1124.jpg",0)
    # diacritics_image, text_image = separate_diacritics_and_text(test_image)
    # wor = WOr(text_image, True)
    # print(wor)

    # Choosing a random example from each class and apply the preprocessing Functions on it
    for i, class_name in enumerate(classes_names):
        # break
        index = randrange(ranges[i], ranges[i+1])
        test_image = dataset_images[index]
        binary_image = binarize(test_image)
        cropped_image = crop(binary_image)
        skeleton_image = extract_skeleton(binary_image)
        edge_image = extract_edges(binary_image)
        diacritics_image, text_image = separate_diacritics_and_text(binary_image)

        assert len(np.unique(np.asarray(binary_image))) == 2

        tth = Tth(skeleton_image, edge_image)
        sds = SDs(diacritics_image)[0]
        wor = WOr(text_image)[0]
        hpp = HPP(cropped_image)[0]
        lvl = LVL(skeleton_image) #list of 5
        hvsl = HVSL(edge_image) #list of 2
        toe = ToE_ToS(edge_image,10) #list of 2
        tos = ToE_ToS(skeleton_image,10) #list of 2

        f, axarr = plt.subplots(1,1, figsize=(10, 7))

        suptitle = "Tth: " + np.array2string(tth)
        suptitle += "\nSDs: " + str(sds)
        suptitle += "\nWOr: " + str(wor)
        suptitle += "\nToE: " + np.array2string(toe)
        suptitle += "\nToS: " + np.array2string(tos)
        suptitle += "\nHPP: " + str(hpp)
        suptitle += "\nHVSL: " + str(hvsl)
        suptitle += "\nLVL: " + str(lvl)

        f.suptitle(suptitle)

        axarr.imshow(test_image, cmap='gray')
        axarr.set_title(class_name)


# In[25]:


if __name__ == '__main__':
    testing()


# In[ ]:


def create_py():
    get_ipython().system('jupyter nbconvert --to script feature_extraction.ipynb')


# In[ ]:


if __name__ == '__main__':
    create_py()

