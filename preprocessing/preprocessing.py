#!/usr/bin/env python
# coding: utf-8

# Hello everyone,
# 
# In this module, it is required to implement the preprocessor for Arabic Font Identification System.
# 
# At first, what are the main steps that we should go through in this module?
# 
# 
# # TODOs:
# 
# 1. Understand the problem
# 2. Binarization
# 3. Extract Edges
# 4. Extract Skeleton
# 5. Extract Diacritics
# 6. Extract Text only
# 7. Testing

# In[241]:


##################################################### imports #####################################################
from skimage import io, filters
from skimage.morphology import skeletonize
import cv2
import os
import matplotlib.pyplot as plt
from random import randrange
import numpy as np
from skimage.exposure import histogram
from matplotlib.pyplot import bar
from skimage.segmentation import flood_fill


# # 1. Understand the problem
# 
# We are interested only in the morphology of text letters, all images are first converted into binary (i.e., black text on a white background). It should be known that most images contain either meaningful texture, which is a part of the decoration or a meaningless one that resulted from the noise while capturing. In either case, we illuminate the background so it does not affect the results.
# 
# - Input: --
# - Output: --

# In[242]:


def analyze_histogram(binary, show_hist=False):

    """Gets a binary image and returns it reverted if text was black (Makes text white for all images)
        Except for Square Kuffi

    Args:
        binary : The binary image you want to transform for consistency (White text on black bg) (Read using cv2)
        
    Returns:
        binary || reverted_binary : The outcome of analysis (Binary image with white text on black bg)
    """

    height = (binary.shape)[0]
    white_pixels_in_row = []
    
    # Calculate number of white pixels in a row
    for row in range(height - 1):
        white_pixels_in_row.append((binary[row, :] == 1).sum())

    # Set the threshold of white pixels in a row to be 70% of the row with most white pixels
    # We use this threshold to see how many rows contain a high number of white pixels 
    pixels_threshold = int(0.7 * max(white_pixels_in_row))

    # Get the number of rows above the pixels_threshold
    rows_above_thresh = 0
    for row_pixels in white_pixels_in_row:
        if row_pixels > pixels_threshold:
            rows_above_thresh += 1

    # Calculate the ratio between the number of rows dominated by white pixels to the number of rows in the entire image
    ratio = rows_above_thresh / height

    # if show_hist:
    #     x = np.arange(height)
    #     plt.figure(figsize =(10, 7)) 
    #     plt.plot(x, white_pixels_in_row)
    #     plt.show()
    
    # If more than 70% of the image is white rows then the background is white and we have to revert the image
    if ratio > 0.7:
        return 1 - binary
    return binary


# # 2. Binarization

# In[243]:


def binarize(image):
    """Gets an image and returns it binarized having values 0 OR 1 using OTSU thresholding technique

    Args:
        image : The image you want to binarize (Read using cv2)
        
    Returns:
        binarized: The binarized version of the image
    """
    # Use Otsu for normal binarization
    # The parameters are 1-`The input image`, 2-`The binarization threshold (will not be used if OTSU technique is used)`,
    #                        3-`Maximum value`, 4-`Binarization technique`
    th, binary_image = cv2.threshold(image, 128, 1, cv2.THRESH_OTSU)

    # Perform histogram analysis to make text white for all images (Except for square Kuffi)
    corrected_binary_image = analyze_histogram(binary_image)
    return corrected_binary_image


# In[244]:


#  np.array([[0,1,0],[1,1,1],[0,1,0]])


# In[245]:


def crop(img):
    # morph_img = img.copy()
    morph_img = img
    # element = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3),( 0, 0 ))
    # element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    # morph_img = cv2.morphologyEx(morph_img, cv2.MORPH_OPEN, element)
    # morph_img = cv2.morphologyEx(morph_img, cv2.MORPH_CLOSE, element)

    rows = morph_img.sum(axis=1)
    cols = morph_img.sum(axis=0)
    rows = rows != 0
    cols = cols != 0

    c = np.where(cols == cols.max())[0]
    r = np.where(rows == rows.max())[0]

    x = c[0], c[-1]
    y = r[0], r[-1]
    
    try:
        cropped_img = img[y[0]:y[1], x[0]:x[1]]
    except:
        cropped_img = img

    return cropped_img


# # 3. Extract Edges

# In[246]:


def extract_edges(image):
    """Gets the original input image and returns an image containing its edges using canny edge detector

    Args:
        image: The image you want to extract its edges (Read using cv2)
        
    Returns:
        edge_image: The edge extracted image
    """
    # Blur the image for better edge detection
    if image.max() == 1:
        image = image * 255
        image = image.astype(np.uint8)

    img_blur = cv2.GaussianBlur(image,(3,3), sigmaX=0, sigmaY=0)

    # Use Canny edge detection to extract edges
    edge_image = cv2.Canny(image=img_blur, threshold1=100, threshold2=200)

    # Change image max value from 255 to 1
    edge_image[edge_image == 255] = 1

    return edge_image


# # 4. Extract Skeleton

# In[247]:


def extract_skeleton(image):
    """Gets an image and returns an image containing its skeleton 

    Args:
        image: The image you want to skeletonize (Read using cv2)
        
    Returns:
        skeleton: The skeletonized image
    """
    # We start by binarizing the image
    # binary = binarize(image)

    # Use Skimage's skeletonize method with the array representation of the binary image as input
    skeleton = skeletonize(np.asarray(image))

    return skeleton


# # 5. Separate Diacritics from Text

# In[248]:


def separate_diacritics_and_text_utility(image):
    """Utility function used by separate_diacritics_and_text
    """
    
    # binary = binarize(image)  
    binary = image  

    # height = (binary.shape)[0]
    # white_pixels_in_row = np.zeros(height, dtype=np.uint32)  
    # Calculate number of white pixels in a row
    # for row in range(height - 1):
        # white_pixels_in_row[row] = (binary[row, :] == 1).sum()

    white_pixels_in_row = binary.sum(axis=1)

    # Get the row with most white pixels (This row would contain only text and commas no diacritics
    #                                        as it is an imaginary line where text is placed on)
    baseline_row = np.argmax(white_pixels_in_row)
    diacritics_image = binary

    # How Flood Fill works for our case:
    #   It starts at a seed point (pixel) and changes all connected pixels to it to have a certain color
    #   So what we want to do is get the first pixel of each word along the line (it would be black) and  
    #   use that point as seed for the flood_fill algorithm changing that word to have the same color as
    #   the background. We keep doing that for all words until only diacritics remain.

    # Loop through the base_line's pixels to apply flood fill on all words
    rowPixels = binary[baseline_row, :]
    for i in range(len(rowPixels) - 1):
        # Handle the case where the first pixel is part of a word so we apply the flood fill from the end of the word
        if rowPixels[i] == 1 and rowPixels[i+1] == 0:
            diacritics_image = flood_fill(diacritics_image, (baseline_row, i), 0)
        elif rowPixels[i] == 0 and rowPixels[i+1] == 1:
            diacritics_image = flood_fill(diacritics_image, (baseline_row, i+1), 0)
    
    # print(baseline_row, binary.shape[0])
    text_image = binary - diacritics_image
    cv2.line(text_image,(0,baseline_row),(image.shape[1],baseline_row),(0,0,255),2)
    
    return diacritics_image, text_image


# In[249]:


def separate_diacritics_and_text(image, diacritics_ratio=0.2, max_iterations=1000):
    """Gets an input image of the arabic text and returns an image containing only the diacritics in it and another
    image with the text as a tuple (diacritics, text)
        
    Args:
        image: The image you want to separate its diacritics from its text (Read using cv2)
        diacritics_ratio: threshold ratio of diacritics to text (default 0.2)
        max_iterations: maximum number of iterations to run the separtion algorithm (default 1000)
    Returns:
        diacritics_image: A binary image containing only diacritics (white text on black bg)
        text_image: A binary image containing only text (white text on black bg)
    """

    diacritics_image, text_image = separate_diacritics_and_text_utility(image)
    # while diacritics_image.sum() > diacritics_image.size * diacritics_ratio:
    while diacritics_image.sum() > text_image.sum() * diacritics_ratio and max_iterations:
        diacritics_image, text_image_1 = separate_diacritics_and_text_utility(diacritics_image)
        text_image = text_image + text_image_1
        max_iterations = max_iterations - 1
        
    return diacritics_image, text_image
    


# # 6. Testing

# In[250]:


def testing():
    
    # importing io module
    import sys
    sys.path.insert(1, "./../io_utils/")
    from io_utils import read_data, read_classes

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

    # plt.imshow(crop(binarize(cv2.imread("../ACdata_base/9/1624.jpg", 0))), cmap='gray')

    # Choosing a random example from each class and apply the preprocessing Functions on it
    for i, class_name in enumerate(classes_names):
        index = randrange(ranges[i], ranges[i+1])
        test_image = dataset_images[index]

        if(i == 8):
            test_image = cv2.imread("../ACdata_base/9/1624.jpg", 0)

        binary_image = binarize(test_image)
        cropped_image = crop(binary_image)
        assert len(np.unique(np.asarray(binary_image))) == 2

        edge_image = extract_edges(cropped_image)
        skeleton_image = extract_skeleton(cropped_image)
        diacritics_image, text_image = separate_diacritics_and_text(cropped_image)

        f, axarr = plt.subplots(3,2, figsize=(15, 10))
        f.suptitle(class_name + " test")

        axarr[0,0].imshow(test_image, cmap='gray')
        axarr[0,0].set_title("Original")

        # axarr[0,1].imshow(binary_image, cmap='gray')
        # axarr[0,1].set_title("Binary")
        axarr[0,1].imshow(cropped_image, cmap='gray')
        axarr[0,1].set_title("Cropped Binary")

        axarr[1,0].imshow(edge_image, cmap='gray')
        axarr[1,0].set_title("Edge")

        axarr[1,1].imshow(skeleton_image, cmap='gray')
        axarr[1,1].set_title("Skeleton")

        axarr[2,0].imshow(diacritics_image, cmap='gray')
        axarr[2,0].set_title("Diacritics")

        axarr[2,1].imshow(text_image, cmap='gray')
        axarr[2,1].set_title("Text Only")
    


# In[251]:


# # importing io module
# import sys
# sys.path.insert(1, "./../io_utils/")
# from io_utils import read_data, read_classes

# # reading data and class names
# classes_names = read_classes('../ACdata_base/names.txt')
# dataset_images, dataset_labels = read_data('../ACdata_base/')


# 

# In[252]:


if __name__ == '__main__':
    testing()


# In[253]:


def create_py():
    get_ipython().system('jupyter nbconvert --to script preprocessing.ipynb')


# In[254]:


if __name__ == '__main__':
    create_py()

