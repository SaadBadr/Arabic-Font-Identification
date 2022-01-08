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

# In[1]:


##################################################### imports #####################################################
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from scipy.signal import convolve2d


# # 2. Text thickness (Tth)
# Stroke thickness plays an important role in defining the style. Some styles use a flat pen, whereas some others use a pointed one. In some styles, calligraphers alter the thickness while writing (via pushing down the pen or the opposite), whereas in others, the thickness is always preserved. Modeling such a feature in form of a descriptor will help the machine to understand more specificities of each style. Text thickness (Tth) descriptor codifies the appearance frequency of different line thicknesses in a text image. To extract this descriptor, we employ both the skeleton and edge image, and thickness is determined by the distance between skeleton and edges.

# In[2]:


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


# # 3. Diacritics
# Thuluth and Mohakik have a similar writing style that is decorated with diacritics
# A Diacritics descriptor will be used to inspect the density of diacritics in a given text image.

# In[3]:


def DD(d_img, bins=12):
    return np.histogram(d_img.sum(axis=0), bins=bins)[0]/d_img.shape[1]


# # 4. Words Orientation (WOr)
# One of the salient features of the Diwani style is the words written in a slanted format. WOr descriptor mainly specifies how, on average, words in text are oriented. the orientation of each word contour (without diacritics) is calculated and then a mean oriatation is calculated weighted by the area of each word contour.
# WOr algorithm was used to distinguish Diwani from other styles. Diwani style yieldan orientation average of about 45 degrees compared to 0 degrees by other styles.
# 

# In[4]:


def WOr(img):
    """Gets an input image of the text without diacritics and return the mean orientation of words
        
    Args:
        text_only_image: A binary image containing only text wihtout diacritics (white text on black bg)
        debug: if true, show the bounding box of words (default False)
    Returns:
        orientation_mean: the weighted mean of words orientation
    """
    thetas = []
    rhos = []
    imgArea = img.shape[0] * img.shape[1]
    numLabels, _, stats, _ = cv2.connectedComponentsWithStats(img)
    indexes = list(range(len(stats)))
    # We sort the indeces list with heuristics being its key (Sorting heuristics and storing the indices of the sorted list)
    indexes.sort(key=stats[:, cv2.CC_STAT_AREA].__getitem__, reverse=True)
    # Then we map the actions_states list according to the indeces of the sorted list 
    stats = list(map(stats.__getitem__, indexes))

    for i in range(0, 5):
        if i >= len(stats):
            break
        x = stats[i][cv2.CC_STAT_LEFT]
        y = stats[i][cv2.CC_STAT_TOP]
        w = stats[i][cv2.CC_STAT_WIDTH]
        h = stats[i][cv2.CC_STAT_HEIGHT]
        area = stats[i][cv2.CC_STAT_AREA]
        if area < 0.001 * imgArea:
            continue
        bounded_image = img[y: y + h, x: x + w]
        lines = cv2.HoughLines(bounded_image, 1, np.pi/180, 1)
        if lines is not None:
            line = lines[0][0]
            rho = line[0]
            theta = line[1]
            if np.rad2deg(theta) > 60 or np.rad2deg(theta) < -10:
                continue
            rhos.append(rho)
            thetas.append(theta)

    contours, _ = cv2.findContours(img, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
    
    orientations = []
    areas = []
    
    for contour in contours:

        rows,cols = img.shape[:2]
        [vx,vy,x,y] = cv2.fitLine(contour, cv2.DIST_L2,0,0.01,0.01)
        angle = np.rad2deg(np.arctan2(vy[0], vx[0]))
        area = cv2.contourArea(contour)

        orientations.append(angle)
        areas.append(area)
    
    m = 2
    n = len(areas)
    if len(areas) > m:
        n = m

    w = []

    if len(orientations) > 0:
        w.append(np.var(orientations))
        w.append(np.mean(orientations))

    else:
        w.append(0)
        w.append(0)

    max_n = np.argpartition(areas, -n)[-n:]

    for i in range(len(max_n)-1,-1,-1):
        w.append(orientations[i])

    for i in range(m-n):
        w.append(0)
    

    if len(thetas) == 0:
        thetas.append(0)
    if len(rhos) == 0:
        rhos.append(0)

    for i in range(len(rhos), 5):
        rhos.append(0)
    for i in range(len(thetas), 5):
        thetas.append(0)

    w.append(np.mean(rhos))
    w.append(np.mean(thetas))

    return np.append(rhos, np.append(thetas, w))


# In[5]:


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
    MINIMUM_VLINE_LENGTH_PERCENTAGE = 0.02
    MINIMUM_HLINE_LENGTH_PERCENTAGE = 0.015

    height, width = edge_image.shape
    vertical_horizontal_lines_freq = 0
    
    vertical_size = max(int(MINIMUM_VLINE_LENGTH_PERCENTAGE * height), 1)
    horizontal_size = max(int(MINIMUM_HLINE_LENGTH_PERCENTAGE * width), 1)
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_size))
    mask1 = cv2.morphologyEx(edge_image, cv2.MORPH_OPEN, horizontalStructure)
    mask2 = cv2.morphologyEx(edge_image, cv2.MORPH_OPEN, verticalStructure)
    
    contours_h, _ = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_v, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes_h = [cv2.boundingRect(contour) for contour in contours_h]
    bounding_boxes_v = [cv2.boundingRect(contour) for contour in contours_v]
    vertical_lines = [bounding_box[3] for bounding_box in bounding_boxes_v]
    horizontal_lines = [bounding_box[2] for bounding_box in bounding_boxes_h]

    vertical_horizontal_lines_pixels = sum(vertical_lines) + sum(horizontal_lines)
    vertical_horizontal_lines_freq = len(vertical_lines) + len(horizontal_lines)

    edge_pixels = edge_image.sum()

    lines_edges_ratio = vertical_horizontal_lines_pixels / edge_pixels

    HVSL_features = [vertical_horizontal_lines_freq, lines_edges_ratio]

    return HVSL_features


# In[6]:


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
    MINIMUM_VLINE_LENGTH_PERCENTAGE = 0.03

    vertical_lines_freq = 0
    height, width = skeleton_image.shape
    vertical_lines = []
    vertical_size = max(int(MINIMUM_VLINE_LENGTH_PERCENTAGE * height), 1)
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_size))

    mask2 = cv2.morphologyEx(skeleton_image, cv2.MORPH_OPEN, verticalStructure)
    
    contours, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = [cv2.boundingRect(contour) for contour in contours]
    vertical_lines = [bounding_box[3] for bounding_box in bounding_boxes]
    vertical_lines_freq = len(bounding_boxes)

    if vertical_lines_freq < 1:
        highest_vertical_line_length = 0
        vertical_lines_variance = 0
    else:
        highest_vertical_line_length = max(vertical_lines)
        vertical_lines_variance = np.var(vertical_lines)

    text_height = height
    if text_height < 1:
        highest_vertical_line_to_text_height_ratio = 0
    else:   
        highest_vertical_line_to_text_height_ratio = highest_vertical_line_length / text_height
    

    LVL = [text_height, vertical_lines_freq, highest_vertical_line_length, highest_vertical_line_to_text_height_ratio, vertical_lines_variance]
    
    return LVL


# In[7]:


def ToE_ToS(image, bins=10):
    """(Text orientation from Edges / Text orientation from Skeleton Descriptor)
    Finds the orientation of edges / skeleton image by applying a sobel filter to capture the intensities of edges 
        and the orientation of these edges 

    Params:
        image: An image with the skeletonized / edge version of the original input image.
        
    Returns:
        ToE/ToS: a vector containing the intensities and orientations of the pixels
    """
    
    Kx = np.array([[-2, 0, 2], 
                   [-1, 0, 1], 
                   [-2, 0, 2]], np.float32)

    Ky = np.array([[2, 1, 2], 
                   [0, 0, 0], 
                   [-2, -1, -2]], np.float32)
    
    Ix = ndimage.filters.convolve(image, Kx)
    Iy = ndimage.filters.convolve(image, Ky)
    
    G = np.hypot(Ix, Iy)
    G = G / G.max()
    # G = G[G != 0]
    theta = np.arctan2(Iy, Ix)
    # theta = theta[theta != 0]

    Gs, _ = np.histogram(G, bins=bins)
    # Gs = np.append(Gs, np.var(G))
    thetas, _ = np.histogram(theta, bins=bins)
    # thetas = np.append(thetas, np.var(theta))
    G_sum = Gs.sum()
    if G_sum > 1:
        Gs = Gs / G_sum 
    T_sum = thetas.sum()
    if T_sum > 1:
        thetas = thetas / T_sum

    return np.append(Gs, thetas)


# In[8]:


def HPP(image, bins=10):
    """(Horizontal Profile Projection)
    Finds the projection of the horizontal lines in an image and allocate them to specific bins based on their positions

    Params:
        image: A binary image
        
    Returns:
        h: a vector containing the number of pixels in each bin 
    """
    white_pixels_in_row = image.sum(axis=1)

    h, h_bins = np.histogram(white_pixels_in_row, range=(1, white_pixels_in_row.max()), density=False, bins=bins)

    # normalizing the histogram
    h_sum = h.sum()
    if h_sum > 1:
        h = h/h_sum

    return h


# In[9]:


def LPQ(img,winSize=5,freqestim=2,hist_size=1024):

    STFTalpha=1/winSize  # alpha in STFT approaches (for Gaussian derivative alpha=1)
    sigmaS=(winSize-1)/4 # Sigma for STFT Gaussian window (applied if freqestim==2)

    convmode='valid' # Compute descriptor responses only on part that have full neigborhood. Use 'same' if all pixels are included (extrapolates np.image with zeros).

    img=np.float64(img) # Convert np.image to double
    r=(winSize-1)/2 # Get radius from window size
    x=np.arange(-r,r+1)[np.newaxis] # Form spatial coordinates in window
    u=np.arange(1,r+1)[np.newaxis]

    if freqestim==1:  #  STFT uniform window
        #  Basic STFT filters
        w0=np.ones_like(x)
        w1=np.exp(-2*np.pi*x*STFTalpha*1j)
            # exp(complex(0,-2*pi*x*STFTalpha))
        w2=np.conj(w1)

    elif freqestim == 2:
        w0=(x*0+1)
        w1=np.exp(-2*np.pi*x*STFTalpha*1j)
        w2=np.conj(w1)

        # Gaussian window
        gs=np.exp(-0.5*np.power(x / sigmaS, 2)) / (np.sqrt(2*np.pi) * sigmaS)

        # Windowed filters
        w0=gs * w0
        w1=gs * w1
        w2=gs * w2

        # Normalize to zero mean 
        w1=w1-np.mean(w1)
        w2=w2-np.mean(w2)

    ## Run filters to compute the frequency response in the four points. Store np.real and np.imaginary parts separately
    # Run first filter

    filterResp1=convolve2d(convolve2d(img,w0.T,convmode),w1,convmode)
    filterResp2=convolve2d(convolve2d(img,w1.T,convmode),w0,convmode)
    filterResp3=convolve2d(convolve2d(img,w1.T,convmode),w1,convmode)
    filterResp4=convolve2d(convolve2d(img,w1.T,convmode),w2,convmode)

        # Initilize frequency domain matrix for four frequency coordinates (np.real and np.imaginary parts for each frequency).
    freqResp=np.dstack([filterResp1.real, filterResp1.imag,
                        filterResp2.real, filterResp2.imag,
                        filterResp3.real, filterResp3.imag,
                        filterResp4.real, filterResp4.imag])

    ## Perform quantization and compute LPQ codewords
    inds = np.arange(freqResp.shape[2])[np.newaxis,np.newaxis,:]
    LPQdesc=((freqResp>0)*(2**inds)).sum(2)

    LPQdesc=np.histogram(LPQdesc.flatten(),range(hist_size))[0]

    LPQdesc=LPQdesc/LPQdesc.sum()
    
    return LPQdesc


# # 3. Testing

# In[10]:


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
        skeleton_image = extract_skeleton(binary_image).astype(np.uint8)
        edge_image = extract_edges(binary_image)
        diacritics_image, text_image = separate_diacritics_and_text(binary_image)

        assert len(np.unique(np.asarray(binary_image))) == 2

        tth = Tth(skeleton_image, edge_image)
        diacritics = DD(diacritics_image)
        wor = WOr(text_image)[0]
        hpp = HPP(cropped_image)
        lvl = LVL(skeleton_image) #list of 5
        hvsl = HVSL(edge_image) #list of 2
        toe = ToE_ToS(edge_image,10) #list of 2
        tos = ToE_ToS(skeleton_image,10) #list of 2
        lpq = LPQ(cropped_image)

        f, axarr = plt.subplots(1,1, figsize=(10, 7))

        suptitle = "Tth: " + np.array2string(tth)
        suptitle += "\nWOr: " + str(wor)
        suptitle += "\nToE: " + np.array2string(toe)
        suptitle += "\nToS: " + np.array2string(tos)
        suptitle += "\nHPP: " + str(hpp)
        suptitle += "\nHVSL: " + str(hvsl)
        suptitle += "\nLVL: " + str(lvl)
        f.suptitle(suptitle)

        axarr.imshow(cropped_image, cmap='gray')
        axarr.set_title(class_name)


# In[11]:


# if __name__ == '__main__':
#     testing()


# In[12]:


def create_py():
    get_ipython().system('jupyter nbconvert --to script feature_extraction.ipynb')


# In[13]:


if __name__ == '__main__':
    create_py()


# In[ ]:




