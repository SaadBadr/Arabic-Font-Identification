from preprocessing import preprocessing
from feature_extraction import feature_extraction 
import numpy as np

def preprocessing_pipline(img, separate_diacritics=True):
    """ apply preprocessing stages for one image

    Args:
        image: grayscale image
    Returns:
        cropped_img: cropped image
        edge_img: edge image
        skeleton_img: skeleton image
        diacritics_img: diacritics only image
        text_img: text only image
    """

    binary_img = preprocessing.binarize(img)
    cropped_img = preprocessing.image_resize(preprocessing.crop(binary_img), height=64)
    edge_img = preprocessing.extract_edges(cropped_img)
    skeleton_img = preprocessing.extract_skeleton(cropped_img)
    diacritics_img, text_img = None, None
    if separate_diacritics:
        diacritics_img, text_img = preprocessing.separate_diacritics_and_text(cropped_img)
    
    return cropped_img, edge_img, skeleton_img, diacritics_img, text_img


def feature_extraction_pipeline(cropped_img, edge_img, skeleton_img, diacritics_img, text_img):
    """ apply feature extraction stages on one image

    Args:
        cropped_img: cropped image
        edge_img: edge image
        skeleton_img: skeleton image
        diacritics_img: diacritics only image
        text_img: text only image
    Returns:
        hvsl, toe, tos, lvl, tth, sds, wor, hpp, lpq
    """
    hvsl = feature_extraction.HVSL(edge_img)
    toe = feature_extraction.ToE_ToS(edge_img,12) #12
    tos = feature_extraction.ToE_ToS(skeleton_img,10) #10
    lvl = feature_extraction.LVL(skeleton_img)
    tth = feature_extraction.Tth(skeleton_img, edge_img, 64) #10
    sds = feature_extraction.Diacritics(diacritics_img, 128)
    wor = feature_extraction.WOr(text_img)
    hpp = feature_extraction.HPP(cropped_img, 64) #13
    lpq = feature_extraction.LPQ(cropped_img) #13

    return hvsl, toe, tos, lvl, tth, sds, wor, hpp, lpq


def preprocessing_feature_extraction_pipeline(data, one_feature_vector=True):
    """ apply preprocessing and feature extraction stages on set of images
    
    Args:
        data: grayscale images
        one_feature_vector: stack features horizontally (default True)
    Returns:
        data_features
    """

    data_features = [[] for i in range(0,9)]
    for element in data:
        preprocessed = preprocessing_pipline(element)
        features = feature_extraction_pipeline(preprocessed[0], preprocessed[1], preprocessed[2], preprocessed[3], preprocessed[4])
        for i, feature in enumerate(features):
            data_features[i].append((feature))

    for i in range(len(data_features)):
        data_features[i] = np.vstack(data_features[i])

    if one_feature_vector:
        data_features = np.hstack(data_features)

    return data_features