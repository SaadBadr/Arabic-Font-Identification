import pickle
from preclassification_pipeline import preprocessing_feature_extraction_pipeline
import numpy as np
import os
import cv2


model = pickle.load(open("./trained_model", 'rb'))

data_collected_test = []
labels_collected_test = np.loadtxt("./New_DB/new_test.csv")
dataset_path="./New_DB/testo"
for dirname, _, filenames in os.walk(dataset_path):
    for filename in filenames:
        img = cv2.imread(os.path.join(dirname, filename), 0)
        data_collected_test.append(img)


features = preprocessing_feature_extraction_pipeline(img)

score = model.score(features, labels_collected_test)

print(score)