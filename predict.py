import pickle
from preclassification_pipeline import preprocessing_feature_extraction_pipeline
import numpy as np
import os
import cv2
import sys
import time


model = pickle.load(open("./trained_model", 'rb'))

test_dir = sys.argv[1]
out_dir = sys.argv[2]


results = str(out_dir) + "/results.txt"
times = str(out_dir) + "/times.txt"

os.makedirs(os.path.dirname(results), exist_ok=True)
os.makedirs(os.path.dirname(times), exist_ok=True)

results_file = open(results, "w")
times_file = open(times, "w")

# labels_collected_test = np.loadtxt("./New_DB/new_test.csv")
dataset_path=str(test_dir)
for dirname, _, filenames in os.walk(dataset_path):
    for index, filename in enumerate(filenames):
        
        prediction = [-1]
        start_time = time.time()
        try:
            img = cv2.imread(os.path.join(dirname, filename), 0)

            features = preprocessing_feature_extraction_pipeline([img])

            prediction = model.predict(features)
            
        except:
            pass
        execution_time = time.time() - start_time
        execution_time = round(execution_time, 2)
        
        if execution_time == 0:
            execution_time = 0.001
            
        if(index != 0):
            results_file.write('\n')
        results_file.write(str(prediction[0]))

        if(index != 0):
            times_file.write('\n')
        times_file.write(str(execution_time))
    
results_file.close()
times_file.close()