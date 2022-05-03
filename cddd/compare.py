####
# Compare cddd
#
#

# Dependencies
import pandas as pd
import numpy as np
import time
from cddd.inference import InferenceModel
from cddd.preprocessing import preprocess_smiles
from sklearn.metrics import mean_squared_error


inference_model = InferenceModel()

mse_arr = []
match_arr = []

for it in range(22,39):
    
    print("batch:", it)


    file1 = open("/mnt/img2mol_val_results/benchmark_predictions_"+str(it)+".pkl", "rb")
    predictions = np.load(file1)
    file1.close()
    
    file2 = open("/mnt/img2mol_val_results/benchmark_test_labels_"+str(it)+".pkl", "rb")
    test_labels = np.load(file2)
    file2.close()
    
    mse = mean_squared_error(predictions, test_labels)
    
    print("mse:", mse)
    
    start_time = time.time()
    
    predicted_smiles = inference_model.emb_to_seq(predictions)
    expected_smiles = inference_model.emb_to_seq(test_labels)
    
    end_time = time.time()
    
    print("time:", end_time - start_time)
    
    match_count = 0
    for i in range(len(predicted_smiles)):
        if predicted_smiles[i] == expected_smiles[i]:
            match_count = match_count + 1
    
    print("matches:", match_count)

    print("\n\n")

    for i in range(10):
        print("\tprediction:", predicted_smiles[i])
        print("\texpected:  ", expected_smiles[i])

    print("\n\n")

    #file1 = open("/mnt/img2mol_val_results/mse_"+str(it)+".txt", "w")
    #file1.write(str(mse))
    #file1.close()
    #
    #file2 = open("/mnt/img2mol_val_results/match_"+str(it)+".txt", "w")
    #file2.write(str(match_count))
    #file2.close()
    
    break
