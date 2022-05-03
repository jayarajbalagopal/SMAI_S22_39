####
# Preprocess smiles
#
#

# Dependencies
import pandas as pd
import numpy as np
import time
from cddd.inference import InferenceModel
from cddd.preprocessing import preprocess_smiles
#import pickle
#import pickle5 as pickle


file1 = open('/mnt/img2mol/smile_list.txt', 'r')
data = file1.read()
file1.close()

smile_list = data.split("\n")

print(type(smile_list))
print(len(smile_list))
print(smile_list[0])

smiles = smile_list

# Process in batches
for it in range(0, 25):
    start_count = it*1000
    end_count = start_count + 1000
    if end_count > len(smiles):
        end_count = len(smiles)


    print("batch:", it)

    sample = smiles[start_count:end_count]

    start_time = time.time()

    inference_model = InferenceModel()
    smiles_embedding = inference_model.seq_to_emb(sample)
    decoded_smiles_list = inference_model.emb_to_seq(smiles_embedding)
    #print(smiles_embedding.shape)
    #print(decoded_smiles_list)
    #print(type(smiles_embedding))
    end_time = time.time()

    print("time: ", end_time - start_time)

    print("writing embeddings")

    embedding_file = open("/mnt/smile_embeddings_img2mol_validation_"+str(start_count)+"_to_"+str(end_count)+".npy", "wb")
    np.save(embedding_file, smiles_embedding, allow_pickle=False)
    embedding_file.close()

    print("converting smiles")

    can_arr = np.array(decoded_smiles_list)

    print("writing embeddings")

    decoded_file = open("/mnt/decoded_smiles_img2mol_validation_"+str(start_count)+"_to_"+str(end_count)+".npy", "wb")
    np.save(decoded_file, can_arr, allow_pickle=False)
    decoded_file.close()

