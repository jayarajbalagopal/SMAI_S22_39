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

#data = pd.read_csv("/mnt/smiles_copy.csv", header=None)
data = pd.read_csv("/mnt/smiles.csv", delimiter='\t')
#print(data.head())

#print(data.shape)

#arr = data.to_numpy()

#print("after conv")
#print(arr.shape)
#print(arr[0])

#print(arr.shape)


smiles = data.Smiles.tolist()
#print(smiles.shape)
print(len(smiles))

# Process in batches
for it in range(35, 76):
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

    embedding_file = open("/mnt/smile_embeddings_training_"+str(start_count)+"_to_"+str(end_count)+".npy", "wb")
    np.save(embedding_file, smiles_embedding, allow_pickle=False)
    embedding_file.close()

    print("converting smiles")

    can_arr = np.array(decoded_smiles_list)

    print("writing embeddings")

    decoded_file = open("/mnt/decoded_smiles_training_"+str(start_count)+"_to_"+str(end_count)+".npy", "wb")
    np.save(decoded_file, can_arr, allow_pickle=False)
    decoded_file.close()

#sample = arr[:, 2]
#sample = np.reshape(sample, (arr.shape[0], 1))
#print(sample.shape)
#print(sample[0])
#print(type(sample[0][0]))


exit()

#print(sample[0:10].shape)
#print(sample[0:10, :])

sub_sample = sample[0:10, :]

#print(sub_sample)
#print(sub_sample.shape)

#print(type(sub_sample[0][0]))
#print(sub_sample[0][0])
inference_model = InferenceModel()
#smiles_embedding = inference_model.seq_to_emb(sub_sample[0][0])
#decoded_smiles_list = inference_model.emb_to_seq(smiles_embedding)
#print(smiles_embedding.shape)
#print(decoded_smiles_list)

#embedding_array = np.zeros((sample.shape[0], 512))
#canonical_smile_array = np.zeros((sample.shape[0], 1))

for count in range(16, 700):


    embedding_array = []
    canonical_smile_array = []

    start_it = count*100
    end_it = start_it+100

    print("start:", start_it, " end:", end_it)

    if end_it > sample.shape[0]:
        end_it = sample.shape[0]

    for it in range(start_it, end_it):
        emb = inference_model.seq_to_emb(sample[it][0])
        embedding_array.append(emb[0])
        dec = inference_model.emb_to_seq(emb)
        canonical_smile_array.append(dec)

    #print(len(embedding_array))
    #print(embedding_array[0].shape)
    #print(embedding_array[0][0][0:5])
    #print(len(canonical_smile_array))
    #print(canonical_smile_array[0])

    emb_arr = np.array(embedding_array)
    can_arr = np.array(canonical_smile_array)
    #print(emb_arr.shape)
    #print(can_arr.shape)

    #print(embedding_array.shape)
    #print(canonical_smile_array.shape)

    embedding_file = open("/mnt/smile_embeddings_"+str(start_it)+"_to_"+str(end_it)+".npy", "wb")
    np.save(embedding_file, emb_arr, allow_pickle=False)
    embedding_file.close()

    decoded_file = open("/mnt/decoded_smiles_"+str(start_it)+"_to_"+str(end_it)+".npy", "wb")
    np.save(decoded_file, can_arr, allow_pickle=False)
    decoded_file.close()

exit()

inference_model = InferenceModel()
smiles_embedding = inference_model.seq_to_emb(sub_sample)
embedding_file = open("/mnt/smile_embeddings.npy", "wb")
print("embedding: ", smiles_embedding.shape)
np.save(embedding_file, smiles_embedding)
embedding_file.close()

decoded_smiles_list = inference_model.emb_to_seq(smiles_embedding)
decoded_file = open("/mnt/decoded_smiles.npy", "wb")
print("decoded: ", decoded_smiles_list.shape)
np.save(decoded_file, decoded_smiles_list)
decoded_file.close()



