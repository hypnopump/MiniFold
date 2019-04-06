# Import libraries
import numpy as np
import matplotlib.pyplot as plt
# Import libraries
import keras
import keras.backend as K
from keras.models import Model, load_model, Sequential
# Activation and Regularization
from keras.regularizers import l2
from keras.activations import softmax
# Keras layers
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers import Dense, Dropout, Flatten, Input, BatchNormalization, Activation
from keras.layers.pooling import MaxPooling2D, AveragePooling2D

# Model architecture
from elu_resnet_2d_distances import *
from func_utils import *
# Logs-related imports
import sys
import logging
import os


# Log progress
logger.info("Gonna evaluate the model")
logger.info("Loading data")

path = "../../data/distanced/full_under_200.txt"
# Opn file and read text
with open(path, "r") as f:
	lines = f.read().split('\n')

# Scan first n proteins
names, seqs, dists, pssms = get_data(paths=[EVAL_SOURCE_PATH], max_prots=MAX_PROTS_EVAL)

# preprocess inputs
inputs_aa = np.array([wider(seq) for seq in seqs])
inputs_aa.shape
inputs_pssm = np.array([wider_pssm(pssms[i], seqs[i]) for i in range(len(pssms))])
inputs_pssm.shape
inputs = np.concatenate((inputs_aa, inputs_pssm), axis=3)
print(inputs.shape)
# Delete unnecessary data
del inputs_pssm
del inputs_aa
dists = np.array([embedding_matrix(matrix) for matrix in dists])
outputs = np.array([treshold(d, CLASS_CUTS) for d in dists])
print(outputs.shape)
del dists

# Log progress
logger.info("Data loaded correctly")
logger.info("Load the model")

# Set WEIGHTS
print("WEIGHTS", WEIGHTS)
 
# Load model
model = load_model(STAGE_MODEL_PATH,
			custom_objects={'loss': weighted_categorical_crossentropy(np.array(WEIGHTS)),
			'softMaxAxis2': softMaxAxis2})	
model.compile(optimizer=adam,
			  loss=weighted_categorical_crossentropy(np.array(WEIGHTS)),
			  metrics=["accuracy"])

# Log progress
logger.info("Model loaded")
logger.info("Let's predict")

# Predict and benchmark
i,k = 0, MAX_PROTS_PREDICT
sample_pred = model.predict([inputs[i:i+k]])
preds4 = np.argmax(sample_pred, axis=3)
preds4[preds4==0] = 6 # Change trash class by class for long distance
outs4 = np.argmax(outputs[i:i+k], axis=3)
outs4[outs4==0] = 6 # Change trash class by class for long distance
# Select the best prediction to display it - (proportional by protein length(area of contact map))
results = [np.sum(np.equal(pred[:len(seqs[i+j]), :len(seqs[i+j])], outs4[j, :len(seqs[i+j]), :len(seqs[i+j]),]),axis=(0,1))/
			len(seqs[i+j])**2 
			for j,pred in enumerate(preds4)]
best_score = max(results)
print("Best score (Accuracy): ", best_score)
sorted_scores = [acc for acc in sorted(results, key=lambda x: x, reverse=True)]
print("Best 5 scores: ", sorted_scores[:5])
sorted_indices = [results.index(x) for x in sorted_scores]
print("Best 5 indices: ", sorted_indices[:5])

# Log results: 
logger.info("Best 10 scores: "+str(sorted_scores[:10]))
logger.info("Best 10 indices: "+str(sorted_indices[:10]))


# Measure metrics and decide if model is better!
preds_crop = np.concatenate( [pred[:len(seqs[i+j]), :len(seqs[i+j])].flatten() for j,pred in enumerate(preds4)] )
outs_crop = np.concatenate( [outs4[j, :len(seqs[i+j]), :len(seqs[i+j])].flatten() for j,pred in enumerate(preds4)] )

total_mse = np.linalg.norm(outs_crop-preds_crop)
mse_prot = np.linalg.norm(outs_crop-preds_crop)/len(preds4)

print("Introducing total mse: ", total_mse)
print("Introducing mse/prot: ", mse_prot)
logger.info("total_mse: "+str(total_mse))
logger.info("mse/prots: "+str(mse_prot))

# loading previous record. If not, make sure this one gets written.
try: 
	with open(RECORD_PATH, "r") as f:
		lines = f.read().split('\n')

	prev_total_mse = float(lines[0].split(" ")[-1])
	prev_mse_prot = float(lines[1].split(" ")[-1])
except: 
	prev_total_mse = 1e8
	prev_mse_prot = 1e8

# compare metrics
if total_mse < prev_total_mse and mse_prot < prev_mse_prot:
	print("model is better. saving. prev mse/prot: "+str(prev_mse_prot)+" new: "+str(mse_prot))
	logger.info("model is better. saving. prev mse/prot: "+str(prev_mse_prot)+" new: "+str(mse_prot))
	# save model and new records
	model.save(GOLDEN_MODEL_PATH)
	with open(RECORD_PATH, "w") as f:
		f.write("total_mse: "+str(total_mse)+"\n")
		f.write("mse/prot: "+str(mse_prot)+"\n")
		# Load WEIGHTS set that produced best outcome till now
		f.write(str(sys.argv[2])+" : "+str(sys.argv[3]))
	print("model saved. Producing images")
	logger.info("model saved.  Producing images")

	# Avoid matplotlib logs
	logger.setLevel(40)
	# Make images 
	for best_score_index in sorted_indices[:3]+[i for i in range(10)]:
		plt.figure(figsize=(20,5))
		# First plot Ground Truth
		plt.subplot(1, 3, 1)
		plt.title('Ground Truth')
		plt.imshow(outs4[best_score_index, :len(seqs[i+best_score_index]), :len(seqs[i+best_score_index])],
					cmap='viridis_r', interpolation='nearest')
		plt.colorbar()
		plt.clim(0, N_CLASSES-1)
		# Then plot predictions by the mode
		plt.subplot(1, 3, 2)
		plt.title("Prediction by model")
		plt.imshow(preds4[best_score_index, :len(seqs[i+best_score_index]), :len(seqs[i+best_score_index])],
				cmap='viridis_r', interpolation='nearest')
		plt.colorbar()
		plt.clim(0, N_CLASSES-1)
		# Prediction by model (diag mirrored)
		plt.subplot(1, 3, 3)
		plt.title("Prediction by model - mirrored+averaged")
		plt.imshow(mirror_diag(preds4[best_score_index, :len(seqs[i+best_score_index]), :len(seqs[i+best_score_index])]),
					cmap='viridis_r', interpolation='nearest')
		plt.colorbar()
		plt.clim(0, N_CLASSES-1)
		# Show/Save them
		plt.savefig(IMAGES_PATH+str(best_score_index)+".png")
		# plt.show()

	# Avoid matplotlib logs
	logger.setLevel(10)
	print("Images are ready")
	logger.info("Images are ready")
else: 
	print("Model is no better than previous one. Not saving")  	
	logger.info("Model is no better than previous one. Not saving")

print("Evaluation finished")
logger.info("Evaluation finished")
logger.setLevel(40) 