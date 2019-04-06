import os
import numpy as np
log_name = "genetic_log"

# genetic algorithm params
RECORD_PATH = "record.txt"
IMPROVE = 1/30 # maximum modification to eaach class
MUTATE = 0.75 # probability of a class mutation

def stringify(vec):
    """ Helper function to save data to .txt file. """
    line = ""
    for v in vec: 
        line += str(v)+","
    return line[:-1] 

for i in range(7*20):
	try:
		with open(RECORD_PATH, "r") as f:
			lines = f.read().split('\n')

		WEIGHTS = [float(w) for w in str(lines[-1]).split(" ")[-1].split(",")]
		# generate new_weights if 
		if int(str(lines[-1]).split(" ")[0]) < i-1:
			# -0.4 since its easier to lose a 50% but hard to regain a 100%
			WEIGHTS = [w+2*(np.random.random()-0.4)*IMPROVE*w 
					   if np.random.random()<MUTATE else w for w in WEIGHTS]
	except: 
		WEIGHTS = [0.0000001,0.45,1.65,1.75,0.73,0.77,0.145]

	os.system("python training_pipeline.py "+log_name+" "+str(i)+" "+stringify(WEIGHTS))
	os.system("python evaluation_pipeline.py "+log_name+" "+str(i)+" "+stringify(WEIGHTS))

	# print(WEIGHTS)
	