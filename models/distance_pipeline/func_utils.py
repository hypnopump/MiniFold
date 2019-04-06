""" File to store functions (DRY) and params that are repeated
    Store config only in one site
"""
import numpy as np 
import keras
# Logs-related imports
import sys
import logging
import os

# Logging info: - For reproducibility // 
# np.random.seed(5)

# PATHS
log_name = str(sys.argv[1])
LOG_PATH = str(os.getcwd())+"/"+log_name+".txt"
RECORD_PATH = "record.txt"
BASE_MODEL_PATH = "models/tester_28_lxl.h5"
STAGE_MODEL_PATH = "models/tester_28_lxl_stage.h5"
GOLDEN_MODEL_PATH = "models/tester_28_lxl_golden.h5"
IMAGES_PATH = "images/golden_img_v"+str(sys.argv[2])+"_"
TRAINING_PATHS = ["../../data/distanced/90_full_under_200.txt"] 
                # "../data/distanced/70_full_under_200.txt"]
EVAL_SOURCE_PATH = "../../data/distanced/full_under_200.txt"

# PARAMS - DEFINE MODEL PARAMS
CROP_SIZE = 200                                      # Crops to feed the mmodel   
PAD_SIZE = 200                                       # Maximum length of any protein
CLASS_CUTS = [-0.5, 500, 750, 1000, 1400, 1700] # , 2000 # Cuts between classes 
N_CLASSES = len(CLASS_CUTS)+1                        # Number of classes
BATCH_SIZE = 2                                       # Size of each batch
MAX_PROTS = 1500 # 3500                              # Max number of prots in the dataset
BATCH_RATIO = 0.1 # Proportion of batches/prots (don't train on whole dataset)
print(str(sys.argv[3]).split(","))
WEIGHTS = [float(w) for w in str(sys.argv[3]).split(",")] # Class weights for the model

# EVAL PARAMS
MAX_PROTS_EVAL = 130
MAX_PROTS_PREDICT = 125 # 125


# Record settings
# LOG_FORMAT = "%(levelname)s %(asctime)s - %(message)s"
LOG_FORMAT = "%(asctime)s - %(message)s"
logging.basicConfig(filename=LOG_PATH,
                    format = LOG_FORMAT,
                    level = logging.DEBUG,
                    filemode = "a")
logger = logging.getLogger()


# Model-related variables 
kernel_size, filters = 3, 16
adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)


# FUNC_UTILS
def parse_lines(raw):
    return [[float(x) for x in line.split("\t") if x != ""] for line in raw]

def parse_line(line):
    return [float(x) for x in line.split("\t") if x != ""]


def get_data(paths, max_prots):
        """ Get the data from files. """
        # Scan first n proteins
        names, seqs, dists, pssms = [], [], [], [] 

        for path in paths: 
            # Opn file and read text
            with open(path, "r") as f:
                lines = f.read().split('\n')

            # Extract numeric data from text
            for i,line in enumerate(lines):
                if len(names) == max_prots+1:
                    break
                # Read each protein separately
                if line == "[ID]":
                    names.append(lines[i+1])
                elif line == "[PRIMARY]":
                    seqs.append(lines[i+1])
                elif line == "[EVOLUTIONARY]":
                    pssms.append(parse_lines(lines[i+1:i+21]))
                elif line == "[DIST]":
                    dists.append(parse_lines(lines[i+1:i+len(seqs[-1])+1]))
                    # Progress control
                    if len(names)%150 == 0:
                        print("Currently @ ", len(names), " out of "+str(max_prots))
                        try: logger.info("Currently @ "+str(len(names))+" out of "+str(max_prots))
                        except:pass

        print("Total length is "+str(len(names)-1)+" out of "+str(max_prots)+" possible.")
        try: logger.info("Total length is "+str(len(names)-1)+" out of "+str(max_prots)+" possible.")
        except:pass

        return names, seqs, dists, pssms


def wider(seq, l=200, n=20):
    """ Converts a seq into a one-hot tensor. Not LxN but LxLxN"""
    key = "HRKDENQSYTCPAVLIGFWM"
    tensor = []
    for i in range(l):
        d2 = []
        for j in range(l):
            d1 = [1 if (j<len(seq) and i<len(seq) and key[x] == seq[i] and key[x] == seq[j]) else 0 for x in range(n)]
    
            d2.append(d1)
        tensor.append(d2)
    
    return np.array(tensor)


def wider_pssm(pssm, seq, l=200, n=20):
    """ Converts a seq into a one-hot tensor. Not LxN but LxLxN"""
    key = "HRKDENQSYTCPAVLIGFWM"
    key_alpha = "ACDEFGHIKLMNPQRSTVWY"
    tensor = []
    for i in range(l):
        d2 = []
        for j in range(l):
            if j<len(seq) and i<len(seq):
                d1 = [aa[i]*aa[j] for aa in pssm]
            else:
                d1 = [0 for i in range(n)]
                
            # Append pssm[i]*pssm[j]
            if j<len(seq) and i<len(seq):
                d1.append(pssm[key_alpha.index(seq[i])][i] *
                          pssm[key_alpha.index(seq[j])][j])
            else: 
                d1.append(0)
            # Append custom distance to diagonal formula (1-abs(i-j))/crop_size
            # Works bad when i,j > len(seq)
            d1.append(1 - abs(i-j)/200)
    
            d2.append(d1)
        tensor.append(d2)
    
    return np.array(tensor)


def embedding_matrix(matrix, l=200):
    """ Embeds matrix of nxn into lxl. n<L """
    # Embed with extra columns
    for i in range(len(matrix)):
        while len(matrix[i])<l:
            matrix[i].extend([-1 for i in range(l-len(matrix[i]))])
    #Embed with extra rows
    while len(matrix)<l:
        matrix.append([-1 for x in range(l)])
    return np.array(matrix)


def treshold(matrix, cuts=None, l=200): 
    # Turns an L*L*1 tensor into an L*L*N 
    trash = (np.array(matrix)<cuts[0]).astype(np.int)
    first = (np.array(matrix)<cuts[1]).astype(np.int)-trash
    sec = (np.array(matrix)<cuts[2]).astype(np.int)-trash-first
    third = (np.array(matrix)<cuts[3]).astype(np.int)-trash-first-sec
    fourth = (np.array(matrix)<cuts[4]).astype(np.int)-trash-first-sec-third
    fifth = (np.array(matrix)<cuts[5]).astype(np.int)-trash-first-sec-third-fourth
#     sixth = (np.array(matrix)<cuts[6]).astype(np.int)-trash-first-sec-third-fourth-fifth
    seventh = np.array(matrix)>=cuts[5]

    return np.concatenate((trash.reshape(l,l,1),
                           first.reshape(l,l,1),
                           sec.reshape(l,l,1),
                           third.reshape(l,l,1),
                           fourth.reshape(l,l,1),
                           fifth.reshape(l,l,1),
                           # sixth.reshape(l,l,1),
                           seventh.reshape(l,l,1)),axis=2)

def mirror_diag(image):
    """ Mirrors image across its diagonal. """
    image = image.astype(float)
    # averages image across diagonal and returns 2 simetric parts
    for i in range(len(image)):
        for j in range(len(image[i])):
            image[i,j] = image[j,i] = np.true_divide((image[i,j]+image[j,i]), 2)
         
    return image