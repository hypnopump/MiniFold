import keras
import numpy as np

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, seqs, pssms, dists, batch_size=8, crop_size=200, pad_size=200,
                 n_classes=5, class_cuts=[-0.5, 500, 1000, 1700], shuffle=True):
        'Initialization'
        # Get data
        self.seqs = seqs
        self.pssms = pssms
        self.dists = dists
        self.list_IDs = [i for i in range(len(self.seqs))]
        # Features
        self.batch_size = batch_size
        self.crop_size = crop_size
        self.pad_size = pad_size
        self.n_classes = n_classes
        self.class_cuts = class_cuts
        if len(self.class_cuts) != self.n_classes-1:
            raise ValueError('len(class_cuts) must be n_classes-1')

        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        x, y = self.__data_generation(list_IDs_temp)

        return x, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
                                                     
    def wider(self, seq, n=20):
        """ Converts a seq into a one-hot tensor. Not LxN but LxLxN"""
        key = "HRKDENQSYTCPAVLIGFWM"
        tensor = []
        for i in range(self.pad_size):
            d2 = []
            for j in range(self.pad_size):
                # Check first for lengths (dont want index_out_of_range)
                d1 = [1 if (j<len(seq) and i<len(seq) and key[x] == seq[i] and key[x] == seq[j])
                      else 0 for x in range(n)]

                d2.append(d1)
            tensor.append(d2)

        return np.array(tensor)
                                                     
    def wider_pssm(self, pssm, seq, n=20):
        """ Converts a seq into a one-hot tensor. Not LxN but LxLxN"""
        key = "HRKDENQSYTCPAVLIGFWM"
        key_alpha = "ACDEFGHIKLMNPQRSTVWY"
        tensor = []
        for i in range(self.pad_size):
            d2 = []
            for j in range(self.pad_size):
                # Check first for lengths (dont want index_out_of_range)
                if (i == j and j<len(seq) and i<len(seq)):
                    d1 = [aa[i] for aa in pssm]
                else:
                    d1 = [0 for i in range(n)]

                # Append pssm[i]*pssm[j]
                if j<len(seq) and i<len(seq):
                    d1.append(pssm[key_alpha.index(seq[i])][i] *
                              pssm[key_alpha.index(seq[j])][j])
                else: 
                    d1.append(0)
                # Append manhattan distance to diagonal but reversed (center=0, xtremes=1)
                d1.append(1 - abs(i-j)/self.crop_size)

                d2.append(d1)
            tensor.append(d2)

        return np.array(tensor)
                                                     
    def treshold(self, matrix):
        """ Turns an L*L*1 tensor into an L*L*N """
        med = []
        # Create labels for classes
        for i,cat in enumerate(self.class_cuts):
            if i < self.n_classes-1:
                inter = (np.array(matrix)<self.class_cuts[i]).astype(np.int)
                # Subtract all previous (don't do it for first one)
                if i != 0: 
                    for prev in med: 
                        inter -= prev

            med.append(inter)

        # Append last class (greater than lass class_cut)
        med.append((np.array(matrix)>=self.class_cuts[-1]).astype(np.int))

        return np.concatenate([cat.reshape(self.pad_size, self.pad_size, 1) 
                               for cat in med], axis=2)
                                                     
    # Embed number of rows
    def embedding_matrix(self, matrix):
        # Embed with extra columns
        for i in range(len(matrix)):
            while len(matrix[i])<self.pad_size:
                matrix[i].extend([-1 for i in range(self.pad_size-len(matrix[i]))])
        #Embed with extra rows
        while len(matrix)<self.pad_size:
            matrix.append([-1 for x in range(self.pad_size)])
        return np.array(matrix)

    # Get indices to crop from a splice of (crop_size x crop_size)
    def random_indices(self, list_IDs_temp):
        indices = []
        for i in list_IDs_temp:
            if len(self.seqs[i])<=self.crop_size:
                indices.append([0,0])
            else:
                indices.append([np.random.randint(0, len(self.seqs[i])-self.crop_size),
                                np.random.randint(0, len(self.seqs[i])-self.crop_size)])
        return indices

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        # Get the random indices to crop from
        r_i = self.random_indices(list_IDs_temp) 
        # Get data
        inputs_aa = np.array([self.wider(self.seqs[i])[r_i[k][0]:r_i[k][0]+self.crop_size, r_i[k][1]:r_i[k][1]+self.crop_size]
                              for k,i in enumerate(list_IDs_temp)])
        inputs_pssm = np.array([self.wider_pssm(self.pssms[i], self.seqs[i])[r_i[k][0]:r_i[k][0]+self.crop_size, r_i[k][1]:r_i[k][1]+self.crop_size]
                                for k,i in enumerate(list_IDs_temp)])
        x = np.concatenate((inputs_aa, inputs_pssm), axis=3)
        # Get labels
        distas = np.array([self.embedding_matrix(self.dists[i]) for i in list_IDs_temp])                      
        y = np.array([self.treshold(d)[r_i[k][0]:r_i[k][0]+self.crop_size, r_i[k][1]:r_i[k][1]+self.crop_size]
                      for k,d in enumerate(distas)])
        
        # try to free some memory
        del inputs_aa
        del inputs_pssm
        del distas
        del r_i

        return x,y