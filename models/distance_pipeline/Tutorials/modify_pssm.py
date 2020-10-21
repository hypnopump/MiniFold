with open("file_name.pssm", "r") as f:
    lines = f.readlines()[2:-6]
key  = "ACDEFGHIKLMNPQRSTVWY"
text_keys = lines[0].replace("\n", "").split("   ")[4:]

# PSSM OPTION 1
text_vals = np.array([" ".join(line.replace("\n", "")[72:-11].split()).split() for line in lines[1:]]).astype(float)

# PSSM OPTION 2
# text_vals = np.array([" ".join(line.replace("\n", "")[10:72].split()).split() for line in lines[1:]]).astype(float)

# normalize to [0,1]
# text_vals = text_vals / np.sum(text_vals, axis=1).reshape((len(text_vals), 1))
for i in range(len(text_vals)):
    text_vals[i] = (text_vals[i] - np.amin(text_vals[i])) / (np.amax(text_vals[i] - np.amin(text_vals[i])))

# create NxL PSSM
pssm      = np.zeros_like(text_vals)
for i,aa in enumerate(text_keys):
    pssm[key.index(aa), :] = text_vals[i, :] 
    
inputs_pssm = wider_pssm(pssm.T, seq)
inputs = np.concatenate((inputs_aa, inputs_pssm), axis=-1)
