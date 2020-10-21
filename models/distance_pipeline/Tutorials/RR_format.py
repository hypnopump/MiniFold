RR_FORMAT = """PFRMAT RR
MODEL 1
"""
def save_rr(seq,sample_pred,file_name_rr):
  my_file=open(file_name_rr,'w')
  my_file.write(RR_FORMAT)
  my_file.write(seq + '\n')
  for i in range(0,len(seq)):
    for j in range(i+5,len(seq)):
      print(max(sample_pred[0][i][j]))
      my_file.write(str(i+1)+" "+ str(j+1)+" "+"0 8 "+ str(max(sample_pred[0][i][j]))+"\n")
  my_file.write('END\n')
  my_file.close()
save_rr(seq,sample_pred,'file_name_rr')
