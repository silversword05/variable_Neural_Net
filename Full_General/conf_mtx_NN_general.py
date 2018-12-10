import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

f_act = open("data_output_test.txt", "r")
f_predc = open("results_test.txt", "r")

f = open("instructions.txt", "r")
lns = f.readlines()
optLyr = int(lns[3].strip(' \n'))  # fourth line should contain no of nodes in output layer

cnf_mtx = np.zeros(shape=(optLyr, optLyr), dtype=int)
# rows are actual and columns are predicted

for lns1, lns2 in zip(f_act, f_predc):
    intgs1 = [(float(x)) for x in lns1.split()]
    intgs2 = [(float(x)) for x in lns2.split()]
    act = np.argmax(intgs1)
    prdc = np.argmax(intgs2)
    cnf_mtx[act][prdc] += 1

df_cm = pd.DataFrame(cnf_mtx, index = [i for i in range(optLyr) ], columns = [i for i in range(optLyr) ] )
plt.figure(figsize = (optLyr,optLyr))
sn.heatmap(df_cm, annot=True)
plt.xlabel("Predicted Values")
plt.ylabel("Actual Values")
plt.title("Confusion Matrix")
plt.show(sn)

f_act.close()
f_predc.close()

f_cnf = open("confusion_matrix.txt", "w")
f_cnf.write('\n'.join('\t'.join('{:3}'.format(item) for item in row) for row in cnf_mtx))
f_cnf.close()
