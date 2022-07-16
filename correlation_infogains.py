import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import tree

#base dataset
data = pd.read_csv("data-final.csv")
#remove entries at IP addresses from which multiple responses were sent
data = data[data['IPC'] == 1].dropna()
#remove country-less entries
data = data[data['country'] != 'NONE']
#responses only
R = data.loc[:,'EXT1':'OPN10']
R = R.astype('int')
#segments
EXT = R.loc[:,'EXT1':'EXT10']
EST = R.loc[:,'EST1':'EST10']
AGR = R.loc[:,'AGR1':'AGR10']
CSN = R.loc[:,'CSN1':'CSN10']
OPN = R.loc[:,'OPN1':'OPN10']

## compare internal/external correlations
corr = R.corr().to_numpy()
in_cat = 0
in_catc = 0
out_cat = 0
out_catc = 0

for i in range(len(corr)):
    for j in range(i):
        if (j//10) == (i//10):
            in_cat += abs(corr[i][j])
            in_catc += 1
        else:
            out_cat += abs(corr[i][j])
            out_catc += 1

print(f"in cat avg = {in_cat/in_catc}")
print(f"out cat avg = {out_cat/out_catc}")


#Compute information gains
for target in R.columns:  #['EXT7','EST1','AGR8','CSN3','OPN7']
    print(f"target = {target}")
    #compute entropy of the target
    vals = R[target].to_numpy()
    counts = [0,0,0,0,0]
    for i in range(len(vals)):
        counts[vals[i]-1] += 1
    total = sum(counts)
    entropy = -1 * sum([(x/total)*np.log2(x/total) for x in counts])
    #print(entropy)
    print(f'information gains:')
    #compute information gain from each of the remaining attributes
    for col in R.columns:
        if col != target:
            #get counts of each value of the attribute(col)
            col_vals = R[col].to_numpy()
            col_counts = [0,0,0,0,0]

            for i in range(len(col_vals)):
                col_counts[col_vals[i]-1] += 1
            col_total = sum(col_counts)

            #compute entropy of the target given values of col
            sub_entropies = [0,0,0,0,0]
            for i in range(1,6):
                #count values of the target given a particular value of col
                sub_counts = [0,0,0,0,0]
                for j in range(len(vals)):
                    if col_vals[j] == i:
                        sub_counts[vals[j]-1]+= 1
                sub_total = sum(sub_counts)
                sub_entropies[i-1] = -1*sum([(x/sub_total)*np.log2(x/sub_total) for x in sub_counts])

            col_entropy = sum([(col_counts[i]/col_total)*sub_entropies[i] for i in range(len(col_counts))])
            
            print(f'{col} : {entropy-col_entropy}')
