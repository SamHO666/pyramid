
# coding: utf-8

# # Import Necessary Package

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# # Load Data from CSV

# In[4]:


DS1 = ['101', '106', '108', '109', '112', '114', '115', '116', '118', '119', '122', '124', 
       '201', '203', '205', '207', '208', '209', '215', '220', '223', '230']

trainingSet_leads = {}
trainingSet_anns = {}
for ds in DS1:
    trainingSet_leads[ds] = pd.read_csv('Cleaned TrainingSet/' + ds + '_lead.csv')
    trainingSet_anns[ds] = pd.read_csv('Cleaned TrainingSet/' + ds + '_ann.csv')






Non_beat_anns = ['[', ']', '!', 'x', '(', ')', 'p', 't', 'u', '`', '~', '^', '|', '+', 's', 'T', '*', 'D', '=', '"', '@']
Hbs = {}
for ds in DS1:
    lead0 = trainingSet_leads[ds]['lead0']
    lead1 = trainingSet_leads[ds]['lead1']
    hbs0 = []
    hbs1 = []
    anns = []
    annIdxs = []
    for row in trainingSet_anns[ds].itertuples():
        if row[2] in Non_beat_anns:
            continue
        elif row[1] < 91:
            continue
        elif row[1] + 144 > len(lead0):
            continue
        else:
            anns.append(row[2])
            annIdxs.append(row[1] - 1)
            hbs0.append(lead0[row[1] - 91: row[1] + 144])
            hbs1.append(lead1[row[1] - 91: row[1] + 144])
    Hbs[ds] = pd.DataFrame({'lead0': hbs0, 'lead1': hbs1, 'ann': anns, 'annIdx': annIdxs})

# 测试segmentation情况
Hbs['109']



# In[110]:


from QRSDetector7 import QRSDetectorOffline


# In[111]:


t = np.array(trainingSet_leads['109']['lead0'][:5000])
f = np.arange(len(t))
ff = np.column_stack((f,t))



qrs_detector = QRSDetectorOffline(ecg_data_input=ff, verbose=True, log_data=False, plot_data=False, show_plot=False)






