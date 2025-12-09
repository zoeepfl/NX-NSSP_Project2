import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

from scipy.io import loadmat 
from scipy.ndimage import convolve1d
from scipy.signal import butter
from scipy.signal import sosfiltfilt
from scipy.signal import welch
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import seaborn as sns

from lib import *



all_subjects = []  # liste qui contiendra les données des 27 sujets

for i in range(1, 28):  # sujets 1 à 27
    filename = f"dataset/s{i}/S{i}_A1_E1.mat"   # construit le chemin
    data = sio.loadmat(filename)                # charge le fichier
    all_subjects.append(data)                   # stocke dans la liste


print(f"Nombre de sujets chargés : {len(all_subjects)}")

for i in range(len(all_subjects)):
    emg = all_subjects[i]['emg']
    stimulus = all_subjects[i]['restimulus']
    repetition = all_subjects[i]['rerepetition']
    fs = 2000
    n_samples, n_channels = emg.shape
    time = np.arange(n_samples) / fs
    print(f"Sujet {i+1} : EMG shape = {emg.shape}, Stimulus shape = {stimulus.shape}, Repetition shape = {repetition.shape}")
    #plot_global_psd_check(emg, fs=2000)




# # Load data
# S2_A1_E1 = sio.loadmat('dataset/s2/S2_A1_E1.mat')
# # print(S2_A1_E1.keys())

# emg = S2_A1_E1['emg']
# # print(emg.shape)
# fs = 2000
# n_samples, n_channels = emg.shape
# time = np.arange(n_samples) / fs
# plot_global_psd_check(emg, fs=2000)

# stimulus = S2_A1_E1['restimulus']
# repetition = S2_A1_E1['rerepetition']