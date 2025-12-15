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
from statsmodels.regression.linear_model import yule_walker

import seaborn as sns

from lib import *

#--------------------------------------------------------------------------

# Load data part 3
S1_E1_A1 = sio.loadmat('S1_E1_A1.mat')
#print(f"data structure is:{type(S1_E1_A1)}")                  ### represents class  --->  <class 'dict'>
#print(f"keys:{[key for key in S1_E1_A1.keys()]}")             ### gives categories---> ['__header__', '__version__', '__globals__', 'subject', 'exercise', 'emg', 'acc', 'gyro', 'mag', 'glove', 'stimulus', 'repetition', 'restimulus', 'rerepetition']

fs = 2000

emg = S1_E1_A1['emg']
#print(emg_P3.shape)                                           ### dim -> (2292526, 16)

subject = S1_E1_A1['subject']                                  ### dim -> (1, 1)
exercise = S1_E1_A1['exercise']                                ### dim -> (1, 1)
acc = S1_E1_A1['acc']                                          ### dim -> (2292526, 48)
gyro = S1_E1_A1['gyro']                                        ### dim -> (2292526, 48)
mag = S1_E1_A1['mag']                                          ### dim -> (2292526, 48)
glove = S1_E1_A1['glove']                                      ### dim -> (2292526, 18)
stimulus = S1_E1_A1['stimulus']                                ### dim -> (2292526, 1)
repetition = S1_E1_A1['repetition']                            ### dim -> (2292526, 1)
restimulus = S1_E1_A1['restimulus']                            ### dim -> (2292526, 1)
rerepetition = S1_E1_A1['rerepetition']                        ### dim -> (2292526, 1)

#print(f"The shapes are:\n subject : {subject.shape} \n exercise : {exercise.shape} \n acc : {acc.shape} \n gyro : {gyro.shape} \n mag : {mag.shape} \n glove : {glove.shape} \n stimulus : {stimulus.shape} \n repetition : {repetition.shape} \n restimulus : {restimulus.shape} \n rerepetition : {rerepetition.shape}")


# GLOVE TREATEMENT
joint_channels = [3, 6, 8, 11, 14]
glove_new = glove[:, joint_channels]

n_timepoints, n_channels = glove_new.shape

n_channels_emg = emg.shape[1]

time_steps = np.arange(0, n_timepoints/fs, 1/fs)

#This proves that the code needs rectification
#t, c = np.where(glove < 0)
#print(f"Valeur en dessous de 0 pour {c} at {t}")

#--------------------------------------------------------------------------
# PLOTS FOR ORIGINAL SIGNALS
# fig1, ax1 = plt.subplots()
# ax1.plot(emg[:,5])
# ax1.set_xlabel("Time [s]")
# ax1.set_ylabel("Signal [uV]")
# plt.show()

# fig, ax = plt.subplots(n_channels, 1, constrained_layout=True, figsize=(10,5))
# for i, ch in enumerate(joint_channels):
#     ax[i].plot(time_steps, glove[:,ch])
#     ax[i].set_xlabel("Time [s]")
#     ax[i].set_ylabel(f"Signal {ch} [uV]")
# plt.show()

#--------------------------------------------------------------------------
#(OPTIONAL GLOVE FILTERING)
# For a glove signal we do not need all that filtering 
#lowpass_cutoff = 5  # Hz
#sos = butter(N=4, Wn=lowpass_cutoff, fs=fs, btype="lowpass", output="sos")
#glove_filtered = sosfiltfilt(sos, glove_new, axis=0)

#fig2, ax2 = plt.subplots(n_channels, 1, constrained_layout=True, figsize=(10,5))
#for i, ch in enumerate(joint_channels):
#    ax2[i].plot(time_steps, glove_filtered[:,i])
#    ax2[i].set_xlabel("Time [s]")
#    ax2[i].set_ylabel(f"Signal {ch} [uV]")
#plt.show()

################################################
# PREPROCESSING
# 1. Filter
#    Bandpass
bandpass_cutoff_frequencies_Hz = (5, 500)       
sos = butter(N=4, Wn=bandpass_cutoff_frequencies_Hz, fs=fs, btype="bandpass", output="sos")    
emg_filtered = sosfiltfilt(sos, emg.T).T 

powergrid_noise_frequencies_Hz = [harmonic_idx*50 for harmonic_idx in range(1,3)] 
for noise_frequency in powergrid_noise_frequencies_Hz:
    sos = butter(N=4, Wn=(noise_frequency - 2, noise_frequency + 2), fs=fs, btype="bandstop", output="sos")
    emg_filtered = sosfiltfilt(sos, emg_filtered.T).T

#  Apply TKEO (ignore 1st and last sample)
emg_tkeo = np.copy(emg_filtered)
emg_tkeo[1:-1] = emg_filtered[1:-1]**2 - emg_filtered[:-2] * emg_filtered[2:]

# 2. Rectify the signal
emg_rectified = np.abs(emg_tkeo)
################################################
# fig1, ax1 = plt.subplots()
# ax1.plot(emg_rectified[:,5])
# ax1.set_xlabel("Time [s]")
# ax1.set_ylabel("Signal [uV]")
# plt.show()

# Moving mean to have the spectrum of the envelope signal
mov_mean_size = 200 
mov_mean_weights = np.ones(mov_mean_size) / mov_mean_size 
EMG_envelopes = convolve1d(emg_rectified, weights=mov_mean_weights, axis=0) 

fig1, ax1 = plt.subplots()
ax1.plot(EMG_envelopes[:,5])
ax1.set_xlabel("Time [s]")
ax1.set_ylabel("Signal [uV]")
plt.show()
#--------------------------------------------------------------------------
# PLOT
fig, ax = plt.subplots() 
ax.plot(EMG_envelopes[:, 5],color = 'cornflowerblue', label="EMG envelope") 
ax.set_ylabel("EMG envelope") 
ax.set_xlabel("Time samples") 
ax.grid(False) 
finger_angle_ax = ax.twinx() 
finger_angle_ax.plot(glove_new[:, :], color="darkorange", label="Angle of finger") 
finger_angle_ax.set_ylabel("Angle of finger") 
finger_angle_ax.grid(False) 
# Get handles and labels for legend 
emg_handles, emg_labels = ax.get_legend_handles_labels() 
finger_angle_handles, finger_angle_labels = finger_angle_ax.get_legend_handles_labels() 
# Combine handles and labels 
combined_handles = emg_handles + finger_angle_handles 
combined_labels = emg_labels + finger_angle_labels 
# Create a single legend 
ax.legend(combined_handles, combined_labels, loc='upper left') 
plt.tight_layout()
plt.show()
#--------------------------------------------------------------------------
# Chronologically cut the whole dataset before creating overlapping windows 
i_train = int(0.70 * n_timepoints)
i_val = int(0.85 * n_timepoints)

EMG_train = EMG_envelopes[:i_train]
Labels_train = glove_new[:i_train]

EMG_val = EMG_envelopes[i_train:i_val]
Labels_val = glove_new[i_train:i_val]

EMG_test = EMG_envelopes[i_val:]
Labels_test = glove_new[i_val:]

print(f"EMG train data shape: {EMG_train.shape}, Train label shape: {Labels_train.shape}") 
print(f"EMG validation data shape: {EMG_val.shape}, Validation label shape: {Labels_val.shape}")
print(f"EMG test data shape: {EMG_test.shape}, Test label shape: {Labels_test.shape}")

emg_window_length_sec = 200e-3 # [s] 
step_window_length_sec = 40e-3 # [s] 

# extract overlapping time windows on train set and test set 
EMG_train_windows, Labels_train_windows = extract_time_windows_regression(EMG_train, Labels_train, fs, emg_window_length_sec, step_window_length_sec) 
EMG_val_windows, Labels_val_windows = extract_time_windows_regression(EMG_val, Labels_val, fs, emg_window_length_sec, step_window_length_sec) 
EMG_test_windows, Labels_test_windows = extract_time_windows_regression(EMG_test, Labels_test, fs, emg_window_length_sec, step_window_length_sec) 
print(f"EMG train windows shape: {EMG_train_windows.shape}, Train label windows shape: {Labels_train_windows.shape}") 
print(f"EMG train windows shape: {EMG_val_windows.shape}, Train label windows shape: {Labels_val_windows.shape}") 
print(f"EMG test windows shape: {EMG_test_windows.shape}, Test label windows shape: {Labels_test_windows.shape}") 


EMG_train_extracted_features, Labels_train_mean = extract_features(EMG_train_windows,Labels_train_windows) 
EMG_val_extracted_features, Labels_val_mean = extract_features(EMG_val_windows,Labels_val_windows) 
EMG_test_extracted_features, Labels_test_mean = extract_features(EMG_test_windows,Labels_test_windows) 
print("EMG train extracted features shape: {}, Finger labels feature shape:{}".format(EMG_train_extracted_features.shape, Labels_train_mean.shape))
print("EMG validation extracted features shape: {}, Finger labels feature shape:{}".format(EMG_val_extracted_features.shape, Labels_val_mean.shape)) 
print("EMG test extracted features shape: {}, Finger labels feature shape:{}".format(EMG_test_extracted_features.shape, Labels_test_mean.shape))

scaler = StandardScaler()
X_train_z = scaler.fit_transform(EMG_train_extracted_features)
X_val_z   = scaler.transform(EMG_val_extracted_features)
X_test_z  = scaler.transform(EMG_test_extracted_features)

corr = pd.DataFrame(X_train_z).corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr, cmap="coolwarm", center=0, square=True)
plt.title("Feature correlation (z-scored, train set)")
plt.show()
