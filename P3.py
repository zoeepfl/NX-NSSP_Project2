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


# ==============================================================================
# DIAGNOSTIC TEST: SHUFFLED SPLIT
# ==============================================================================

print("\n" + "="*50)
print(">>> STARTING DIAGNOSTIC TEST: SHUFFLED SPLIT")
print("Goal: Check if model works when training data is randomized.")

# 1. Concatenate all available data (train + val + test)
# Merge everything to redistribute it randomly
X_total = np.concatenate([
    EMG_train_extracted_features, 
    EMG_val_extracted_features, 
    EMG_test_extracted_features
], axis=0)

Y_total = np.concatenate([
    Labels_train_mean, 
    Labels_val_mean, 
    Labels_test_mean
], axis=0)

# 2. Random split (Shuffle=True)
# Keep the same ratio (approx 15% for testing)
from sklearn.model_selection import train_test_split
X_train_shuf, X_test_shuf, Y_train_shuf, Y_test_shuf = train_test_split(
    X_total, Y_total, test_size=0.15, random_state=42, shuffle=True # delete random_state=42 to get a random set of values
)

# 3. New normalization
scaler_shuf = StandardScaler()
X_train_z_shuf = scaler_shuf.fit_transform(X_train_shuf)
X_test_z_shuf  = scaler_shuf.transform(X_test_shuf)

# 4. Train and evaluate on shuffled data
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error

finger_names_diag = ["J3", "J6", "J8", "J11", "J14"]
print(f"Training on {X_train_z_shuf.shape[0]} samples, Testing on {X_test_z_shuf.shape[0]} samples (Randomized)...")

for f in range(Y_total.shape[1]):
    # Train a temporary regressor
    gbr_diag = GradientBoostingRegressor(
        n_estimators=200, 
        learning_rate=0.05, 
        max_depth=3, 
        random_state=0
    )
    gbr_diag.fit(X_train_z_shuf, Y_train_shuf[:, f])
    
    # Predict
    y_pred_diag = gbr_diag.predict(X_test_z_shuf)
    
    # Calculate metrics
    rmse_diag = np.sqrt(mean_squared_error(Y_test_shuf[:, f], y_pred_diag))
    r2_diag = r2_score(Y_test_shuf[:, f], y_pred_diag)
    
    print(f"  -> Finger {finger_names_diag[f]}: R² = {r2_diag:.2f} (RMSE = {rmse_diag:.2f})")

print(">>> END OF DIAGNOSTIC TEST")
print("="*50 + "\n")
# ==============================================================================


scaler = StandardScaler()
X_train_z = scaler.fit_transform(EMG_train_extracted_features)
X_val_z   = scaler.transform(EMG_val_extracted_features)
X_test_z  = scaler.transform(EMG_test_extracted_features)

corr = pd.DataFrame(X_train_z).corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr, cmap="coolwarm", center=0, square=True)
plt.title("Feature correlation (z-scored, train set)")
plt.show()

# ------------------------------------------------------------------------------
# 4. Gradient boosting regression ----------------
print (">>> STARTING REGRESSION")
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Train one regressor per finger angle
n_fingers = Labels_train_mean.shape[1]

models = []
Y_test_pred = np.zeros_like(Labels_test_mean)

for f in range(n_fingers):
    gbr = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=0
    )
    gbr.fit(X_train_z, Labels_train_mean[:, f])
    Y_test_pred[:, f] = gbr.predict(X_test_z) # Prediction

    # Smoothing process --------------------------
    window_size = 15 
    weights = np.ones(window_size) / window_size
    pred_smooth = np.convolve(Y_test_pred[:, f], weights, mode='same')
    
    Y_test_pred[:, f] = pred_smooth # Replace old value
    # --------------------------------------------

    models.append(gbr)

# 4. Visualization -------------------------------
finger_names = ["J3", "J6", "J8", "J11", "J14"]

plt.figure(figsize=(14, 8))
for f in range(n_fingers):
    plt.subplot(3, 2, f + 1)
    plt.plot(Labels_test_mean[:, f], label="True", alpha=0.8)
    plt.plot(Y_test_pred[:, f], label="Predicted", alpha=0.8)
    plt.title(f"Finger angle {finger_names[f]}")
    plt.xlabel("Window index")
    plt.ylabel("Angle")
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()
print(">>> DONE")
print("="*50 + "\n")

# 5. Performance metric --------------------------
print (">>> STARTING PERFORMANCE METRIC")
rmse_per_finger = []
r2_per_finger = []

for f in range(n_fingers):
    rmse = np.sqrt(mean_squared_error(
        Labels_test_mean[:, f],
        Y_test_pred[:, f]
    ))
    r2 = r2_score(
        Labels_test_mean[:, f],
        Y_test_pred[:, f]
    )

    rmse_per_finger.append(rmse)
    r2_per_finger.append(r2)

    print(f"  -> Finger {finger_names[f]}: R² = {r2:.2f} (RMSE = {rmse:.2f})")
print("="*50 + "\n")

# 6. Stability -----------------------------------
print (">>> STARTING STABILITY")
plt.figure(figsize=(8, 4))
plt.bar(finger_names, rmse_per_finger)
plt.ylabel("RMSE (angle units)")
plt.title("Regression error across finger joints")
plt.grid(True)
plt.show()

