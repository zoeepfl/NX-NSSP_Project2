import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

from scipy.io import loadmat 
from scipy.ndimage import convolve1d
from scipy.signal import butter
from scipy.signal import sosfiltfilt
from scipy.signal import welch
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns

#--------------------------------------------------------------------------
# Load data
S2_A1_E1 = sio.loadmat('s2/S2_A1_E1.mat')
print(S2_A1_E1.keys())

emg = S2_A1_E1['emg']
print(emg.shape)
fs = 2000
n_samples, n_channels = emg.shape
time = np.arange(n_samples) / fs

"""
# Remove channel 1 and 2
bad_channels = [0, 1] 
emg_clean = np.delete(emg, bad_channels, axis=1)
n_channels = emg_clean.shape[1] 
print(f"Removed channels {bad_channels}. Remaining channels: {n_channels}")
"""
#----------------------------------------------------------------------------
# PART 1: Preprocessing
# ---------------------

# 1. Filter (Bandpass + Notch)
bandpass_cutoff_frequencies_Hz = (5, 500) 
sos = butter(N=4, Wn=bandpass_cutoff_frequencies_Hz, fs=fs, btype="bandpass", output="sos") 
emg_filtered = sosfiltfilt(sos, emg.T).T 

powergrid_noise_frequencies_Hz = [harmonic_idx*50 for harmonic_idx in range(1,3)] 
for noise_frequency in powergrid_noise_frequencies_Hz:
    sos = butter(N=4, Wn=(noise_frequency - 2, noise_frequency + 2), fs=fs, btype="bandstop", output="sos")
    emg_filtered = sosfiltfilt(sos, emg_filtered.T).T
    
# 2. Rectify
emg_rectified = np.abs(emg_filtered)

# 3. Envelope
mov_mean_size = 400
mov_mean_weights = np.ones(mov_mean_size) / mov_mean_size
emg_continuous_env = convolve1d(emg_rectified, weights=mov_mean_weights, axis=0)

# Plot
n_channels_to_plot = 10
fig, axes = plt.subplots(n_channels_to_plot, 1, figsize=(10, 8), sharex=True)

for i in range(n_channels_to_plot):
    # Plotting the whole file (or use zoom_slice)
    axes[i].plot(emg_continuous_env[:, i], color='tab:blue')
    axes[i].set_ylabel(f'Ch {i+1}')
    axes[i].grid(True)

axes[-1].set_xlabel('Time (samples)')
plt.suptitle('Continuous Processed Envelopes (First 5 Channels)')
plt.tight_layout()
plt.show()

# PART 2: Segmentation
# --------------------

stimulus = S2_A1_E1['restimulus']
repetition = S2_A1_E1['rerepetition']

n_stimuli = len(np.unique(stimulus)) - 1 
n_repetitions = len(np.unique(repetition)) - 1 

# Initializing the data structure
emg_windows = [[None for repetition_idx in range(n_repetitions)] for stimuli_idx in range(n_stimuli)] 
emg_envelopes = [[None for repetition_idx in range(n_repetitions)] for stimuli_idx in range(n_stimuli)]

for stimuli_idx in range(n_stimuli):
    for repetition_idx in range(n_repetitions):
        idx = np.logical_and(stimulus == stimuli_idx + 1, repetition == repetition_idx + 1).flatten()
        emg_windows[stimuli_idx][repetition_idx] = emg[idx, :]
        emg_envelopes[stimuli_idx][repetition_idx] = emg_continuous_env[idx, :]
        
#----------------------------------------------------------------------------------------

def build_dataset_from_ninapro(emg, stimulus, repetition, features=None):
    # Calculate the number of unique stimuli and repetitions, subtracting 1 to exclude the resting condition
    n_stimuli = np.unique(stimulus).size - 1
    n_repetitions = np.unique(repetition).size - 1
    # Total number of samples is the product of stimuli and repetitions
    n_samples = n_stimuli * n_repetitions
    
    # Number of channels in the EMG data
    n_channels = emg.shape[1]
    # Calculate the total number of features by summing the number of channels for each feature
    n_features = sum(n_channels for feature in features)
    
    # Initialize the dataset and labels arrays with zeros
    dataset = np.zeros((n_samples, n_features))
    labels = np.zeros(n_samples)
    current_sample_index = 0
    
    # Loop over each stimulus and repetition to extract features
    for i in range(n_stimuli):
        for j in range(n_repetitions):
            # Assign the label for the current sample
            labels[current_sample_index] = i + 1
            # Calculate the current sample index based on stimulus and repetition
            current_sample_index = i * n_repetitions + j
            current_feature_index = 0
            # Select the time steps corresponding to the current stimulus and repetition
            selected_tsteps = np.logical_and(stimulus == i + 1, repetition == j + 1).squeeze()
            
            # Loop over each feature function provided
            for feature in features:
                # Determine the indices in the dataset where the current feature will be stored
                selected_features = np.arange(current_feature_index, current_feature_index + n_channels)
                # Apply the feature function to the selected EMG data and store the result
                dataset[current_sample_index, selected_features] = feature(emg[selected_tsteps, :])
                # Update the feature index for the next feature
                current_feature_index += n_channels

            # Move to the next sample
            current_sample_index += 1
            
    # Return the constructed dataset and corresponding labels
    return dataset, labels

#----------------------------------------------------------------------------------------------

def get_ssc(x, threshold=0):
    diff = np.diff(x, axis=0) # shape becomes (N-1, Channels)
    consecutive_prod = diff[:-1] * diff[1:]
    return np.sum(consecutive_prod < 0, axis=0)
    
# Define the features 
emg_features = [
    lambda x: np.mean(x, axis=0),                                        # Mean Absolute Value (MAV)
    lambda x: np.std(x, axis=0),                                         # Standard Deviation
    lambda x: np.max(x, axis=0),                                         # Peak Amplitude
    lambda x: np.sqrt(np.mean(x**2, axis=0)),                            # Root Mean Square (RMS)
    lambda x: np.sum(np.abs(np.diff(x, axis=0)), axis=0),                # Waveform length (WL)
    lambda x: get_ssc(x, threshold=0)                                    # Slope sign changes (SSC)

]
#Feel free to add more features, e.g. frequency domain features from the two papers 

dataset, labels = build_dataset_from_ninapro(
    emg=emg,
    stimulus=stimulus,
    repetition=repetition,
    features=emg_features
    )

print(f"dataset dimension: {dataset.shape}")
print(f"labels dimension: {labels.shape}")

#----------------------------------------------------------------------------------------------------

# Split the dataset into training and testing sets
# Here, 30% of the data is reserved for testing, and 70% is used for training
X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.33)