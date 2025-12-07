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

def get_ssc(x, threshold=0): #slope sign changes
    diff = np.diff(x, axis=0) # shape becomes (N-1, Channels)
    consecutive_prod = diff[:-1] * diff[1:]
    return np.sum(consecutive_prod < 0, axis=0)
    
def plot_global_psd_check(data, fs, nperseg=1024):
    """
    Plots the Power Spectral Density (PSD) of all channels superimposed on one figure.
    Useful for spotting dead channels, line noise, and artifacts globally.

    Parameters:
    - data: numpy array (n_samples, n_channels) - usually the raw EMG data
    - fs: int, sampling frequency
    - nperseg: int, length of each segment for Welch's method (default 1024)
    """
    n_channels = data.shape[1]
    
    plt.figure(figsize=(10, 6))
    
    # Generate a unique color for each channel
    colors = plt.cm.jet(np.linspace(0, 1, n_channels))

    for i in range(n_channels):
        freqs, psd = welch(data[:, i], fs=fs, nperseg=nperseg)
        plt.semilogy(freqs, psd, label=f'Ch {i+1}', color=colors[i], alpha=0.7)

    # Visual Guides
    plt.axvline(50, color='r', linestyle='--', alpha=0.5, label='Mains (50Hz)')
    plt.axvspan(0, 20, color='orange', alpha=0.2, label='Motion Artifact Zone')

    # Formatting
    plt.title('Global PSD Check (All Channels)')
    plt.ylabel('Power Spectral Density (V**2/Hz)')
    plt.xlabel('Frequency (Hz)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, which='both', alpha=0.3)
    plt.xlim(0, 400)
    plt.tight_layout()
    plt.show()

def plot_time_domain_check(raw_data, filtered_data, envelope_data, stimulus_arr, time_arr, target_stim, ch_idx):
    """
    Plots Raw vs Filtered vs Envelope signals for a specific stimulus block.
    
    Parameters:
    - raw_data: numpy array (n_samples, n_channels)
    - filtered_data: numpy array (n_samples, n_channels)
    - envelope_data: numpy array (n_samples, n_channels)
    - stimulus_arr: numpy array (n_samples, 1) or (n_samples,) containing stimulus IDs
    - time_arr: numpy array (n_samples,) time vector
    - target_stim: int, the stimulus ID to zoom in on (e.g., 1)
    - ch_idx: int, channel index to plot (0-indexed)
    """
    # Find start and end indices for this stimulus block
    stim_indices = np.where(stimulus_arr == target_stim)[0]

    if len(stim_indices) > 0:
        start_idx = stim_indices[0] - 500
        end_idx = stim_indices[-1] + 500
        
        # Safe bounds check
        start_idx = max(0, start_idx)
        end_idx = min(raw_data.shape[0], end_idx)
        
        t_slice = time_arr[start_idx:end_idx]
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        
        # 1. Raw
        axes[0].plot(t_slice, raw_data[start_idx:end_idx, ch_idx], color='silver', label='Raw')
        axes[0].set_title(f'Raw Signal (Ch {ch_idx+1}, Stimulus {target_stim})')
        axes[0].legend(loc='upper right')
        axes[0].grid(True, alpha=0.3)
        
        # 2. Filtered
        axes[1].plot(t_slice, filtered_data[start_idx:end_idx, ch_idx], color='tab:blue', label='Filtered')
        axes[1].set_title('Bandpass + Notch Filtered')
        axes[1].legend(loc='upper right')
        axes[1].grid(True, alpha=0.3)
        
        # 3. Envelope
        axes[2].plot(t_slice, envelope_data[start_idx:end_idx, ch_idx], color='tab:red', linewidth=2, label='Envelope')
        axes[2].set_title('Envelope')
        axes[2].set_xlabel('Time (s)')
        axes[2].legend(loc='upper right')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    else:
        print(f"Stimulus {target_stim} not found in data.")

def plot_spectral_check(raw_data, filtered_data, ch_idx, fs, nperseg=1024):
    """
    Computes and plots the PSD comparison between Raw and Filtered signals.
    
    Parameters:
    - raw_data: numpy array (n_samples, n_channels)
    - filtered_data: numpy array (n_samples, n_channels)
    - ch_idx: int, channel index to plot (0-indexed)
    - fs: int, sampling frequency
    """
    # Calculate PSD
    freqs, psd_raw = welch(raw_data[:, ch_idx], fs=fs, nperseg=nperseg)
    freqs, psd_clean = welch(filtered_data[:, ch_idx], fs=fs, nperseg=nperseg)

    # Plot
    plt.figure(figsize=(10, 5))
    plt.semilogy(freqs, psd_raw, label='Raw Signal', alpha=0.5)
    plt.semilogy(freqs, psd_clean, label='Filtered Signal', color='black')

    # Visual Guides
    plt.axvline(50, color='r', linestyle='--', alpha=0.5, label='Mains (50Hz)')
    plt.axvspan(0, 20, color='y', alpha=0.2, label='Motion Artifact Zone')

    # Formatting
    plt.title(f'Spectral Check: Channel {ch_idx+1}')
    plt.ylabel('Power Spectral Density (V**2/Hz)')
    plt.xlabel('Frequency (Hz)')
    plt.legend()
    plt.grid(True, which='both', alpha=0.3)
    plt.xlim(0, 400)
    plt.tight_layout()
    plt.show()

def plot_pca(X_train_z) :
    # Fit PCA with ALL components
    pca_full = PCA().fit(X_train_z)

    # Explained variance
    expl_var = pca_full.explained_variance_ratio_
    cum_var = np.cumsum(expl_var)

    plt.figure(figsize=(8,5))
    plt.plot(cum_var, marker='o')
    plt.xlabel("Number of components")
    plt.ylabel("Cumulative explained variance")
    plt.title("PCA : choose nb of components at the elbow")
    plt.grid(True)
    plt.show(block=False)
    plt.pause(0.1)

def print_metrics(y_test, y_pred, label):
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy score : {accuracy:.3f} ({label})")
    f1 = f1_score(y_test, y_pred, average="macro")
    print(f"Macro F1 : {f1:.3f} ({label})")

def plot_conf_mat(y_test, y_pred, y_pred_kbest, y_pred_pca, y_pred_HO, y_pred_kbest_HO, y_pred_pca_HO):

    predictions = [y_pred, y_pred_kbest, y_pred_pca, y_pred_HO, y_pred_kbest_HO, y_pred_pca_HO]
    titles = [
        "GB (gradient boosting)",
        "GB with SelectKBest",
        "GB with PCA",
        "GB with HO (hyperparam opt)",
        "GB with SelectKBest with HO",
        "GB with PCA with HO"
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for i, ax in enumerate(axes.flat):
        confmat = confusion_matrix(y_test, predictions[i], normalize="true")
        sns.heatmap(confmat,
                    annot=True,
                    fmt=".2f",
                    ax=ax,
                    cbar=True,
                    annot_kws={"size": 6})

        ax.set_title(titles[i], fontsize=12)
        ax.set_xlabel("Predicted label", fontsize=10)
        ax.set_ylabel("True label", fontsize=10)
        ax.tick_params(axis='both', labelsize=8)

    plt.subplots_adjust(hspace=0.35, wspace=0.25)
    plt.show()
    # plt.show(block=False)
    # plt.pause(0.5)
    return
