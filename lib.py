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

def plot_movements(emg_data, stimulus_arr, time_arr):
    """
    Plots raw EMG data in two figures (2x3 subplots each) for the 12 movements.
    - Figure 1: Movements 1-6
    - Figure 2: Movements 7-12
    """
    n_samples, n_channels = emg_data.shape
    colors = plt.cm.jet(np.linspace(0, 1, n_channels))

    # --- Helper function to plot a single movement on an ax ---
    def plot_on_ax(ax, stim_id):
        # Find indices for this movement
        stim_indices = np.where(stimulus_arr == stim_id)[0]
        
        if len(stim_indices) > 0:
            # Add padding
            start_idx = max(0, stim_indices[0] - 200)
            end_idx = min(n_samples, stim_indices[-1] + 200)
            
            t_slice = time_arr[start_idx:end_idx]
            data_slice = emg_data[start_idx:end_idx, :]
            
            # Plot all channels superimposed
            for ch in range(n_channels):
                ax.plot(t_slice, data_slice[:, ch], 
                        color=colors[ch], 
                        alpha=0.6, 
                        linewidth=0.8)
            
            ax.set_title(f'Movement {stim_id}')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center')
            ax.set_title(f'Movement {stim_id} (Missing)')

    # --- Setup Legend Lines (Used for both figures) ---
    legend_lines = [plt.Line2D([0], [0], color=colors[i], lw=2) for i in range(n_channels)]
    legend_labels = [f'Ch {i+1}' for i in range(n_channels)]

    # FIGURE 1: Movements 1 to 6
    fig1, axes1 = plt.subplots(2, 3, figsize=(15, 7), sharey=True)
    fig1.suptitle('Raw Data: Movements 1 - 6', fontsize=16)
    axes1_flat = axes1.flatten()

    for i, stim_id in enumerate(range(1, 7)): # 1 to 6
        plot_on_ax(axes1_flat[i], stim_id)
        
        # Only set labels on outer edges to keep it clean
        if i >= 3: axes1_flat[i].set_xlabel('Time (s)')
        if i % 3 == 0: axes1_flat[i].set_ylabel('Amplitude (mV)')
        
    fig1.legend(legend_lines, legend_labels, 
                loc='lower center', ncol=n_channels, 
                bbox_to_anchor=(0.5, 0.0))
    plt.tight_layout(rect=[0, 0.05, 1, 1]) # Adjust layout for legend space
    plt.show()

    # FIGURE 2: Movements 7 to 12
    fig2, axes2 = plt.subplots(2, 3, figsize=(15, 7), sharey=True)
    fig2.suptitle('Raw Data: Movements 7 - 12', fontsize=16)
    axes2_flat = axes2.flatten()

    for i, stim_id in enumerate(range(7, 13)): # 7 to 12
        plot_on_ax(axes2_flat[i], stim_id)
        
        if i >= 3: axes2_flat[i].set_xlabel('Time (s)')
        if i % 3 == 0: axes2_flat[i].set_ylabel('Amplitude (mV)')

    fig2.legend(legend_lines, legend_labels, 
                loc='lower center', ncol=n_channels, 
                bbox_to_anchor=(0.5, 0.0))
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.show()

def plot_rejection_results(emg_original, emg_cleaned, fs, trim_s=0.2):
    """
    Visualizes kept vs. rejected trials based on RMS amplitude.
    
    Parameters:
    - emg_original: The raw/enveloped data before cleaning (emg_envelopes)
    - emg_cleaned: The data after cleaning (emg_envelopes_cleaned)
    - fs: Sampling frequency
    - trim_s: Time in seconds to ignore at start of trial for RMS calc (default 0.2s)
    """
    plt.figure(figsize=(12, 6))
    
    n_stimuli = len(emg_original)
    n_repetitions = len(emg_original[0])
    trim_samples = int(trim_s * fs)

    # We use these flags to ensure the legend only appears once, not 100 times
    label_rejected_done = False
    label_kept_done = False

    for s in range(n_stimuli):
        for r in range(n_repetitions):
            
            # 1. Plot Original (Grey X)
            # This represents the "Potential" data
            original = emg_original[s][r]
            if original is not None and original.shape[0] > trim_samples:
                val = np.sqrt(np.mean(original[trim_samples:]**2))
                
                # Logic to handle legend labels cleanly
                lbl = 'Rejected' if not label_rejected_done else ""
                plt.scatter(s+1, val, color='lightgray', marker='x', s=50, label=lbl)
                if not label_rejected_done: label_rejected_done = True

            # 2. Plot Kept (Blue Dot)
            # This represents the data that passed your filters
            kept = emg_cleaned[s][r]
            if kept is not None:
                # Note: kept data is usually already trimmed by your cleaning function
                val = np.sqrt(np.mean(kept**2))
                
                lbl = 'Kept' if not label_kept_done else ""
                plt.scatter(s+1, val, color='tab:blue', s=50, label=lbl)
                if not label_kept_done: label_kept_done = True

    plt.title('Trial Rejection Results (Blue = Kept, Grey X = Rejected)')
    plt.xlabel('Movement ID')
    plt.ylabel('Trial RMS Amplitude')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

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


def clean_and_reject_trials_hard_limit(emg_data_structure, fs, trim_ms=0, threshold=2.5, absolute_max=None):
    """
    Combines Robust Statistics (Median/MAD) with a HARD Upper Limit.
    
    Parameters:
    - absolute_max: Any trial with RMS > this value is AUTOMATICALLY rejected. 
                    (e.g., set to 0.006 based on your plot).
    """
    n_stimuli = len(emg_data_structure)
    n_repetitions = len(emg_data_structure[0])
    
    cleaned_structure = [[None for _ in range(n_repetitions)] for _ in range(n_stimuli)]
    trim_samples = int((trim_ms / 1000) * fs)
    
    print(f"--- Cleaning Report (Hard Limit + Robust Stats) ---")

    for s in range(n_stimuli):
        rms_values = []
        valid_indices = []
        
        # 1. Collect RMS
        for r in range(n_repetitions):
            trial = emg_data_structure[s][r]
            if trial is not None and trial.shape[0] > trim_samples:
                cropped_signal = trial[trim_samples:, :]
                rms = np.sqrt(np.mean(cropped_signal**2))
                
                # --- CHECK 1: HARD LIMIT ---
                if absolute_max is not None and rms > absolute_max:
                     print(f"  -> Rejected Mvt {s+1}, Rep {r+1} [HARD LIMIT]: RMS {rms:.4f} > {absolute_max}")
                     cleaned_structure[s][r] = None # Reject immediately
                else:
                    # Keep for statistical checking
                    rms_values.append(rms)
                    valid_indices.append(r)
            else:
                cleaned_structure[s][r] = None

        # 2. Check Statistics (only on trials that passed the Hard Limit)
        if len(rms_values) > 0:
            rms_array = np.array(rms_values)
            median_rms = np.median(rms_array)
            mad_rms = np.median(np.abs(rms_array - median_rms)) * 1.4826
            
            upper_bound = median_rms + (threshold * mad_rms)
            lower_bound = median_rms - (threshold * mad_rms)
            
            # Ensure we don't accidentally reject valid low-power signals if MAD is tiny
            lower_bound = max(0, lower_bound) 

            for r in valid_indices:
                idx = valid_indices.index(r)
                val = rms_values[idx]
                
                # --- CHECK 2: STATISTICAL OUTLIER ---
                if lower_bound <= val <= upper_bound:
                    cleaned_structure[s][r] = emg_data_structure[s][r][trim_samples:, :]
                else:
                    cleaned_structure[s][r] = None
                    reason = "Statistically High" if val > upper_bound else "Statistically Low"
                    print(f"  -> Rejected Mvt {s+1}, Rep {r+1} [{reason}]: RMS {val:.4f} (Bounds: {lower_bound:.4f}-{upper_bound:.4f})")
        else:
            print(f"Movement {s+1}: All trials rejected by Hard Limit or Empty.")

    return cleaned_structure

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

# This function is used to cut the time windows from the raw EMG 
# # It return a lists containing the EMG of each time window. 
# # It also returns the target corresponding to the time of the end of the window 
def extract_time_windows_regression(EMG: np.ndarray, Label: np.ndarray, fs: int, win_len: int, step: int): 
    """This function is defined to perform an overlapping sliding window 
    :param EMG: Numpy array containing the data 
    :param Label: Numpy array containing the targets 
    :param fs: the sampling frequency of the signal 
    :param win_len: The size of the windows (in seconds) 
    :param step: The step size between windows (in seconds) 
    :return: A Numpy array containing the windows 
    :return: A Numpy array containing the targets aligned for each window 
    :note: The lengths of both outputs are the same 
    """ 
    n,m = EMG.shape 
    win_len = int(win_len*fs) 
    start_points = np.arange(0,n-win_len,int(step*fs)) 
    end_points = start_points + win_len 

    EMG_windows = np.zeros((len(start_points),win_len,m)) 
    Labels_window = np.zeros((len(start_points),win_len,Label.shape[1])) 
    for i in range(len(start_points)): 
        EMG_windows[i,:,:] = EMG[start_points[i]:end_points[i],:] 
        Labels_window[i,:,:] = Label[start_points[i]:end_points[i],:] 
    return EMG_windows, Labels_window 

def extract_features(EMG_windows: np.ndarray, Labels_windows: np.ndarray): 
    """ This function is defined to extract the mean and standard deviation of each window 
    :param EMG_windows: A Numpy array containing the windows 
    :return: A Numpy array containing the mean, the standard deviation and the maximum amplitude of each window and the mean of the labels window 
    """ 
    # along axis 1, which is the time axis 
    EMG_mean = np.mean(EMG_windows,axis=1) 
    EMG_std = np.std(EMG_windows,axis=1) 
    EMG_max_amplitude = np.max(EMG_windows, axis=1) 
    Labels_mean = np.mean(Labels_windows,axis=1) 
    # Concatenate the mean and std of each window 
    EMG_extracted_features = np.concatenate((EMG_mean, EMG_std, EMG_max_amplitude), axis=1) 
    return EMG_extracted_features, Labels_mean 