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

import numpy as np

def build_dataset(emg_windows, features):
    dataset = []
    labels = []
    n_stimuli = len(emg_windows)

    if n_stimuli > 0:
        n_repetitions = len(emg_windows[0])
    else:
        n_repetitions = 0    
   
    for s in range(n_stimuli):
        for r in range(n_repetitions):
            signal = emg_windows[s][r]
            
            if signal is not None and signal.size > 0:
                trial_features = []                
                
                for func in features:
                    val = func(signal) 
                    trial_features.extend(val)
                
                dataset.append(trial_features)
                labels.append(s + 1) # Label (1 à 12)

    return np.array(dataset), np.array(labels)

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
    plt.xlim(0, 100)
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


def plot_emg_before_after(emg, fs, emg_filtered, channels=None):
    """
    Plot EMG avant et après filtrage pour plusieurs canaux.

    Parameters
    ----------
    emg : ndarray (n_samples, n_channels)
        Signal EMG brut
    fs : float
        Fréquence d'échantillonnage (Hz)
    channels : list or None
        Liste des canaux à afficher (ex: [0,1,2]).
        Si None, affiche tous les canaux.
    """

    n_samples, n_channels = emg.shape

    if channels is None:
        channels = list(range(n_channels))

    # Filtrage
    #emg_filtered = bandpass_filter_emg(emg, fs)
    #emg_filtered = emg_highpass_filter(emg,fs)
    # Axe temporel
    t = np.arange(n_samples) / fs

    plt.figure(figsize=(12, 3 * len(channels)))

    for i, ch in enumerate(channels):
        plt.subplot(len(channels), 1, i + 1)

        plt.plot(t, emg[:, ch], label="Brut", alpha=0.6)
        plt.plot(t, emg_filtered[:, ch], label="Filtré", linewidth=1.5)

        plt.ylabel(f"Canal {ch}")
        plt.grid(True)

        if i == 0:
            plt.title("EMG avant et après filtrage")

        if i == len(channels) - 1:
            plt.xlabel("Temps (s)")

        plt.legend(loc="upper right")

    plt.tight_layout()
    plt.show()

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