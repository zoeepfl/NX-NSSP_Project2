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
