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
from matplotlib.gridspec import GridSpec 


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


# =================================================================
# Segmentation

def create_emg_windows(stimulus,repetition,emg):
    n_stimuli = len(np.unique(stimulus)) - 1 
    n_repetitions = len(np.unique(repetition)) - 1 
    # Initialisation
    emg_windows = [[None for r in range(n_repetitions)] for s in range(n_stimuli)] 

    for s in range(n_stimuli):
        for r in range(n_repetitions):
            idx = np.logical_and(stimulus == s + 1, repetition == r + 1).flatten()
            emg_windows[s][r] = emg[idx, :]

    return emg_windows


# =================================================================
# Features
def build_dataset_with_features(emg_windows):

    emg_features = [
        lambda x: np.mean(x, axis=0),             # MAV 
        lambda x: np.max(x, axis=0),              # Peak Amplitude
        lambda x: np.std(x, axis=0),              # Standard Deviation
        lambda x: np.sqrt(np.mean(x**2, axis=0)), # RMS
        lambda x: np.sum(np.abs(np.diff(x, axis=0)), axis=0) # Waveform Length
    ]

    # Building dataset
    dataset, labels = build_dataset(
        emg_windows=emg_windows,
        features=emg_features
        )
    return dataset, labels

def plot_consistency_mav_wl(dataset, labels, feature_indices=[0, 4], n_channels=10, channel_to_plot=0):
    """
    Plots the consistency of repetitions for MAV and another feature (e.g., WL).
    Superimposes all movements on the same plot (X=Repetition, Y=Value).
    
    Parameters:
    - dataset: The features matrix (n_samples, n_features_total)
    - labels: The movement labels
    - feature_indices: List of 2 integers. 
                       Default [0, 4] assumes: 0=MAV, 4=WL (based on your list order).
    """
    
    # 1. Feature Names (for the plot title)
    feat_names = {0: 'MAV', 1: 'Max', 2: 'Std', 3: 'RMS', 4: 'Waveform Length (WL)'}
    
    f1_idx = feature_indices[0]
    f2_idx = feature_indices[1]
    
    name1 = feat_names.get(f1_idx, f'Feature {f1_idx}')
    name2 = feat_names.get(f2_idx, f'Feature {f2_idx}')
    
    # 2. Reorganize Data for Plotting
    # We need to extract the specific columns for the chosen channel
    # Column Index = (Feature_Index * n_channels) + Channel_Index
    
    data_rows = []
    n_samples = dataset.shape[0]
    
    # Assuming standard order: [Feat1_Ch1, Feat1_Ch2... Feat2_Ch1...]
    col_idx_1 = (f1_idx * n_channels) + channel_to_plot
    col_idx_2 = (f2_idx * n_channels) + channel_to_plot
    
    for i in range(n_samples):
        # Determine Repetition ID (Assuming data is ordered: 10 reps per mvt)
        # Or calculated from index if labels are sorted
        # Ideally, we should pass the 'repetition' array, but here we estimate:
        rep_id = (i % 10) + 1 
        
        mvt_label = labels[i]
        
        data_rows.append({
            'Movement': str(int(mvt_label)), # String for categorical coloring
            'Repetition': rep_id,
            name1: dataset[i, col_idx_1],
            name2: dataset[i, col_idx_2]
        })
        
    df = pd.DataFrame(data_rows)
    
    # 3. Create the Plots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # --- Plot 1: MAV Consistency ---
    sns.lineplot(data=df, x='Repetition', y=name1, hue='Movement', 
                 marker='o', palette='tab20', ax=axes[0])
    axes[0].set_title(f'Consistency of {name1} (Channel {channel_to_plot+1})')
    axes[0].set_ylabel(f'{name1} Value')
    axes[0].set_xlabel('Repetition (1-10)')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Movement')

    # --- Plot 2: WL Consistency ---
    sns.lineplot(data=df, x='Repetition', y=name2, hue='Movement', 
                 marker='o', palette='tab20', ax=axes[1])
    axes[1].set_title(f'Consistency of {name2} (Channel {channel_to_plot+1})')
    axes[1].set_ylabel(f'{name2} Value')
    axes[1].set_xlabel('Repetition (1-10)')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend().remove()
    
    plt.tight_layout()
    plt.show()

# =================================================================

# 4) Classification    
def classify_without_hyperparameter_optimization(X_train_z, X_test_z, y_train, y_test):
    # Train a classifier on the normalized data
    clf = GradientBoostingClassifier()
    clf.fit(X_train_z, y_train)  # Fit the model on the training data
    # Predict the labels for the test set
    y_pred = clf.predict(X_test_z)
    # Performance metrics
    print_metrics(y_test, y_pred, "GB without hyperparameter optimization")
    return y_pred

#----------------------------------
# Hyperparameter optimization
def classify_with_hyperparameter_optimization(X_train_z, X_test_z, y_train, y_test,param_grid):
   
    grid = GridSearchCV(GradientBoostingClassifier(), param_grid, cv=3, n_jobs=-1) #3 folds cross validation
    grid.fit(X_train_z, y_train)

    # print(f"Best estimator: {grid.best_estimator_}")
    # print(f"Best hyperparameters: {grid.best_params_}")

    y_pred_HO = grid.predict(X_test_z)

    # Performance metrics
    print_metrics(y_test, y_pred_HO, "GB with hyperparameter optimization")
    return y_pred_HO

def Kbest_feature_selection(X_train_z, X_test_z, y_train, y_test,param_grid):
    # Select the top 10 features based on mutual information scores.
    # Note: You can change 'k' to 30 if you are working with more features.
    k_best = SelectKBest(mutual_info_classif, k=10)
    k_best.fit(X_train_z, y_train)

    # Transform the training and test datasets to only include the selected features.
    X_train_best = k_best.transform(X_train_z)
    X_test_best = k_best.transform(X_test_z)

    clf_kbest = GradientBoostingClassifier()
    clf_kbest.fit(X_train_best, y_train)

    # Predict the labels for the test set using the trained model.
    y_pred_kbest = clf_kbest.predict(X_test_best)

    # Performance metrics
    print_metrics(y_test, y_pred_kbest, "GB with kbest")

    #------
    # Select k-best with hyperparameter optimization

    grid_kbest = GridSearchCV(
        GradientBoostingClassifier(),
        param_grid,
        cv=3,
        n_jobs=-1
    )
    grid_kbest.fit(X_train_best, y_train)
    y_pred_kbest_HO = grid_kbest.predict(X_test_best)
    print_metrics(y_test, y_pred_kbest_HO, "GB with kbest with hyperparameter optimization")
    return y_pred_kbest, y_pred_kbest_HO


def PCA_feature_reduction(X_train_z, X_test_z, y_train, y_test,param_grid):
    pca = PCA(n_components=10) #10 components : chosen with the elbow method
    # pca.fit(X_train_z, y_train)
    pca.fit(X_train_z)

    X_train_pca = pca.transform(X_train_z)
    X_test_pca = pca.transform(X_test_z)

    # Train Gradient Boosting on PCA-reduced features
    clf_pca = GradientBoostingClassifier()
    clf_pca.fit(X_train_pca, y_train)

    y_pred_pca = clf_pca.predict(X_test_pca)

    # Performance metrics
    print_metrics(y_test, y_pred_pca, "GB with PCA")

    #--------
    # PCA with hyperparameter optimization
    grid_pca = GridSearchCV(
        GradientBoostingClassifier(),
        param_grid,
        cv=3,
        n_jobs=-1
    )
    grid_pca.fit(X_train_pca, y_train)
    y_pred_pca_HO = grid_pca.predict(X_test_pca)
    print_metrics(y_test, y_pred_pca_HO, "GB with PCA with hyperparameter optimization")
    return y_pred_pca, y_pred_pca_HO

def Mean_all_features_one_chanel(datasets_all, n_channels=10, n_features=5, feature_names=None):
   
    n_subjects = len(datasets_all)

    # Matrice features × subjects
    mean_matrix = np.zeros((n_features, n_subjects))

    for f in range(n_features):
        for s, dataset in enumerate(datasets_all):
            start = f * n_channels
            ch1_index = start + 0         
            mean_matrix[f, s] = np.mean(dataset[:, ch1_index])
    return mean_matrix


def plot_heatmaps_all_features(mean_matrix, n_channels=10, n_features=5, feature_names=None):
    if feature_names is None:
        feature_names = ["MAV", "STD", "MAX", "RMS", "WL"]

    fig = plt.figure(figsize=(14, 2.3 * n_features))
    gs = GridSpec(n_features, 1, figure=fig, hspace=0.55)   
    n_subjects = mean_matrix.shape[1]
    axes = []
    for f in range(n_features):
        ax = fig.add_subplot(gs[f, 0])
        axes.append(ax)
        
        heat_data = mean_matrix[f][np.newaxis, :]          
        im = ax.imshow(heat_data, aspect='auto', cmap='viridis')

        ax.set_yticks([0])
        ax.set_yticklabels([feature_names[f]])

        ax.set_title(f"Feature : {feature_names[f]} (Canal 1)", pad=12, fontsize=10)

        if f < n_features - 1:
            ax.set_xticks([])

    axes[-1].set_xticks(range(n_subjects))
    axes[-1].set_xticklabels([str(i+1) for i in range(n_subjects)], rotation=45)
    axes[-1].set_xlabel("Subject")

    cbar = fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.03, pad=0.02)
    cbar.set_label("Mean value", rotation=90)

    plt.show()


def plot_cv_scores(accuracy_scores, f1_scores):
    subjects = np.arange(1, len(accuracy_scores) + 1)

    plt.figure(figsize=(12, 8))

    # --- Plot Accuracy ---
    plt.subplot(2, 1, 1)
    plt.plot(subjects, accuracy_scores, marker='o')
    plt.title("Accuracy ")
    plt.xlabel("Nb of suject for trainning")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.xticks(subjects)

    # --- Plot F1 Score ---
    plt.subplot(2, 1, 2)
    plt.plot(subjects, f1_scores, marker='o')
    plt.title("F1 Score")
    plt.xlabel("Nb of suject for trainning")
    plt.ylabel("F1 Score")
    plt.grid(True)
    plt.xticks(subjects)

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