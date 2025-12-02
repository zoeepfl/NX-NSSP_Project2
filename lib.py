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

def plot_channels(n_channels, emg_continuous_env):
    n_channels_to_plot = n_channels
    fig, axes = plt.subplots(n_channels_to_plot, 1, figsize=(10, 8), sharex=True)

    for i in range(n_channels_to_plot):
        # Plotting the whole file (or use zoom_slice)
        axes[i].plot(emg_continuous_env[:, i], color='tab:blue')
        axes[i].set_ylabel(f'Ch {i+1}')
        axes[i].grid(True)

    axes[-1].set_xlabel('Time (samples)')
    plt.suptitle('Continuous Processed Envelopes')
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)

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

def plot_conf_mat(y_test, y_pred, y_pred_opt, y_pred_kbest, y_pred_pca):
    # Plot confusion matrices
    predictions = [y_pred, y_pred_opt, y_pred_kbest, y_pred_pca]
    titles = [
        "GB (no hyperparam opt)",
        "GB (with hyperparam opt)",
        "GB with SelectKBest",
        "GB with PCA"
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    for i, ax in enumerate(axes.flat):
        confmat = confusion_matrix(y_test, predictions[i], normalize="true")
        sns.heatmap(confmat, annot=True, ax=ax)
        ax.set_title(titles[i])
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")

    plt.tight_layout()
    plt.show()
    # plt.show(block=False)
    # plt.pause(0.5)
    return