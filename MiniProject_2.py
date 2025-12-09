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
# Load data
#S2_A1_E1 = sio.loadmat('s2/S2_A1_E1.mat')
# print(S2_A1_E1.keys())

# emg = S2_A1_E1['emg']
# # print(emg.shape)
# fs = 2000
# n_samples, n_channels = emg.shape
# time = np.arange(n_samples) / fs
# #plot_global_psd_check(emg, fs=2000)

# stimulus = S2_A1_E1['restimulus']
# repetition = S2_A1_E1['rerepetition']

#plot_movements(emg, stimulus, time)
# =================================================================
# PART 1: Preprocessing
# =================================================================

# 1. Filter (Bandpass + Notch)

def bandpass_filter_emg(emg, fs, lowcut=20, highcut=500, order=4):
    bandpass_cutoff_frequencies_Hz = (20, 500) 
    sos = butter(N=4, Wn=bandpass_cutoff_frequencies_Hz, fs=fs, btype="bandpass", output="sos") 
    emg_filtered = sosfiltfilt(sos, emg.T).T 
    powergrid_noise_frequencies_Hz = [harmonic_idx*50 for harmonic_idx in range(1,3)] 
    for noise_frequency in powergrid_noise_frequencies_Hz:
        sos = butter(N=4, Wn=(noise_frequency - 2, noise_frequency + 2), fs=fs, btype="bandstop", output="sos")
        emg_filtered = sosfiltfilt(sos, emg_filtered.T).T
    return emg_filtered   


"""    
# 2. Rectify (Absolute value)
emg_rectified = np.abs(emg_filtered)

# 3. Envelope (Low-Pass Filter)
envelope_cutoff_Hz = 6  
sos_env = butter(N=4, Wn=envelope_cutoff_Hz, fs=fs, btype="low", output="sos")
emg_continuous_env = sosfiltfilt(sos_env, emg_rectified.T).T
"""
def preprocess_emg(emg_filtered, fs):
    # 1. Apply TKEO (The "Contrast Booster")
    emg_tkeo = np.copy(emg_filtered)
    emg_tkeo[1:-1] = emg_filtered[1:-1]**2 - emg_filtered[:-2] * emg_filtered[2:]
    # (Note: The first and last samples remain 0 or raw, usually we ignore them)

    # 2. Rectify the TKEO signal
    emg_rectified = np.abs(emg_tkeo)

    # 3. Envelope (Standard Low-Pass)
    # You might want to slightly increase the cutoff to 10Hz to catch the fast peaks
    envelope_cutoff_Hz = 10 
    sos_env = butter(N=4, Wn=envelope_cutoff_Hz, fs=fs, btype="low", output="sos")
    emg_continuous_env = sosfiltfilt(sos_env, emg_rectified.T).T
    return emg_continuous_env,emg_filtered

'''
plot_time_domain_check(
    raw_data=emg, 
    filtered_data=emg_filtered, 
    envelope_data=emg_continuous_env, 
    stimulus_arr=stimulus, 
    time_arr=time, 
    target_stim=2, 
    ch_idx=1
)'''

# =================================================================
# PART 2: Segmentation
# =================================================================
def clean_emg_envelope(emg_filtered, emg_continuous_env,stimulus, repetition):

    n_stimuli = len(np.unique(stimulus)) - 1 
    n_repetitions = len(np.unique(repetition)) - 1 

    # Initializing the data structure
    emg_windows = [[None for repetition_idx in range(n_repetitions)] for stimuli_idx in range(n_stimuli)] 
    emg_envelopes = [[None for repetition_idx in range(n_repetitions)] for stimuli_idx in range(n_stimuli)]

    for stimuli_idx in range(n_stimuli):
        for repetition_idx in range(n_repetitions):
            idx = np.logical_and(stimulus == stimuli_idx + 1, repetition == repetition_idx + 1).flatten()
            emg_windows[stimuli_idx][repetition_idx] = emg_filtered[idx, :]
            emg_envelopes[stimuli_idx][repetition_idx] = emg_continuous_env[idx, :]
            
    ch_idx = 1
    #plot_spectral_check(emg, emg_filtered, ch_idx, fs=2000)

    #----------------------------------------------------------------------------------------

    #emg_envelopes_cleaned = clean_and_reject_trials(emg_envelopes, fs=2000, trim_ms=0, threshold=3)

    emg_envelopes_cleaned = clean_and_reject_trials_hard_limit(
        emg_envelopes, 
        fs=2000, 
        trim_ms=0, 
        threshold=3, 
        absolute_max=0.006
    )
    return emg_envelopes_cleaned

'''
# --- Visualization of Rejection ---
plot_rejection_results(
    emg_original=emg_envelopes, 
    emg_cleaned=emg_envelopes_cleaned, 
    fs=2000, 
    trim_s=0.2
)'''


def build_dataset_with_features(emg,stimulus,repetition):

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
    return dataset, labels

# print(f"dataset dimension: {dataset.shape}")
# print(f"labels dimension: {labels.shape}")

#----------------------------------------------------------------------------------------------------

# Split the dataset into training and testing sets
# Here, 30% of the data is reserved for testing, and 70% is used for training
#X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.33)

# ----------------------------------------------------------------------------------------------------
# 4) Classification

    # Normalizing the data
    # StandardScaler is used to scale the features so that they have a mean of 0 and a standard deviation of 1
    # scaler = StandardScaler()
    # X_train_z = scaler.fit_transform(X_train)  # Fit the scaler on the training data and transform it
    # X_test_z = scaler.transform(X_test)        # Transform the test data using the same scaler
def classify_without_hyperparameter_optimization(X_train_z, X_test_z, y_train, y_test):
    # Train a classifier on the normalized data
    clf = GradientBoostingClassifier()
    clf.fit(X_train_z, y_train)  # Fit the model on the training data

    # Predict the labels for the test set
    y_pred = clf.predict(X_test_z)

    # Performance metrics
    print_metrics(y_test, y_pred, "GB without hyperparameter optimization")

    return y_pred
#--------------------------------
# # Perform cross-validation
# scores = cross_val_score(clf, X_train_z, y_train, cv=3)
# formatted_scores = [f"{s:.3f}" for s in scores]
# print(f"4) Cross validation : Accuracy scores of all models: {formatted_scores} (GB without hyperparameter optimization)")
# print(f"4) Cross validation : Mean accuracy across all models: {np.mean(scores):.3f} (GB without hyperparameter optimization)")

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
#---------------------------------------
# 5) Performance evaluation
# Assignement :  Evaluate the performance using a metric of your choice. 
#                Justify why the metric is suitable for this task and whether the performance is satisfactory.
# Different possible metrics : Accuracy, F1-score, Confusion matrix

# F1 = better than accuracy for imbalanced class distibutions
# But we have exactly 10 repetitions for each movement, so the classes are balanced -> accuracy is sufficient
# Confusion matrix : useful for qualitative analysis, but not really a comparison metric (too many values)
#----------------------------------------
# 6) Feature selection / dimension reduction using 2 methods : SelectKBest, PCA

#----------------------
# 6)a) First method : SelectKBest

# Calculate mutual information between each feature and the target variable.
# Mutual information is a measure of the dependency between variables.
# A higher value indicates a stronger relationship.
# mutual_info = mutual_info_classif(X_train_z, y_train)
# print(f"Estimated mutual information between each feature and the target:\n {mutual_info}\n")

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


#----------------------------
# 6)b) Second method : PCA

# plot_pca(X_train_z) # plot to decide number of PCA components
# result : we chose 10 components by lookint at the elbow of the curve

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


########################
# Confusion matrices
#plot_conf_mat(y_test, y_pred, y_pred_kbest, y_pred_pca,  y_pred_HO, y_pred_kbest_HO, y_pred_pca_HO)


def main():
    ####PART 1: Single subject Classification####
    # Load data
    S2_A1_E1 = sio.loadmat('dataset/s2/S2_A1_E1.mat')
    stimulus = S2_A1_E1['restimulus']
    repetition = S2_A1_E1['rerepetition']
    emg_raw = S2_A1_E1['emg']
    fs = 2000

    #1) Preprocessing
    emg_filtered = bandpass_filter_emg(emg_raw, fs)
    emg_continusous_env, emg_filtered = preprocess_emg(emg_filtered, fs)
    emg_envelopes_cleaned = clean_emg_envelope(emg_filtered, emg_continusous_env,stimulus,repetition)



    #3) Dataset building with feature extraction
    dataset, labels = build_dataset_with_features(emg_raw,stimulus,repetition)

    # 2) Split the dataset into training and testing sets
    # Here, 30% of the data is reserved for testing, and 70% is used for training
    X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.33)

    #4) Classification
    # Normalizing the data
    # StandardScaler is used to scale the features so that they have a mean of 0 and a standard deviation of 1
    scaler = StandardScaler()
    X_train_z = scaler.fit_transform(X_train)  # Fit the scaler on the training data and transform it
    X_test_z = scaler.transform(X_test)        # Transform the test data using the same scaler

    y_pred=classify_without_hyperparameter_optimization(X_train_z, X_test_z, y_train, y_test)

    param_grid = {
        "n_estimators": [50, 100, 150], #default : 100
        "learning_rate": [0.01, 0.05, 0.1], #default : 0.1
        "max_depth": [2, 3, 4], #default : 3
    }
    y_pred_HO=classify_with_hyperparameter_optimization(X_train_z, X_test_z, y_train, y_test,param_grid)
    
    # 5) Performance evaluation
    # Assignement :  Evaluate the performance using a metric of your choice. 
    #                Justify why the metric is suitable for this task and whether the performance is satisfactory.
    # Different possible metrics : Accuracy, F1-score, Confusion matrix

    # F1 = better than accuracy for imbalanced class distibutions
    # But we have exactly 10 repetitions for each movement, so the classes are balanced -> accuracy is sufficient
    # Confusion matrix : useful for qualitative analysis, but not really a comparison metric (too many values)

    # 6) Feature selection / dimension reduction using 2 methods : SelectKBest, PCA
    # 6)a) First method : SelectKBest
    # Calculate mutual information between each feature and the target variable.
    # Mutual information is a measure of the dependency between variables.
    # A higher value indicates a stronger relationship.
    y_pred_kbest, y_pred_kbest_HO=Kbest_feature_selection(X_train_z, X_test_z, y_train, y_test,param_grid)

     #6)b) Second method : PCA
    # plot_pca(X_train_z) # plot to decide number of PCA components
    # result : we chose 10 components by lookint at the elbow of the curve
    y_pred_pca, y_pred_pca_HO=PCA_feature_reduction(X_train_z, X_test_z, y_train, y_test,param_grid)
    
    # Confusion matrices
    plot_conf_mat(y_test, y_pred, y_pred_kbest, y_pred_pca,  y_pred_HO, y_pred_kbest_HO, y_pred_pca_HO)

    


    



if __name__ == "__main__":    main()