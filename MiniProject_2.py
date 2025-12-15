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
# Segmentation

n_stimuli = len(np.unique(stimulus)) - 1 
n_repetitions = len(np.unique(repetition)) - 1 

# Initialisation
emg_windows = [[None for r in range(n_repetitions)] for s in range(n_stimuli)] 

for s in range(n_stimuli):
    for r in range(n_repetitions):
        idx = np.logical_and(stimulus == s + 1, repetition == r + 1).flatten()
        emg_windows[s][r] = emg[idx, :]

# =================================================================
# Features
def build_dataset_with_features(emg,stimulus,repetition):

    emg_features = [
        lambda x: np.mean(x, axis=0),             # MAV 
        lambda x: np.max(x, axis=0),              # Peak Amplitude
        lambda x: np.std(x, axis=0),              # Standard Deviation
        lambda x: np.sqrt(np.mean(x**2, axis=0)), # RMS
        lambda x: np.sum(np.abs(np.diff(x, axis=0)), axis=0) # Waveform Length
    ]

# =================================================================
# Building dataset

    dataset, labels = build_dataset(
        emg=emg,
        stimulus=stimulus,
        repetition=repetition,
        features=emg_features
        )
    return dataset, labels

print(f"Dataset Shape: {dataset.shape}")

# =================================================================
# Split the dataset into training and testing sets

# Here, 30% of the data is reserved for testing, and 70% is used for training
#X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.33)

# ----------------------------------------------------------------------------------------------------
# 4) Classification    emg_features = [
        lambda x: np.mean(x, axis=0),             # MAV 
        lambda x: np.max(x, axis=0),              # Peak Amplitude
        lambda x: np.std(x, axis=0),              # Standard Deviation
        lambda x: np.sqrt(np.mean(x**2, axis=0)), # RMS
        lambda x: np.sum(np.abs(np.diff(x, axis=0)), axis=0) # Waveform Length
    ]


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

def plot_feature_multi_subject(datasets_all, feature_index, n_channels, feature_names=None):
    """
    Affiche pour UNE feature donnée un subplot par subject,
    tous regroupés dans une seule figure.

    :param datasets_all: liste ou dict de datasets (dataset[i] = dataset du subject i)
    :param feature_index: index de la feature
    :param n_channels: nombre de canaux EMG
    :param feature_names: noms des features
    """

    if feature_names is None:
        feature_names = ["MAV", "STD", "MAX", "RMS", "WL", "SSC"]

    n_subjects = len(datasets_all)

    # Figure globale
    fig, axes = plt.subplots(n_subjects, 1, figsize=(12, 4*n_subjects), sharex=True)
    if n_subjects == 1:
        axes = [axes]  # uniformiser

    for i, (subj_id, dataset) in enumerate(zip(range(n_subjects), datasets_all)):

        start = feature_index * n_channels
        end = (feature_index + 1) * n_channels

        feat_data = dataset[:, start:end]  # shape (samples, channels)

        axes[i].boxplot(
            [feat_data[:, ch] for ch in range(n_channels)],
            labels=[f"Ch {c+1}" for c in range(n_channels)]
        )

        axes[i].set_title(f"Subject {subj_id} — Feature: {feature_names[feature_index]}")
        axes[i].set_ylabel("Valeurs")
        axes[i].grid(True)

    axes[-1].set_xlabel("Canaux EMG")

    plt.tight_layout()
    plt.show()

def plot_feature_violin_multi_subject(datasets_all, feature_index, n_channels, feature_names=None):
    if feature_names is None:
        feature_names = ["MAV", "STD", "MAX", "RMS", "WL", "SSC"]

    n_subjects = len(datasets_all)
    fig, axes = plt.subplots(n_subjects, 1, figsize=(12, 4*n_subjects), sharex=True)

    if n_subjects == 1:
        axes = [axes]

    for i, dataset in enumerate(datasets_all):
        start = feature_index * n_channels
        end = (feature_index + 1) * n_channels

        feat_data = dataset[:, start:end]

        axes[i].violinplot([feat_data[:, ch] for ch in range(n_channels)], showmedians=True)
        axes[i].set_title(f"Subject {i} — {feature_names[feature_index]}")
        axes[i].set_ylabel("Valeur")
        axes[i].set_xticks(range(1, n_channels + 1))
        axes[i].set_xticklabels([f"Ch {c+1}" for c in range(n_channels)])
        axes[i].grid(True)

    axes[-1].set_xlabel("Canaux EMG")
    plt.tight_layout()
    plt.show()




def plot_feature_grid(datasets_all, feature_index, n_channels, n_cols=2, feature_names=None):
    """
    Affiche les plots dans une grille (horizontal puis vertical), pour éviter
    la superposition et rendre la figure lisible.
    """

    if feature_names is None:
        feature_names = ["MAV", "STD", "MAX", "RMS", "WL", "SSC"]

    n_subjects = len(datasets_all)
    n_rows = int(np.ceil(n_subjects / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
    axes = np.array(axes).reshape(n_rows, n_cols)

    for idx, dataset in enumerate(datasets_all):

        r = idx // n_cols
        c = idx % n_cols
        ax = axes[r, c]

        start = feature_index * n_channels
        end = (feature_index + 1) * n_channels

        feat_data = dataset[:, start:end]

        ax.boxplot([feat_data[:, ch] for ch in range(n_channels)],
                    labels=[f"Ch {c+1}" for c in range(n_channels)])

        ax.set_title(f"Subject {idx} — {feature_names[feature_index]}")
        ax.set_ylabel("Valeur")
        ax.grid(True)

    # Supprimer les cases vides si la grille n'est pas complète
    for empty_idx in range(n_subjects, n_rows*n_cols):
        r = empty_idx // n_cols
        c = empty_idx % n_cols
        axes[r, c].axis("off")

    plt.tight_layout()
    plt.show()

def compute_channel1_feature_means(datasets_all, n_channels=10, n_features=6):
    """
    datasets_all : liste de datasets des subjects, chacun shape = (120, 60)
    Retourne une matrice (n_subjects, n_features)
    """
    n_subjects = len(datasets_all)

    # Matrice où on stocke la moyenne pour canal 1 de chaque feature
    result = np.zeros((n_subjects, n_features))

    for s, dataset in enumerate(datasets_all):
        for f in range(n_features):
            start = f * n_channels
            ch1_index = start + 0       # canal 1 = index 0
            # moyenne sur 120 samples
            result[s, f] = np.mean(dataset[:, ch1_index])

    return result


def compute_channel1_feature1_means(datasets_all, n_channels=10, feature_index=1):
    """
    Calcule la moyenne de la feature 1 (feature_index=1) au canal 1 (index 0)
    pour chaque subject.

    Retourne un vecteur de taille (n_subjects,)
    """
    n_subjects = len(datasets_all)
    result = np.zeros(n_subjects)

    for s, dataset in enumerate(datasets_all):
        start = feature_index * n_channels
        ch1_index = start + 0   # canal 1 = index 0
        result[s] = np.mean(dataset[:, ch1_index])

    return result

from matplotlib.gridspec import GridSpec 

def Mean_all_features_one_chanel(datasets_all, n_channels=10, n_features=6, feature_names=None):
   
    n_subjects = len(datasets_all)

    # Matrice features × subjects
    mean_matrix = np.zeros((n_features, n_subjects))

    for f in range(n_features):
        for s, dataset in enumerate(datasets_all):
            start = f * n_channels
            ch1_index = start + 0         # canal 1
            mean_matrix[f, s] = np.mean(dataset[:, ch1_index])
    return mean_matrix


def plot_heatmaps_all_features(mean_matrix, n_channels=10, n_features=6, feature_names=None):
    if feature_names is None:
        feature_names = ["MAV", "STD", "MAX", "RMS", "WL", "SSC"]

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

        # Padding supérieur pour éviter de toucher les labels X
        ax.set_title(f"Feature : {feature_names[f]} (Canal 1)", pad=12, fontsize=10)

        # Tous les axes sauf le dernier n'affichent pas les ticks X
        if f < n_features - 1:
            ax.set_xticks([])

    # --- Dernier subplot : labels X + nom de l’axe ---
    axes[-1].set_xticks(range(n_subjects))
    axes[-1].set_xticklabels([str(i+1) for i in range(n_subjects)], rotation=45)
    axes[-1].set_xlabel("Subject")

    # --- Colorbar sur le côté (pas superposée) ---
    cbar = fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.03, pad=0.02)
    cbar.set_label("Mean value", rotation=90)

    #plt.tight_layout(rect=[0, 0, 0.96, 1])  
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


########################
# Confusion matrices
#plot_conf_mat(y_test, y_pred, y_pred_kbest, y_pred_pca,  y_pred_HO, y_pred_kbest_HO, y_pred_pca_HO)


def main():
    ####PART 1: Single subject Classification####
    # Load data
    fs_1 = 100

    S2_A1_E1 = sio.loadmat('dataset/s2/S2_A1_E1.mat')
    stimulus = S2_A1_E1['restimulus']
    repetition = S2_A1_E1['rerepetition']
    emg_raw = S2_A1_E1['emg']

   
    #1) Preprocessing
    #emg_filtered = bandpass_filter_emg(emg_raw,  fs)
    #emg_windows, emg_envelopes_cleaned= preprocess_emg(emg_filtered, fs, stimulus, repetition)



    #3) Dataset building with feature extraction
    #dataset, labels = build_dataset_with_features(emg_filtered,stimulus,repetition)

    #plot_feature(dataset, feature_index=0, n_channels=emg_raw.shape[1])  # Plot MAV feature distribution
    
    # 2) Split the dataset into training and testing sets
    # Here, 30% of the data is reserved for testing, and 70% is used for training
    #X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.33)

    #4) Classification
    # Normalizing the data
    # StandardScaler is used to scale the features so that they have a mean of 0 and a standard deviation of 1
    # scaler = StandardScaler()
    # X_train_z = scaler.fit_transform(X_train)  # Fit the scaler on the training data and transform it
    # X_test_z = scaler.transform(X_test)        # Transform the test data using the same scaler

    #y_pred=classify_without_hyperparameter_optimization(X_train_z, X_test_z, y_train, y_test)

    # param_grid = {
    #     "n_estimators": [50, 100, 150], #default : 100
    #     "learning_rate": [0.01, 0.05, 0.1], #default : 0.1
    #     "max_depth": [2, 3, 4], #default : 3
    # }
   # y_pred_HO=classify_with_hyperparameter_optimization(X_train_z, X_test_z, y_train, y_test,param_grid)
    
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
   # y_pred_kbest, y_pred_kbest_HO=Kbest_feature_selection(X_train_z, X_test_z, y_train, y_test,param_grid)

     #6)b) Second method : PCA
    # plot_pca(X_train_z) # plot to decide number of PCA components
    # result : we chose 10 components by lookint at the elbow of the curve
    #y_pred_pca, y_pred_pca_HO=PCA_feature_reduction(X_train_z, X_test_z, y_train, y_test,param_grid)
    
    # Confusion matrices
    #plot_conf_mat(y_test, y_pred, y_pred_kbest, y_pred_pca,  y_pred_HO, y_pred_kbest_HO, y_pred_pca_HO)

    ###Part 2: Multi-Subject Classification###
    #extract data from all subjects
    all_subjects = [] 
    for i in range(1, 28):  # sujets 1 à 27
        filename = f"dataset/s{i}/S{i}_A1_E1.mat"   # construit le chemin
        data = sio.loadmat(filename)                # charge le fichier
        all_subjects.append(data)                   # stocke dans la liste

    emg_filtered_all = []
    emg_windows_all = []
    stimulus_all = []
    repetition_all = []
    datasets_all = []
    labels_all = []
    # 1) prepocessing for all subjects
    for i in range(len(all_subjects)):
        emg_raw_subject = all_subjects[i]['emg']
        stimulus_subject = all_subjects[i]['restimulus']
        repetition_subject = all_subjects[i]['rerepetition']
        fs_1 = 100 #sampling frequency 100hz
        #No preprocessing step as 
        #emg_filtered_all.append(emg_highpass_filter(emg_raw_subject, fs_1))
        #emg_windows_all.append(preprocess_emg(emg_filtered_all[i], fs_1,stimulus_subject,repetition_subject))
        #emg_envelopes_cleaned_all.append(clean_emg_envelope(emg_filtered_all[i], emg_continusous_env_all[i],stimulus_subject,repetition_subject))
        stimulus_all.append(stimulus_subject)
        repetition_all.append(repetition_subject)
        dataset_subject, labels_subject = build_dataset_with_features(emg_raw_subject,stimulus_subject,repetition_subject)
        datasets_all.append(dataset_subject)
        labels_all.append(labels_subject)
       
    # # #2) Extract features 
    # mean_matrix = Mean_all_features_one_chanel(datasets_all, n_channels=emg_raw_subject.shape[1], n_features=6)
    # plot_heatmaps_all_features(mean_matrix)


    
    #3) Classification
    #Subject 1 for TEST
    # X_test = datasets_all[0]
    # y_test = labels_all[0]
    # # Subject 2 à 27 for TRAIN
    # X_train = np.vstack(datasets_all[1:])   # concatène verticalement (tous samples)
    # y_train = np.hstack(labels_all[1:])    
    # # Normalizing the data
    # # StandardScaler is used to scale the features so that they have a mean of 0 and a standard deviation of 1
    # scaler = StandardScaler()
    # X_train_z = scaler.fit_transform(X_train)  # Fit the scaler on the training data and transform it
    # X_test_z = scaler.transform(X_test)        # Transform the test data using the same scaler
    # y_pred=classify_without_hyperparameter_optimization(X_train_z, X_test_z, y_train, y_test)
 
    # ##Seems to be worse than test and train on single subject only##
    #4) Cross validation
    # accuracy_scores_all = []
    # f1_scores_all = []
    # for i in range(len(datasets_all)):
    #     X_test_cv = datasets_all[i]
    #     y_test_cv = labels_all[i]
    #     X_train_cv = np.vstack(datasets_all[:i] + datasets_all[i+1:])
    #     y_train_cv = np.hstack(labels_all[:i] + labels_all[i+1:])
    #     scaler_cv = StandardScaler()
    #     X_train_cv_z = scaler_cv.fit_transform(X_train_cv)
    #     X_test_cv_z = scaler_cv.transform(X_test_cv)
    #     y_pred_cv=classify_without_hyperparameter_optimization(X_train_cv_z, X_test_cv_z, y_train_cv, y_test_cv)
    #     print(f"Cross validation Subject {i+1} completed.")
    #     accuracy_scores_all.append(accuracy_score(y_test_cv, y_pred_cv))
    #     f1_scores_all.append(f1_score(y_test_cv, y_pred_cv, average='macro'))
    
    # plot_cv_scores(accuracy_scores_all, f1_scores_all)

    #5) Variying number of subjects in training set
    #print("len datasets_all:", len(datasets_all))
    id_subject_for_training = []
    id_subject_for_training.append(0)
    accuracy_score_varied_all=[]
    f1_score__varied_all=[]
   
    subject_for_testing = 26  # Subject 1 for testing
    for i in range(20):
        datasets_training = [datasets_all[i] for i in id_subject_for_training]
        print(len(datasets_training))
        labels_training = [labels_all[i] for i in id_subject_for_training]
        X_train_varied = np.vstack(datasets_training)
        y_train_varied = np.hstack(labels_training)
        X_test_varied = datasets_all[subject_for_testing]
        y_test_varied = labels_all[subject_for_testing]
        scaler_varied = StandardScaler()
        X_train_varied_z = scaler_varied.fit_transform(X_train_varied)
        X_test_varied_z = scaler_varied.transform(X_test_varied)
        y_pred_varied=classify_without_hyperparameter_optimization(X_train_varied_z, X_test_varied_z, y_train_varied, y_test_varied)
        accuracy_score_varied = accuracy_score(y_test_varied, y_pred_varied)
        accuracy_score_varied_all.append(accuracy_score_varied)
        f1_score_varied = f1_score(y_test_varied, y_pred_varied, average='macro')
        f1_score__varied_all.append(f1_score_varied)
        print(f"Varied subjects training - Accuracy: {accuracy_score_varied:.3f}, F1 Score: {f1_score_varied:.3f}")
        id_subject_for_training.append(i)

    plot_cv_scores(f1_score__varied_all, f1_score__varied_all)



if __name__ == "__main__":    main()