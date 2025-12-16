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


def main():
    ####PART 1: Single subject Classification####
    # Load data
    print("Load data subject 2")
    fs_1 = 100
    S2_A1_E1 = sio.loadmat('dataset/s2/S2_A1_E1.mat')
    stimulus = S2_A1_E1['restimulus']
    repetition = S2_A1_E1['rerepetition']
    emg_raw = S2_A1_E1['emg']

    #1) Segementation
    print("Segementation")
    emg_windows = create_emg_windows(stimulus,repetition,emg_raw)

    #3) Dataset building with feature extraction
    print("Build dataset with feature")
    dataset, labels = build_dataset_with_features(emg_windows)

    #2) Split the dataset into training and testing sets
    # Here, 30% of the data is reserved for testing, and 70% is used for training
    print("Split dataset")
    X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.33)

    #4)Classification
    # Normalizing the data
    # StandardScaler is used to scale the features so that they have a mean of 0 and a standard deviation of 1
    print("Classification with Gradient Boosting without hyperparameter optimization")
    scaler = StandardScaler()
    X_train_z = scaler.fit_transform(X_train)  # Fit the scaler on the training data and transform it
    X_test_z = scaler.transform(X_test)        # Transform the test data using the same scaler

    y_pred=classify_without_hyperparameter_optimization(X_train_z, X_test_z, y_train, y_test)

    # # Perform cross-validation (For testing purpose)
    # scores = cross_val_score(clf, X_train_z, y_train, cv=3)
    # formatted_scores = [f"{s:.3f}" for s in scores]
    # print(f"4) Cross validation : Accuracy scores of all models: {formatted_scores} (GB without hyperparameter optimization)")
    # print(f"4) Cross validation : Mean accuracy across all models: {np.mean(scores):.3f} (GB without hyperparameter optimization)")

    # Hyperparameter optimization
    print("Classification with Gradient Boosting with hyperparameter optimization")

    param_grid = {
        "n_estimators": [50, 100, 150], #default : 100
        "learning_rate": [0.01, 0.05, 0.1], #default : 0.1
        "max_depth": [2, 3, 4], #default : 3
    }
    y_pred_HO=classify_with_hyperparameter_optimization(X_train_z, X_test_z, y_train, y_test,param_grid)
    
    # 5) Performance evaluation
    # Different possible metrics : Accuracy, F1-score, Confusion matrix
    # F1 = better than accuracy for imbalanced class distibutions
    # But we have exactly 10 repetitions for each movement, so the classes are balanced -> accuracy is sufficient
    # Confusion matrix : useful for qualitative analysis, but not really a comparison metric (too many values)

    # 6) Feature selection / dimension reduction using 2 methods : SelectKBest, PCA
    # 6)a) First method : SelectKBest
    print("Feature selection with Kbest")
    # Calculate mutual information between each feature and the target variable.
    # Mutual information is a measure of the dependency between variables.
    # A higher value indicates a stronger relationship.
    y_pred_kbest, y_pred_kbest_HO=Kbest_feature_selection(X_train_z, X_test_z, y_train, y_test,param_grid)

    #6)b) Second method : PCA
    print("Feature selection with PCA")
    # plot_pca(X_train_z) # plot to decide number of PCA components
    # result : we chose 10 components by lookint at the elbow of the curve
    y_pred_pca, y_pred_pca_HO=PCA_feature_reduction(X_train_z, X_test_z, y_train, y_test,param_grid)
    
    # Confusion matrices
    plot_conf_mat(y_test, y_pred, y_pred_kbest, y_pred_pca,  y_pred_HO, y_pred_kbest_HO, y_pred_pca_HO)

    ###Part 2: Multi-Subject Classification###
    #extract data from all subjects
    print("Part 2 :  Multi-Subject Classification")
    print("Load data")
    all_subjects = [] 
    for i in range(1, 28):  # sujets 1 à 27
        filename = f"dataset/s{i}/S{i}_A1_E1.mat"  
        data = sio.loadmat(filename)               
        all_subjects.append(data)                  

    #create varaibles to store data
    emg_windows_all = []
    stimulus_all = []
    repetition_all = []
    datasets_all = []
    labels_all = []

    #1) Store data 
    print("Store data")
    for i in range(len(all_subjects)):
        emg_raw_subject = all_subjects[i]['emg']
        stimulus_subject = all_subjects[i]['restimulus']
        repetition_subject = all_subjects[i]['rerepetition']
        fs_1 = 100 #sampling frequency 100hz

        emg_windows_all.append(create_emg_windows(stimulus_subject,repetition_subject,emg_raw_subject))
        stimulus_all.append(stimulus_subject)
        repetition_all.append(repetition_subject)
        dataset_subject, labels_subject = build_dataset_with_features(emg_windows_all[i])
        datasets_all.append(dataset_subject)
        labels_all.append(labels_subject)

    #2) Extract features 
    print("Extract feature")
    mean_matrix = Mean_all_features_one_chanel(datasets_all, n_channels=emg_raw_subject.shape[1])
    plot_heatmaps_all_features(mean_matrix)


    #3) Classification
    print("Classification: subject 1 for testing and  subject 2 to 27 for training")
    #Subject 1 for TEST
    X_test = datasets_all[0]
    y_test = labels_all[0]
    # Subject 2 à 27 for train
    X_train = np.vstack(datasets_all[1:]) 
    y_train = np.hstack(labels_all[1:])    
    # Normalizing the data
    scaler = StandardScaler()
    X_train_z = scaler.fit_transform(X_train)  # Fit the scaler on the training data and transform it
    X_test_z = scaler.transform(X_test)        # Transform the test data using the same scaler
    y_pred=classify_without_hyperparameter_optimization(X_train_z, X_test_z, y_train, y_test)

    #4) Cross validation
    print("Cross Validation by rotating the subject for testing")
    accuracy_scores_all = []
    f1_scores_all = []
    for i in range(len(datasets_all)):
        X_test_cv = datasets_all[i]
        y_test_cv = labels_all[i]
        X_train_cv = np.vstack(datasets_all[:i] + datasets_all[i+1:])
        y_train_cv = np.hstack(labels_all[:i] + labels_all[i+1:])
        scaler_cv = StandardScaler()
        X_train_cv_z = scaler_cv.fit_transform(X_train_cv)
        X_test_cv_z = scaler_cv.transform(X_test_cv)
        y_pred_cv=classify_without_hyperparameter_optimization(X_train_cv_z, X_test_cv_z, y_train_cv, y_test_cv)
        print(f"Cross validation Subject {i+1} completed.")
        accuracy_scores_all.append(accuracy_score(y_test_cv, y_pred_cv))
        f1_scores_all.append(f1_score(y_test_cv, y_pred_cv, average='macro'))
    
    plot_cv_scores(accuracy_scores_all, f1_scores_all)

    #5) Variying number of subjects in training set
    print("Variying the number of subject to train")
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