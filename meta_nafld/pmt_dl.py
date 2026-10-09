import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, matthews_corrcoef, f1_score, roc_auc_score, balanced_accuracy_score, precision_score, confusion_matrix
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Conv1D, MaxPooling1D, Flatten, Input
from tensorflow.keras.optimizers import Adam
from sklearn.neighbors import NearestNeighbors
from joblib import dump
import joblib
import matplotlib.pyplot as plt
import shap
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_curve, auc
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score


# Define the directory name
name = "deep learning model"  # <-- Change this to your actual directory

# Load y_train and y_test
y_train = np.array(pd.read_csv(os.path.join(name, "train", "y_train.csv"), index_col=0).values.ravel(), dtype=float)
y_test = np.array(pd.read_csv(os.path.join(name, "test", "y_test.csv"), index_col=0).values.ravel(), dtype=float)

# Load stacked_train and stacked_test
stacked_train = pd.read_csv("stacked_train.csv").values
stacked_test = pd.read_csv("stacked_test.csv").values

# Save stacked_train and stacked_test to CSV (if needed)
pd.DataFrame(stacked_train).to_csv("stacked_train.csv", index=False)
pd.DataFrame(stacked_test).to_csv("stacked_test.csv", index=False)
# Define Meta-Model (Dense Neural Network)
def create_meta_model(input_shape):
    inputs = Input(shape=(input_shape,))
    x = Dense(32, activation='relu')(inputs)
    x = Dense(16, activation='relu')(x)
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def permutation_importance(meta_model, x_test, y_test, n_repeats=10):
    def get_predictions(X):
        y_pred = meta_model.predict(X)
        return np.argmax(y_pred, axis=1) if y_pred.ndim > 1 and y_pred.shape[1] > 1 else (y_pred > 0.5).astype(int)
    
    baseline_score = accuracy_score(y_test, get_predictions(x_test))
    importances = np.zeros(x_test.shape[1])
    
    for i in range(x_test.shape[1]):
        score_diffs = []
        for _ in range(n_repeats):
            x_test_perm = x_test.copy()
            np.random.shuffle(x_test_perm[:, i])
            permuted_score = accuracy_score(y_test, get_predictions(x_test_perm))
            score_diffs.append(baseline_score - permuted_score)
        importances[i] = np.mean(score_diffs)
    
    return importances

def train_permutation(name, x_train, x_test, y_train, y_test, feature_names,epochs=50, batch_size=2):
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train).ravel()
    y_test = np.array(y_test).ravel()
    # Meta model
    meta_model = create_meta_model(stacked_train.shape[1])
    meta_model.fit(stacked_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=0)
    meta_model_filename = os.path.join(name, 'stack_permutation.keras')
    meta_model.save(meta_model_filename)
    
    importance_scores = permutation_importance(meta_model, x_test, y_test)
    sorted_indices = np.argsort(importance_scores)[::-1]
    sorted_features = [feature_names[i] for i in sorted_indices]
    sorted_importance_scores = [importance_scores[i] for i in sorted_indices]
    
    return importance_scores, sorted_features, sorted_importance_scores

def run_permutation(name, x_train, x_test, y_train, y_test):
    # Manually define feature names
    feature_names = [
        "BiLSTM_xat_train", "CNN_xat_train",
        "BiLSTM_xes_train", "CNN_xes_train",
        "BiLSTM_xke_train", "CNN_xke_train",
        "BiLSTM_xpc_train", "CNN_xpc_train",
        "BiLSTM_xss_train", "CNN_xss_train",
        "BiLSTM_xcd_train", "CNN_xcd_train",
        "BiLSTM_xcn_train", "CNN_xcn_train",
        "BiLSTM_xkc_train", "CNN_xkc_train",
        "BiLSTM_xce_train", "CNN_xce_train",
        "BiLSTM_xsc_train", "CNN_xsc_train",
        "BiLSTM_xac_train", "CNN_xac_train",
        "BiLSTM_xma_train", "CNN_xma_train"
    ]

    # Ensure feature names match the dataset dimensions
    if len(feature_names) != x_test.shape[1]:
        raise ValueError(f"Feature names length {len(feature_names)} does not match dataset shape {x_test.shape[1]}.")

    importance_scores, sorted_features, sorted_importance_scores = train_permutation(name, x_train, x_test, y_train, y_test, feature_names)

    df_sorted_features = pd.DataFrame({
        'Feature': sorted_features,
        'Importance Score': sorted_importance_scores
    })
    save_features = os.path.join(name, 'cnn_stack_importance.csv')
    df_sorted_features.to_csv(save_features, index=False)
    print('Sorted features with importance scores saved successfully!')

    print("Feature Names:", feature_names)
    print("Importance Scores:", importance_scores)
    print("Sorted Features:", sorted_features)
    print("Sorted Importance Scores:", sorted_importance_scores)

    top_n = 5
    print(f"Top {top_n} Feature Importance Ranking:")
    for i in range(min(top_n, len(sorted_features))):
        feature = sorted_features[i]
        importance_score = sorted_importance_scores[i]
        print(f"{i+1}. {feature}: {importance_score:.3f}")

    print("Feature Importance Scores:")
    for i, score in enumerate(importance_scores):
        print(f"Feature {feature_names[i]}: Score {score:.3f}")

    df_sorted_features = df_sorted_features.sort_values(by='Importance Score', ascending=False)
    top_features = df_sorted_features.head(10)
    print(top_features)
    plt.figure(figsize=(4, 4))
    plt.barh(top_features['Feature'], top_features['Importance Score'], color='orange')
    plt.xlabel('Permutation Importance', fontsize=12, fontstyle='italic', weight="bold")
    plt.ylabel('Features', fontsize=12, fontstyle='italic', weight="bold")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    save_fig = os.path.join(name, 'cnn_stack_permutation_5.svg')
    plt.savefig(save_fig, format='svg')
    plt.close()

    print("Finished permutation!")
