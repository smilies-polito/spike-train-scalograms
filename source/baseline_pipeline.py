import pickle

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, balanced_accuracy_score
from skopt import BayesSearchCV
from lightgbm import LGBMClassifier, plot_importance, Booster
import matplotlib.pyplot as plt
import shap
import random
# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)
first_saving=False
input_path=os.path.join("..", "data", "Split")

# Function to get top-N most important feature indices for a given class
def get_top_features(shap_values_array, X, class_index, top_n=5):
    """
    Returns the indices of the top N most important features for a given class.
    """
    class_shap_values = shap_values_array[:, :, class_index]  # Extract SHAP values for this class
    mean_abs_shap = np.absolute(class_shap_values.values).mean(axis=0)  # Compute mean absolute SHAP per feature
    top_feature_indices = np.argsort(mean_abs_shap)[-top_n:]  # Get indices of top N features
    return top_feature_indices

# Function to plot SHAP summary for top N features
def plot_top_shap(shap_values_array, X, class_index, class_name, top_n=5):
    """
    Plots SHAP summary for a specific class using only top N features.
    """
    top_feature_indices = get_top_features(shap_values_array, X, class_index, top_n)  # Get top feature indices
    X_subset = X.iloc[:, top_feature_indices]  # Subset data using column indices
    shap_values_subset = shap_values_array[:, top_feature_indices, class_index]  # Select SHAP values correctly

    print(f"Plotting SHAP summary for {class_name}. Top features: {X.columns[top_feature_indices]}")  # Restore to default or another size
    shap.summary_plot(shap_values_subset, X_subset, title=f'Top-{top_n} Features - {class_name}', show=False)
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)
    plt.savefig(os.path.join("..", "output", f"shap_summary_plot_for_{class_name}.png"), dpi=300)

# Load the datasets
patch_clamp_df = pd.read_csv(os.path.join(input_path, "PatchClamp_EP_features.csv"))
patch_seq_df = pd.read_csv(os.path.join(input_path,"PatchSeq_EP_features.csv"))
cell_types_df = pd.read_csv(os.path.join("..", "data", "cell_types_GouwensAll_new.txt"), delimiter=' ', names=["Label", "specimen_id"])
train_split_df = pd.read_csv(os.path.join(input_path,"Train_split.csv"), header=None, names=["specimen_id", "dataset_info"])
test_split_df = pd.read_csv(os.path.join(input_path,"Test_split.csv"), header=None, names=["specimen_id", "dataset_info"])

# Rename column if necessary
if "cell_specimen_id" in patch_seq_df.columns:
    patch_seq_df.rename(columns={"cell_specimen_id": "specimen_id"}, inplace=True)

if "cell_specimen_id" in patch_clamp_df.columns:
    patch_clamp_df.rename(columns={"cell_specimen_id": "specimen_id"}, inplace=True)


# Define required features
required_features = [
    "f_i_curve_slope",
    "avg_isi",
    "adaptation",
    "latency",
    "has_burst",
    "has_pause",
    "has_delay",
    "specimen_id",
    "long_square"
]

# Filter columns by feature names
def filter_columns(columns, required_features):
    return [col for col in columns if any(feature in col.lower() for feature in required_features)]

# Rename columns for consistency
feature_mapping = {
    "avg_isi": "average_ISI",
    "has_burst": "had_burst",
    "has_pause": "had_pause",
    "has_delay": "had_delay"
}
patch_clamp_df.rename(columns=feature_mapping, inplace=True)
patch_seq_df.rename(columns=feature_mapping, inplace=True)

# Filter columns
patch_clamp_filtered_columns = filter_columns(patch_clamp_df.columns, required_features)
patch_seq_filtered_columns = filter_columns(patch_seq_df.columns, required_features)

patch_clamp_df_filtered = patch_clamp_df[patch_clamp_filtered_columns]
patch_seq_df_filtered = patch_seq_df[patch_seq_filtered_columns]

# Retain only valid specimen IDs
valid_specimen_ids = set(train_split_df['specimen_id']).union(set(test_split_df['specimen_id']))
patch_clamp_df_filtered = patch_clamp_df_filtered[patch_clamp_df_filtered['specimen_id'].isin(valid_specimen_ids)]
patch_seq_df_filtered = patch_seq_df_filtered[patch_seq_df_filtered['specimen_id'].isin(valid_specimen_ids)]

# Align datasets on common columns
common_columns = set(patch_clamp_df_filtered.columns).intersection(set(patch_seq_df_filtered.columns))
patch_clamp_df_filtered = patch_clamp_df_filtered[list(common_columns)]
patch_seq_df_filtered = patch_seq_df_filtered[list(common_columns)]

# Combine datasets and add labels
combined_df = pd.concat([patch_clamp_df_filtered, patch_seq_df_filtered], ignore_index=True)
combined_df = combined_df.merge(cell_types_df, on="specimen_id", how="left").drop_duplicates(subset="specimen_id")

# Split into Train and Test datasets
train_df = combined_df[combined_df['specimen_id'].isin(train_split_df['specimen_id'])]
test_df = combined_df[combined_df['specimen_id'].isin(test_split_df['specimen_id'])]

print("Data loading and reorganization completed")
print(f'Shape of test dataframe', test_df.shape)
print(f'Shape of train dataframe', train_df.shape)

# Prepare training data
X_train = train_df.drop(columns=["specimen_id", "Label"])
y_train = train_df["Label"]
X_test = test_df.drop(columns=["specimen_id", "Label"])
y_test = test_df["Label"]
specimen_ids=test_df['specimen_id']

# Encode labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)


# Define the LightGBM parameter search space
search_space = {
    "num_leaves": (10, 100),
    "max_depth": (3, 12),
    "learning_rate": (1e-3, 0.2, "log-uniform"),
    "n_estimators": (50, 300),
    "min_child_samples": (5, 30),
    "subsample": (0.5, 1.0, "uniform"),
    "colsample_bytree": (0.5, 1.0, "uniform"),
    "reg_alpha": (0.0, 1.0, "uniform"),
    "reg_lambda": (0.0, 1.0, "uniform"),
}

best_model_path=os.path.join("..", "models")
if os.path.isfile(os.path.join(best_model_path, 'best_model_baseline.txt')): #to speed-up testing, the pre-trained Booster is loaded
    best_model=Booster(model_file=os.path.join(best_model_path, 'best_model_baseline.txt'))
    # Get the model's features in the correct order
    cols = best_model.feature_name()  # -> ['feat1', 'feat2', 'feat3', ...]
    # Use col to reindex the prediction DataFrame
    X_test = X_test.reindex(columns=cols)  # -> df now has the same col ordering as the model

else:
    #Initialize the LGBMClassifier
    lgbm = LGBMClassifier(random_state=39, verbose=-1)
    cv=StratifiedKFold(n_splits=3, random_state=42, shuffle=True)
    # Initialize BayesSearchCV
    bayes_search = BayesSearchCV(
        estimator=lgbm,
        scoring='balanced_accuracy',
        search_spaces=search_space,
        n_iter=50,
        cv=cv,
        random_state=42,
        n_jobs=-1,
        refit=True,
    )

    # Run the optimization
    bayes_search.fit(X_train, y_train_encoded)

    # Get the best model and predict on the test set
    best_model = bayes_search.best_estimator_
    best_model.booster_.save_model(os.path.join(best_model_path, 'best_model_baseline.txt'))
    first_saving=True

    # Get cross-validation results from the BayesSearchCV object
    cv_results = bayes_search.cv_results_['mean_test_score']  # Mean accuracy over the folds
    std_results = bayes_search.cv_results_['std_test_score']  # Standard deviation of accuracy over the folds

    # Print mean and standard deviation of accuracy across 5 folds
    mean_acc = cv_results[bayes_search.best_index_]
    std_acc = std_results[bayes_search.best_index_]

    print(f"Mean CV accuracy: {mean_acc:.4f} ± {std_acc:.4f}")

y_pred_encoded = best_model.predict(X_test) #the LGBMCLassifier predicts classes, so only decoding is needed
if not first_saving: #the pre-trained booster predicts per-class probabilities
    y_pred_encoded=np.argmax(y_pred_encoded, axis=1)

# Decode the predictions

df = pd.DataFrame({'Predictions':y_pred_encoded, 'True_labels':y_test, 'SpecimenID':specimen_ids})
# Assuming `test_split_df` has a column named 'specimen_id' with the desired order
ordered_specimen_ids = test_split_df['specimen_id']

# Reordering the `df` dataframe based on `test_split_df`
df_reordered = df.set_index('SpecimenID').loc[ordered_specimen_ids].reset_index()
y_pred = label_encoder.inverse_transform(y_pred_encoded)

# Generate classification results
conf_matrix = confusion_matrix(y_test, y_pred, labels=label_encoder.classes_, normalize=None)
class_report = classification_report(y_test, y_pred, labels=label_encoder.classes_, target_names=label_encoder.classes_)
overall_accuracy = accuracy_score(y_test, y_pred)
balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
per_class_accuracy = conf_matrix.diagonal() / conf_matrix.sum(axis=1)

# Display results
print("Confusion Matrix:")
print(pd.DataFrame(conf_matrix, index=label_encoder.classes_, columns=label_encoder.classes_))
print("\nClassification Report:")
print(class_report)
print("\nOverall Accuracy:", overall_accuracy)
print("Balanced Accuracy:", balanced_accuracy)
print("Per Class Accuracy:", dict(zip(label_encoder.classes_, per_class_accuracy)))


#interpretability using shap
X_test.rename(columns=lambda x: x.replace('_long_square', ''), inplace=True)
X_test.rename(columns={'upstroke_downstroke_ratio': 'upstroke/downstroke'}, inplace=True)
explainer = shap.TreeExplainer(best_model)  # Assuming best_model is a tree-based model
shap_values = explainer(X_test)  # SHAP values for all classes
# Plot for each class
plot_top_shap(shap_values, X_test, 0, 'Excitatory')
plot_top_shap(shap_values, X_test, 1, 'PValb')
plot_top_shap(shap_values, X_test, 2, 'Sst')
plot_top_shap(shap_values, X_test, 3, 'Vip/Lamp5')
