#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import xgboost as xgb
import pickle

import shap
from collections import defaultdict

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve, confusion_matrix, classification_report
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.base import clone
from sklearn.metrics import r2_score
from sklearn.metrics import make_scorer

from processing_functions import *

pd.set_option('display.max_colwidth', None)
df = pd.read_csv('../data/input_file.csv')
#nan_percentage(df) 


### Processing

# There is a lot of data missing for how people think others perceive them, and what attributes they think the rest of their sex prefer. 
# Since I am also going to be working with no gender, I will drop any column related to the binary separation of gender.
# The attribute that overall has the most amount of missing data is 'shared interests'. I was hoping to use this attribute as a reliable feature to find partners, but based on the findings made in the EDA it is the feature that is the least important. 

def drop_columns(df, suffix):
    cols = [col for col in df.columns if any(col.endswith(s) for s in suffix)]
    
    df = df.drop(columns=cols, errors='ignore')
    
    return df
    

suffix= ['_fellow_want', 'perceived', '_o_want', '_diff']
df = drop_columns(df, suffix)
df_final = df.drop(columns=['Unnamed: 0'])

# Looking into how many waves are left after dropping all rows with nan values
df_nan = df_final.dropna()
df_nan['wave'].unique()

# Waves left are 1, 2, 3, 4 and 5
# Wave 1 is the one with the least amount of missing data, will rely on it.
waves = [1, 2, 3, 4, 5]
df_1 = df_final[df_final['wave'] == 1]
df_rest = df_final[df_final['wave'].isin([2, 3, 4, 5])].dropna()

# Fill those columns (prob and prob_o) that have missing data by their partner's decision
def fill_nan(df, col1, col2):
    """
    Fills NaN values in two columns of a DataFrame based on each other's non-NaN values.

    Parameters:
    df (pd.DataFrame): DataFrame containing the columns to be processed.
    col1 (str): Name of the first column.
    col2 (str): Name of the second column.

    Returns:
    pd.DataFrame: DataFrame with NaN values filled in the specified columns.
    """
    df_filled = df.copy()
    df_filled[col1] = df_filled[col1].fillna(df_filled[col2])
    df_filled[col2] = df_filled[col2].fillna(df_filled[col1])

    return df_filled

df_1 = fill_nan(df_1, 'prob', 'prob_o')
# Rest of the columns with missing data (shar, amb, attr, attr_o, amb_o, shar_o) are filled with 1
df_1 = df_1.fillna(1) 

columns = df_1.columns.to_list()
df_final = pd.concat([df_1, df_rest], axis=0, ignore_index=True)

# Save the final dataframe in order to use it later to predict the preference lists.
df_final.to_csv('../data/df_final.csv')

# Dropping identifier columns and irrelevant features for prediction
df = df_final.drop(columns=['iid', 'id', 'pid', 'partner', 'wave', 'dec', 'dec_o', 'age', 'age_o'], axis=1)

### Models training
X = df.drop(['match'], axis=1)
y = df['match']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

# Start by modelling a LightGBM and Random Forest model 
# Will do comparison of both and decide based on 
# maybe can try to use LightGBM but using 'random forest' as the boosting_type(?)

lgb_model = lgb.LGBMClassifier(random_state=42, verbose=-1)
rf_model = RandomForestClassifier(random_state=42)
xgb_model = xgb.XGBClassifier(random_state=42)

lgb_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)
xgb_model.fit(X_train, y_train)

# rf_accuracy = cross_val_score(rf_model, X_train, y_train, cv=5)
# lgb_accuracy = cross_val_score(lgb_model, X_train, y_train, cv=5)
# xgb_accuracy = cross_val_score(xgb_model, X_train, y_train, cv=5)

# print("Random Forest Accuracy: ", rf_accuracy.mean())
# print("LightGBM Accuracy: ", lgb_accuracy.mean())
# print("XGBClassifier Accuracy: ", xgb_accuracy.mean())

### Comparison between models with classification_report
# Make prediction on the testing data
y_pred_rf = rf_model.predict(X_test)
y_pred_lgb = lgb_model.predict(X_test)
y_pred_xgb = xgb_model.predict(X_test)

## Classification Report
# print("RandomForestClassifier:\n", classification_report(y_test, y_pred_rf, zero_division=0))
# print("LightGBM:\n", classification_report(y_pred_lgb, y_test))
# print("XGBClassifer:\n", classification_report(y_pred_xgb, y_test))


### F1 Score
# print("RandomForestClassifier F1 score:\n", f1_score(y_test, y_pred_rf, zero_division=0, average="weighted"),"\n")
# print("LightGBM F1 score:\n", f1_score(y_pred_lgb, y_test, average="weighted"),"\n")
# print("XGBClassifer F1 score:\n", f1_score(y_pred_xgb, y_test, average="weighted"))


### Gridsearch
f1 = make_scorer(f1_score, average='weighted')
kf = KFold(n_splits=3, shuffle=True, random_state=42) 

## XGBClassifier
xgb_param_grid = {
    'max_depth': [3, 6, 10],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [50, 100, 200],
    'subsample': [0.8, 1],
    'colsample_bytree': [0.8, 1],
    'random_state': [42],
    'scale_pos_weight': [np.sum(y==0) / np.sum(y==1)]
}

xgb_classifier = xgb.XGBClassifier(objective='binary:logistic', use_label_encoder=False, eval_metric='logloss')

grid_search_xgb_classifier = GridSearchCV(estimator=xgb_classifier, param_grid=xgb_param_grid, cv=kf, verbose=3, n_jobs=-1, scoring=f1)

# Fit GridSearchCV
grid_search_xgb_classifier.fit(X_train, y_train)

# print("Best parameters found xgb_classifier:\n", grid_search_xgb_classifier.best_params_)
# print("\n The best score across ALL searched params for xgb_classifier:",grid_search_xgb_classifier.best_score_)
xgb_model = grid_search_xgb_classifier.best_estimator_


## Random Forest
rf_params = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'class_weight': ['balanced'],  # Handling imbalance
    'criterion': ['gini', 'entropy', 'log_loss'],
    'max_features': ['sqrt', 'log2']
}
rf_grid = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=rf_params, cv=kf, verbose=3, n_jobs=-1, scoring=f1, error_score='raise')

rf_grid.fit(X_train, y_train)

# print("Best parameters found rf_classifier:\n", rf_grid.best_params_)
# print("\n The best score across ALL searched params for rf_classifier:",rf_grid.best_score_)
# rf_model = rf_grid.best_estimator_

## LGBClassifier
lgb_params = {
    'num_leaves': [25, 30, 50],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [50, 100, 200],
    'scale_pos_weight': [np.sum(y==0) / np.sum(y==1)],  # Handling imbalance
    'boosting_type': ['gbdt'],
    'objective': ['binary'],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

lgb_grid = GridSearchCV(estimator=lgb.LGBMClassifier(random_state=42, verbose=-1), param_grid=lgb_params, cv=kf, verbose=3, n_jobs=-1, scoring=f1)

lgb_grid.fit(X_train, y_train)
# print("Best parameters found lgb_classifier:\n", lgb_grid.best_params_)
# print("\n The best score across ALL searched params for lgb_classifier:",lgb_grid.best_score_)
# lgb_model = lgb_grid.best_estimator_


# Save the chosen model in a pickle file so that it is easier to handle later on
with open('model/xgb_model.pkl', "wb") as f:
    pickle.dump(xgb_model, f)


### Threshold computation
# After looking at the accuracy and the other metrics we can see that, same as had previously measured in the EDA, this is a very unbalanced dataset. Almost 80% of the values belong to 0, while only 20% is part of the positive results.

# There is a much higher number of 0s (no matches) than 1s (matches), there this makes the precision, accuracy and recall of the matches much more lower than the no matches.

# We need to find a weight balance in order to work with it and in return choose the most appropriate model to work with.
# References found in: [https://stackoverflow.com/questions/51190809/high-auc-but-bad-predictions-with-imbalanced-data]

# We might also want to look into the performance of a =XGBoost= model since it performs better for unbalanced data. 
# The boosting algorithm iteratively learns from the mistakes of the previous tree. So if a tree fails to predict a class (most likely the imbalanced class), the proceeding tree will give more weightage to this sample.

def pr_threshold(model, model_name='Model'):
    """
    Calculate and plot the precision-recall curve for a given model and test data.
    Also, find the threshold that yields the highest F1 score using an iterative method and plot the classification report based on this threshold.

    Returns:
    optimal_threshold: The threshold value that yields the highest F1 score.
    optimal_report: Classification report using the optimal threshold.
    """
    # probabilities for the positive class
    probabilities = model.predict_proba(X_val)[:,1]

    # precision, recall, and thresholds using the predicted probabilities
    precision, recall, thresholds = precision_recall_curve(y_val, probabilities)

    # F1 score for each threshold with tiny constant to avoid division by zero
    fscore = (2 * precision * recall) / (precision + recall + 1e-12)

    # Find the index of the maximum F1 score
    optimal_index = np.argmax(fscore)
    best_threshold = thresholds[optimal_index]

    # Make final predictions using the optimal threshold
    final_predictions = (model.predict_proba(X_test)[:,1] >= best_threshold).astype(int)

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, precision[:-1], 'b--', label='Precision')
    plt.plot(thresholds, recall[:-1], 'g-', label='Recall')
    plt.plot(thresholds, (2 * precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-12), 'r-.', label='F1 Score')
    plt.axvline(x=best_threshold, color='k', linestyle='--', label='Optimal Threshold')
    plt.xlabel('Threshold')
    plt.legend(loc='best')
    plt.ylim([0, 1])
    plt.title(f'{model_name}: Precision, Recall, and F1 Score for different thresholds')
    # plt.show()

    optimal_report = classification_report(y_test, final_predictions, zero_division=0)
    return best_threshold, optimal_report

## Random Forest Threshold
optimal_threshold_rf, optimal_report_rf = pr_threshold(rf_model, model_name='Random Forest Classifier')

# print("Optimal threshold for maximum Random Forest F1-score:", optimal_threshold_rf)
# print(f"Using threshold ~{optimal_threshold_rf:.4f} for evaluation.\n")
# print(optimal_report_rf, "\n")

final_predictions_rf = (rf_model.predict_proba(X_test)[:,1] >= optimal_threshold_rf).astype(int)
# print("Random Forest: f1 score", f1_score(final_predictions_rf, y_test, average="weighted"))

## LGBClassifier Threshold
optimal_threshold_lgb, optimal_report_lgb = pr_threshold(lgb_model, model_name='LightGBM')

# print("Optimal threshold for maximum LightGBM F1-score:", optimal_threshold_lgb)
# print(f"Using threshold ~{optimal_threshold_lgb:.4f} for evaluation.\n")
# print(optimal_report_lgb)

final_predictions_lgb = (lgb_model.predict_proba(X_test)[:,1] >= optimal_threshold_lgb).astype(int)
# print("LightGBM: F1 score", f1_score(final_predictions_lgb, y_test, average="weighted"))

## XGBClassifier Threshold
optimal_threshold_xgb, optimal_report_xgb = pr_threshold(xgb_model, model_name='XGBClassifier')

# print("Optimal threshold for maximum XGBClassifier F1-score:", optimal_threshold_xgb)
# print(f"Using threshold ~{optimal_threshold_xgb:.4f} for evaluation.\n")
# print(optimal_report_xgb)

final_predictions_xgb = (xgb_model.predict_proba(X_test)[:,1] >= optimal_threshold_xgb).astype(int)
# print("XGBClassifier: F1 score", f1_score(final_predictions_xgb, y_test, average="weighted"))

### Confusion Matrix for XGBoost
conf_matrix = confusion_matrix(y_test, final_predictions_xgb)
# print("Confusion Matrix:\n", conf_matrix)

fig, ax = plt.subplots(figsize=(8, 6))
ConfusionMatrixDisplay(confusion_matrix=conf_matrix).plot(ax=ax)
plt.title('Confusion Matrix for XGBClassifier')
#plt.show()


### SHAP Values Computation
explainer = shap.Explainer(xgb_model, X_train)
shap_values_xgb = explainer.shap_values(X_test)

# Visualize the SHAP values
# shap.summary_plot(shap_values_xgb, X_test)

def shap_importance(model, shap_values):
    """
    Return a dataframe containing the features sorted by Shap importance.

    Parameters:
    model : The tree-based model (like RandomForest, XGBoost, etc.).

    Returns:
    pd.DataFrame
        A dataframe containing the features sorted by Shap importance.
    """
    # explainer = shap.Explainer(model, X_train)
    
    # shap_values = explainer.shap_values(X_test, check_additivity=False)
    
    if isinstance(shap_values, list):
        shap_values = np.abs(shap_values[1])
    else:
        shap_values = np.abs(shap_values)
        
    mean_abs_shap_values = shap_values.mean(axis=0)
    
    feature_importance = pd.DataFrame({
        'features': X_test.columns,
        'importance': mean_abs_shap_values
    })
    
    feature_importance.sort_values(by='importance', ascending=False, inplace=True)
    return feature_importance.reset_index(drop=True).head(30)

top_features = shap_importance(xgb_model, shap_values_xgb).drop([22, 24, 0, 1])
top_features.to_csv('../data/shap_top_features.csv')
shap_columns = top_features['features'].to_list()


### Cosine Similarity
def cosine_pairs_similarity(df, pairs, features):
    """
    Calculate cosine similarity for specified pairs in the DataFrame.

    Arguments:
    - df (DataFrame): The DataFrame containing user features.
    - pairs (list of lists): A list containing sublists, each sublist is a pair of IDs [iid, pid].
    - features (list): A list of column names to use as features for calculating similarity.

    Returns:
    - DataFrame: A DataFrame with the pair identifiers and their similarity scores.
    """
    results = []

    # Standardize the features
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])

    # Process each pair
    for pair in pairs:
        id1, id2 = pair
        # Extract features for both ids in the pair
        person1_features = df.loc[df['iid'] == id1, features].values
        person2_features = df.loc[df['iid'] == id2, features].values

        # Calculate cosine similarity
        if len(person1_features) > 0 and len(person2_features) > 0:  # Ensure both persons are in the dataset
            similarity_score = cosine_similarity(person1_features, person2_features)[0][0]
            results.append([id1, id2, similarity_score])

    # Convert results to DataFrame
    similarity_df = pd.DataFrame(results, columns=['iid', 'pid', 'similarity_score'])

    return similarity_df

features = ['imprelig', 'goal', 'date', 'go_out', 'sports', 'tvsports', 'exercise',
            'dining', 'museums', 'art', 'hiking', 'gaming', 'clubbing', 'reading',
            'tv', 'theater', 'movies', 'concerts', 'music', 'shopping', 'yoga',
            'exphappy', 'attr_important', 'sinc_important', 'intel_important',
            'fun_important', 'amb_important', 'shar_important', 'like', 'like_o']

pairs = [[1.0, 14.0], [1.0, 15.0], [1.0, 19.0], [2.0, 14.0], [2.0, 19.0], [4.0, 14.0], [4.0, 19.0], [5.0, 14.0], [5.0, 19.0], [6.0, 14.0], [6.0, 19.0], [7.0, 14.0], [7.0, 19.0], [8.0, 12.0], [8.0, 13.0], [8.0, 14.0], [8.0, 16.0], [8.0, 18.0], [8.0, 19.0], [8.0, 20.0], [9.0, 12.0], [9.0, 13.0], [9.0, 14.0], [9.0, 15.0], [9.0, 16.0], [9.0, 17.0], [9.0, 19.0], [10.0, 19.0], [12.0, 8.0], [12.0, 9.0], [13.0, 8.0], [13.0, 9.0], [14.0, 1.0], [14.0, 2.0], [14.0, 4.0], [14.0, 5.0], [14.0, 6.0], [14.0, 7.0], [14.0, 8.0], [14.0, 9.0], [15.0, 1.0], [15.0, 9.0], [16.0, 8.0], [16.0, 9.0], [17.0, 9.0], [18.0, 8.0], [19.0, 1.0], [19.0, 2.0], [19.0, 4.0], [19.0, 5.0], [19.0, 6.0], [19.0, 7.0], [19.0, 8.0], [19.0, 9.0], [19.0, 10.0], [20.0, 8.0]]
similarity_results = cosine_pairs_similarity(df_wave, pairs, features)

def percentage_positive_negative_values(df, column_name):
    
    total_count = len(df[column_name])
    positive_count = (df[column_name] > 0).sum()
    negative_count = (df[column_name] < 0).sum()
    
    positive_percentage = (positive_count / total_count) * 100
    negative_percentage = (negative_count / total_count) * 100
    
    result_df = pd.DataFrame({
        'column': [column_name],
        'positive_percentage': [positive_percentage],
        'negative_percentage': [negative_percentage]
    })
    
    return result_df
    
percentage_negative = percentage_positive_negative_values(similarity_results, 'similarity_score')


### Grid Search for XGBoost Models (Option for Future Works) ###

# Made two models with XGBoost:
# - Classifier: to predict the matches once more so that these predictions can be added as a key component when creating and ranking the potential parter prederences lists

# - Regressor: as a different way to find the preferences lists. This way the 'liking' (column 'like') prediction of each individual can give us their rank preference of everyone they went on a date with. Since some of them did not have dates with everyone or the information was missing, the rest of the subjects would be added in an ordered way.

# X_reg = df.drop(columns=['like', 'like_o', 'match'], axis=1) 
# y_reg = df_final['like']

# X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
# #X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

# # Define a K-Fold cross-validator
# kf = KFold(n_splits=3, shuffle=True, random_state=42) 

# param_grid_regressor = {
#     'learning_rate': [0.01, 0.1, 0.2],
#     'subsample': [0.8, 0.5, 0.1],
#     'n_estimators': [100,500,1000],
#     'max_depth': [3, 6 ,10],
#     'colsample_bytree': [0.8, 1]
    
# }
# xgb_regressor = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

# grid_search_xgb_regressor = GridSearchCV(estimator=xgb_regressor, param_grid=param_grid_regressor, cv=kf, verbose=2, n_jobs=-1)
# grid_search_xgb_regressor.fit(X_reg_train, y_reg_train)

# print("Best parameters found xgb_regressor:\n", grid_search_xgb_regressor.best_params_)
# print("\n The best score across ALL searched params for xgb_regressor:",grid_search_xgb_regressor.best_score_)

# Prediction on best Regressor model
# cv_scores = cross_val_score(grid_search_xgb_regressor, X_reg_train, y_reg_train, cv=kf, scoring='neg_mean_squared_error')

# best_model = grid_search_xgb_regressor.best_estimator_
# y_pred = best_model.predict(X_reg_test)

# results = pd.DataFrame({
#     # 'iid': df_final['iid'],
#     # 'pid': df_final['pid'].astype(int),
#     'predicted_like': y_pred,
#     'actual_like': y_reg_test
# })

# r2_score(y_reg_test, y_pred)

# def generate_combinations_predictions(list_a, list_b, preferences):
#     n = len(list_a)
#     combinations = []
    
#     current_a = list_a[:]
#     current_b = list_b[:]
    
#     for i in range(n + 1):
#         # Filter preferences for each list against the opposing list and add missing ids
#         filtered_preferences_a = {
#             a: [iid for iid in preferences.get(a, []) if iid in current_b] + [b for b in current_b if b not in preferences.get(a, [])]
#             for a in current_a
#         }
#         filtered_preferences_b = {
#             b: [iid for iid in preferences.get(b, []) if iid in current_a] + [a for a in current_a if a not in preferences.get(b, [])]
#             for b in current_b
#         }

#         # Store the combination before the rotation to avoid order alteration
#         combination_dict = {
#             'Group_A': current_a[:],
#             'Group_B': current_b[:],
#             'Preferences_A': filtered_preferences_a,
#             'Preferences_B': filtered_preferences_b
#         }
        
#         # Rotate elements between the lists
#         # Move last element of A to the front of B and vice versa
#         element_a = current_a.pop(-1)
#         element_b = current_b.pop(-1)
#         current_a.insert(0, element_b)
#         current_b.insert(0, element_a)
        
#         combinations.append(combination_dict)

#     return combinations

