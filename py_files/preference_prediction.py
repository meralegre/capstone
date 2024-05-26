#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
import pickle

import shap
from collections import defaultdict

from processing_functions import *
from gale_shapley import *
pd.set_option('display.max_colwidth', None)

df_final = pd.read_csv('../data/df_final.csv')
df_final = df_final.drop(columns=['Unnamed: 0'])

xgb_model = pickle.load(open('../model/xgb_model.pkl', 'rb'))

### Chosen wave data preparation
# print(df_final['iid'].unique())
df_final[df_final['wave'] == 1]

#### Matches per wave
# def wave_matches_display(df):
#     # Dictionary to store data for each wave
#     wave_data = {}

#     # Group by wave and process each group
#     for wave, group in df.groupby('wave'):
#         # Collect all participants in the wave
#         participants = list(group['iid'].unique())
        
#         # Collect matches as lists of pairs
#         matches = []
#         for index, person in group.iterrows():
#             if person['match'] == 1 and person['pid'] in group['iid'].values:
#                 matches.append([person['iid'], person['pid']])
        
#         # Storing results in the dictionary with wave as key
#         wave_data[wave] = {
#             "participants": participants,
#             "matches": matches
#         }

#     # Convert the dictionary into a DataFrame
#     wave_df = pd.DataFrame([
#         {'wave': wave, 'participants': data['participants'], 'matches': data['matches']}
#         for wave, data in wave_data.items()
#     ])

#     return wave_df

wave_df = wave_matches_display(df_final)

#### Preparation of first wave dataframe
def indices_column(df, value):
    filtered_df = df[df['wave'] == value]
    indices = filtered_df.index.tolist()
    return indices

# select rows from another dataframe using the obtained indices
def select_rows(df, indices):
    return df.iloc[indices]

indices = indices_column(df_final, 1)
df_wave = select_rows(df_final, indices)
df_wave = df_wave.drop(columns=['id', 'partner', 'wave', 'expnum'], axis=1)


#### Match predictions for dataset
# The predictions have already been done in [modelling_preferences.ipynb](modelling_preferences.ipynb).

match_results = pd.read_csv('../data/match_results.csv').drop(columns=['Unnamed: 0'])
df_wave['match_predictions'] = match_results['predicted_match']
# match_results[['iid', 'pid', 'predicted_match', 'actual_match']].iloc[20:40]

### Preparing distinct dataframes for Preference Lists
attributes = [
    'iid', 
    'attr_o', 
    'sinc_o',
    'intel_o', 
    'fun_o', 
    'amb_o', 
    'shar_o',    
    # 'attr_self', 
    # 'sinc_self', 
    # 'intel_self', 
    # 'fun_self', 
    # 'amb_self'
]

attributes_averages = df_wave[attributes].groupby('iid').mean().reset_index()
# print(attributes_averages)

top_features = pd.read_csv('../data/shap_top_features.csv').drop(columns=['Unnamed: 0'])
shap_columns = top_features['features'].to_list()

shap_features = df_wave[shap_columns]
# print(shap_features)

# def goal_mapping(df):
#     # First, create a mapping dictionary for the 'goal' values
#     mapping = {
#         1: 1, 2: 1,
#         3: 2, 4: 2,
#         5: 3, 6: 3
#     }

#     # Aggregate the DataFrame by 'iid' and take the first 'goal' from each group
#     aggregated_df = df.groupby('iid').agg({'goal': 'first'}).reset_index()

#     # Apply the mapping to the 'goal' column and create a new column for remapped values
#     aggregated_df['goal'] = aggregated_df['goal'].map(mapping).fillna(aggregated_df['goal'])

#     return aggregated_df

goals = goal_mapping(df_wave)
attributes_averages = pd.concat([attributes_averages, goals['goal']], axis=1)
# attributes_averages

importance = df_wave[[
    'iid',
    'attr_important', 
    'sinc_important', 
    'intel_important',
    'fun_important', 
    'amb_important', 
    'shar_important'
]]

importance = importance.groupby('iid').mean().round().astype(int)
importance = importance.apply(lambda x: np.clip(x, 1, 6))

# def resolve_conflicts(row):
#     # Sort row values and get indices of the sorted elements
#     sorted_indices = np.argsort(row)
#     ranks = np.arange(1, 7)

#     # Assign ranks according to the sorted indices (smallest value gets rank 1, largest gets rank N)
#     sorted_row = np.empty_like(ranks)
#     sorted_row[sorted_indices] = ranks

#     return pd.Series(sorted_row, index=row.index)

importance_attributes = importance.apply(resolve_conflicts, axis=1).reset_index()
importance_attributes = pd.concat([importance_attributes, goals['goal']], axis=1)
# importance_attributes

date_rankings = df_wave[['iid', 'pid', 'attr']]
shared = df_wave[[
    'iid',
    'sports', 
    'tvsports', 
    'exercise', 
    'dining', 
    'museums', 
    'art', 
    'hiking', 
    'gaming', 
    'clubbing', 
    'reading', 
    'tv', 
    'theater', 
    'movies', 
    'concerts', 
    'music', 
    'shopping', 
    'yoga'
]].drop_duplicates()

### Preferences Lists for each subject
#iid = 4
iids = range(1, 21)
for iid in iids:
    preference_list = individual_preference(iid, df_wave, attributes_averages, importance_attributes, shared, date_rankings)
    # print(f"Preference list for iid {iid}:\n{preference_list}\n")


#### Function to generate all the possible combinations
def generate_combinations(list_a, list_b, preferences):
    n = len(list_a)
    combinations = []
    
    current_a = list_a[:]
    current_b = list_b[:]
    
    for i in range(n + 1):
        # filter preferences for each list against the opposing list
        filtered_preferences_a = {
            a: [iid for iid in preferences[a] if iid in current_b] for a in current_a
        }
        filtered_preferences_b = {
            b: [iid for iid in preferences[b] if iid in current_a] for b in current_b
        }

        # store the combination before the rotation to avoid order alteration
        combination_dict = {
            'Group_A': current_a[:],
            'Group_B': current_b[:],
            'Preferences_A': filtered_preferences_a,
            'Preferences_B': filtered_preferences_b
        }
        
        # Rotate elements between the lists
        # Move last element of A to the front of B and vice versa
        element_a = current_a.pop(-1)
        element_b = current_b.pop(-1)
        current_a.insert(0, element_b)
        current_b.insert(0, element_a)
        
        combinations.append(combination_dict)

    return combinations

list_a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
list_b = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

iids = range(1, 21)
preferences = {} 
for iid in iids:
    preference_list = individual_preference(iid, df_wave, attributes_averages, importance_attributes, shared, date_rankings)
    preferences[iid] = preference_list

preference_combinations = generate_combinations(list_a, list_b, preferences)
# preference_combinations
# for index, combo in enumerate(preference_combinations):
#     print(f"Rotation {index + 1}:")
#     print("Group A:", combo['Group_A'])
#     print("Preferences of Group A:", combo['Preferences_A'])
#     print("Group B:", combo['Group_B'])
#     print("Preferences of Group B:", combo['Preferences_B'])
#     print("\n")


### Gale-Shapley Algorithm
def combinations_matchings(preferences):
    all_engagements = []
    
    for round_number, combination in enumerate(preferences):
        group_a = combination['Group_A']
        group_b = combination['Group_B']
        preferences_a = combination['Preferences_A']
        preferences_b = combination['Preferences_B']

        print(f"Processing Round {round_number + 1}:")

        # Apply the stable matching algorithm
        engagements = stable_matching(group_a, group_b, preferences_a, preferences_b)
        all_engagements.append(engagements)

        # Output the results of the matching
        print(f"Engagements for Round {round_number + 1}: {engagements}")
        print()

    return all_engagements

# Attribute importance
all_engagements_attributes = combinations_matchings(preference_combinations)
# all_engagements_attributes

def calculate_match_counts(all_engagements):
    """
    Calculates and prints the number of times each individual has matched with others.
    """
    match_counts = defaultdict(lambda: defaultdict(int))

    for engagements in all_engagements:
        for k, v in engagements.items():
            match_counts[k][v] += 1

    for iid, partners in match_counts.items():
        print(f"{iid} matched with:")
        for partner, count in partners.items():
            print(f"  {partner} -> {count} times")
        print()

print(calculate_match_counts(all_engagements_attributes))