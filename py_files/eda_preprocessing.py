#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from processing_functions import *

df = pd.read_csv('../data/SpeedDatingData.csv', encoding='latin1')

# Let's start by changing the empty spaces by nan values to have a clear image of the dataset. 
# Since most of these columns have '?' as the nan value, they  have mismatched datatypes, 
# being objects where they should be ints or floats.

# Therefore I decided to create a function to change the column to its corresponding datatype.
df.applymap(convert_to_numeric).replace('?', np.nan, inplace=True)

# Check percentage of nan values per column. # To make life easier I decided to define a function to return the corresponding percentage for each column.

# During the experiment they devided the speed dating events in three steps. # First step consisted on asking the participants questions based on their initial thoughts on the selected attributes, how they perceive themselves, how they perceive the others, what is the most important attribute, etc.# Since I will be only be taking into account first impressions (simulating a dating app) I will remove the data observed after the dates themselves.

filtered_columns = [
    col for col in df.columns if not col.endswith('_2') 
    and not col.endswith('_3')
    and not col.endswith('_s')
]
df_filtered = df[filtered_columns]

# Now we have in total 111 columns that center more around the actual dates
# I am going to rename the columns associated to rankings to have a clear 
# understanding of which is which isntead of having it associated with numbers

rename_suffix = [
    ('1_1', '_important'),
    ('2_1', '_o_want'),
    ('3_1', '_self'),
    ('4_1', '_fellow_want'),
    ('5_1', '_perceived')
]

df_renamed = rename_columns(df_filtered, rename_suffix)

# Since 'field' has too many unique values, we'll drop it. It is not that important either for this research
# Each participant provided information about their field that was later encoded with a number.
# Let's check which fields we have (we will not be using this since they are not relevant for this study,
# but it is still useful to know what we are working with.

fields = df_renamed.groupby('field_cd').agg({
    'field': lambda x: ','.join(x.drop_duplicates().astype(str)),
    'iid': 'nunique'
}).reset_index()

fields.columns = ['Field encoded', 'Fields', 'Participants count']
df_renamed.drop(['field'], axis=1)

# In some of the waves they implement a different way of rating by attributes:
# Waves 6-9: rate the importance of the attributes in a potential date on a scale of 1-10 (1 = not at all important, 10 = extremely important):
# Waves 1-5, 10-21: 100 point saro distribtdde among teg attributse,  more point sare given  to those attributes that are more important in a potential date, and fewer points to those attributes that are less important in a potential date.  Total points must equal 100.

# In order to have more objective ratings, I decided to design a function that would change the 100 points ranking system for the importance rating in a scale from 1 to 6.
def attribute_ranking(row, attribute_groups):
    """
    Ranks attribute points directly in the original columns based on their importance, with tie-breaking.
    Fills NaN values with 0 before ranking and applies a small random noise to break ties.
    
    Parameters:
    - row: A single row (Series) from DataFrame.
    - attribute_columns: List of columns containing attribute points to be ranked.
    - wave_column: The column indicating the wave to check for specific ranking rules.
    
    Returns:
    - A modified row with uniquely ranked attributes.
    """
    for attribute_columns in attribute_groups:
        row[attribute_columns] = row[attribute_columns].fillna(0)
        
        # Add a small random noise to break ties
        noise = np.random.normal(0, 0.01, len(attribute_columns))
        noisy_attributes = row[attribute_columns] + noise
        
        ranks = noisy_attributes.rank(method='first', ascending=False).astype(int)
    
        row[attribute_columns] = ranks

    return row

attribute_columns = [
    'attr_important',
    'sinc_important',
    'intel_important',
    'fun_important',
    'amb_important',
    'shar_important',
    'attr_fellow_want',
    'sinc_fellow_want',
    'intel_fellow_want',
    'fun_fellow_want',
    'amb_fellow_want',
    'shar_fellow_want',
    'attr_o_want',
    'sinc_o_want',
    'intel_o_want',
    'fun_o_want',
    'amb_o_want',
    'shar_o_want',
    'attr_perceived', 
    'sinc_perceived', 
    'intel_perceived', 
    'fun_perceived', 
    'amb_perceived',
]

attribute_groups = [
    ['attr_important', 'sinc_important', 'intel_important', 'fun_important', 'amb_important', 'shar_important'],
    ['attr_fellow_want', 'sinc_fellow_want', 'intel_fellow_want', 'fun_fellow_want', 'amb_fellow_want', 'shar_fellow_want'],
    ['attr_o_want', 'sinc_o_want', 'intel_o_want', 'fun_o_want', 'amb_o_want', 'shar_o_want'],
    ['attr_perceived', 'sinc_perceived', 'intel_perceived', 'fun_perceived', 'amb_perceived']
]

df = df_renamed.apply(lambda row: attribute_ranking(row, attribute_groups), axis=1)


## PLOTS ##

## Gender
gender_counts = df['gender'].value_counts()

# pie chart
plt.figure(figsize=(8, 8))
plt.pie(gender_counts, labels=['Male', 'Female'], autopct='%.2f%%', startangle=140)
plt.title('Proportion of Gender')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
#plt.show()


## Age Distribution
plt.figure(figsize=(10, 6))
plt.hist(df['age'].dropna(), bins=20, color='skyblue', edgecolor='black')
plt.title('Age Distribution of Participants')
plt.xlabel('Age')
plt.ylabel('Count')
plt.grid(axis='y', alpha=0.75)
#plt.show()

# For a clearer and narrower understanding of the age values:
age_dist = age_distribution(df_renamed)

## Plotting the data percentage distribution of population by age range
plt.figure(figsize=(10, 6))
sns.barplot(x='Age Range', y='Percentage', data=age_dist, palette='mako')
plt.title('Percentage Distribution of Population by Age Range')
plt.xlabel('Age Range')
plt.ylabel('Percentage of population')
plt.grid(axis='y', alpha=0.75)
#plt.show()

## Attributes importance ranking per gender
df_ranked = df.copy()
df_ranked['gender'] = df_ranked['gender'].map({'Female': 0, 'Male': 1})
actual_attribute_names = ['attr_important', 'sinc_important', 'intel_important', 'fun_important', 'amb_important', 'shar_important']
df_melted = df.melt(id_vars=['gender'], value_vars=actual_attribute_names, var_name='attribute', value_name='rank')
rank_counts = df_melted.groupby(['gender', 'attribute', 'rank']).size().unstack(fill_value=0)

rank_labels = [1, 2, 3, 4, 5, 6]
cols = ['Attractive Importance', 'Sincerity Importance', 'Intelligence Importance', 'Fun Importance', 'Ambition Importance', 'Shared Interests Importance']

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 10))
axes = axes.flatten()
bar_width = 0.4

for i, actual_name in enumerate(actual_attribute_names):
    ax = axes[i]
    
    # Male data for this attribute
    male_data = rank_counts.loc[1, actual_name].fillna(0)
    ax.bar(np.arange(6) - bar_width/2, male_data, bar_width, label='Men', color='darkblue')
    
    # Female data for this attribute
    female_data = rank_counts.loc[0, actual_name].fillna(0)
    ax.bar(np.arange(6) + bar_width/2, female_data, bar_width, label='Women', color='lightblue')

    ax.set_title(cols[i])  # Using the human-readable names from `cols`
    ax.set_xlabel('Rank')
    ax.set_ylabel('Count')
    ax.set_xticks(np.arange(6))
    ax.set_xticklabels(rank_labels)
    ax.legend()

plt.tight_layout()
#plt.show()


## Distribution of 'like' ratings by match outcome
plt.figure(figsize=(10, 6))
palette = {0: "teal", 1: "lightblue"}
box = sns.boxplot(x='match', y='like', data=df, palette=palette)
plt.title("Distribution of 'Like' Ratings by Match Outcome")
plt.xlabel('Match Outcome')
plt.ylabel('Like Rating')
plt.grid(True)

# Adding custom tick labels
box.set_xticklabels(['No Match', 'Match'])

# Adding text annotations inside the plot
medians = df.groupby(['match'])['like'].median().values
nobs = df['match'].value_counts().values
pos = range(len(nobs))
for tick, label in zip(pos, box.get_xticklabels()):
    box.text(pos[tick], medians[tick] + 0.05, 'n: {}'.format(nobs[tick]),
             horizontalalignment='center', size='x-small', color='black', weight='semibold')

#plt.show()


## Density of 'like' ratings by match outcome (other representation of the previous plot)
plt.figure(figsize=(10, 6))
# Plot density for each match outcome
sns.kdeplot(data=df, x='like', hue='match', fill=True, common_norm=False, palette={0: "teal", 1: "darkblue"})
# Set title and labels
plt.title("Density of 'Like' Ratings by Match Outcome")
plt.xlabel("Like Rating")
plt.ylabel("Density")
#plt.show()
# Participants who ended up matching (1) generally gave higher 'like' ratings than those who didn't match (0). The 'like' ratings for matches are more tightly clustered, as indicated by the narrower box and lack of outliers, suggesting a consensus on the likability of matches. In contrast, 'like' ratings for no matches are more spread out, showing that participants' opinions varied more widely when they did not find a match.


## Overall importance of dates' attributes (counting both genders)
preferences = ['pf_o_att', 'pf_o_sin', 'pf_o_int', 'pf_o_fun', 'pf_o_amb', 'pf_o_sha']
avg_preferences = df[preferences].mean()

labels = {
    'pf_o_att': 'Attractiveness',
    'pf_o_sin': 'Sincerity',
    'pf_o_int': 'Intelligence',
    'pf_o_fun': 'Fun',
    'pf_o_amb': 'Ambition',
    'pf_o_sha': 'Shared Interests'
}

avg_preferences.rename(index=labels, inplace=True)
avg_preferences = avg_preferences.reset_index()
avg_preferences.columns = ['attribute', 'average_rating']

with plt.style.context('seaborn-white'):
    plt.figure(figsize=(10, 6))
    sns.barplot(x='attribute', y='average_rating', data=avg_preferences, palette='mako')
    plt.title("Importance of the date's attributes")
    plt.xlabel('Attribute')
    plt.ylabel('Average Rating')
    plt.xticks(rotation=0)
    plt.grid(axis='y', alpha=0.75)
    #plt.show()


## Religion Importance
# This will show how participants rated the importance of religion, 
# I think this is an important factor when it comes to matching 
df_sorted = df.sort_values(by=['iid'])
df_unique = df_sorted.drop_duplicates(subset='iid').dropna(subset=['imprelig'])

with plt.style.context('seaborn-white'):
    plt.figure(figsize=(10, 6))
    total_count = len(df_unique)
    ax = sns.histplot(df_unique['imprelig'], bins=10, edgecolor='black', kde=False, stat='percent')
    ax.set(title='Importance of Religion',
           xlabel='Rating (out of 10)',
           ylabel='Percentage')
    plt.grid(axis='y', linestyle='-', alpha=0.75)

    # bin centers and percentages
    counts, bin_edges = np.histogram(df_unique['imprelig'], bins=10)
    percentages = (counts / total_count) * 100
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    plt.xticks(bin_centers, labels=[f"{int(center)}" for center in np.round(bin_centers)])

    # percentage of people above each bar
    for percentage, center in zip(percentages, bin_centers):
        ax.text(center, percentage, f'{percentage:.1f}%', ha='center', va='bottom')

    #plt.show()

## Goals in participating
# Illustrating participants' goals in joining the speed dating event.
goals = df['goal'].value_counts(normalize=True).mul(100)
labels = {
    1: 'Fun Night Out',
    2: 'Meet New People',
    3: 'Get a Date',
    4: 'Serious Relationship',
    5: 'To say I did it',
    6: 'Other'
}

goals.rename(index=labels, inplace=True)
goals = goals.reset_index()
goals.columns = ['goal', 'percentage']

with plt.style.context('seaborn-white'):
    plt.figure(figsize=(10, 6))
    sns.barplot(x='goal', y='percentage', data=goals, palette='mako')
    plt.title('Goals in Participating')
    plt.xlabel('Goal')
    plt.ylabel('Percentage of Participants')
    plt.xticks(rotation=0)
    plt.grid(axis='y', alpha=0.75)
    #plt.show()


## Dating frequency of the participants
df['date'] = pd.Categorical(df['date'], categories=[1, 2, 3, 4, 5, 6, 7], ordered=True)

date_frequency = df['date'].value_counts(normalize=True, sort=False).mul(100).reset_index()
date_frequency.columns = ['date', 'percentage']

with plt.style.context('seaborn-white'):
    plt.figure(figsize=(10, 6))
    sns.barplot(x='date', y='percentage', data=date_frequency, palette='mako')
    plt.title('How Often Participants Date')
    plt.xlabel('Frequency')
    plt.ylabel('Percentage')
    plt.xticks(rotation=0)
    plt.grid(axis='y', alpha=0.75)
    #plt.show()

## Heatmap to portray the relationship between religion importance and dating habits
data_crosstab = pd.crosstab(df['date'], df['imprelig'], normalize='index')

plt.figure(figsize=(10, 8))
sns.heatmap(data_crosstab, annot=True, cmap='coolwarm')
plt.title('Heatmap of Date vs. Importance of Religion')
plt.xlabel('Importance of Religion')
plt.ylabel('Frequency of Going on Dates')
#plt.show()


## Representation of the attribute ranking each person got and their partner's decision at the end of the date.
# 1 means their partner wants a second date (match)
# 0 means thir partner does not want a second date (no match)
attributes = ['attr','intel','fun','sinc','amb', 'shar']

with plt.style.context('seaborn-white'):
    fig, axes = plt.subplots(2, 3, sharey=True, figsize=(14, 8))
    fig.text(0.02, 0.5, 'Density', va='center', rotation='vertical')
    fig.suptitle('Partner Attribute Distributions by Decision')

    # Iterate over the attributes and plot on the appropriate subplot
    for n, attribute in enumerate(attributes):
        row, col = divmod(n, 3)  # Calculate row and column index
        ax = axes[row, col]  # Select the correct subplot
        sns.histplot(df, x=attribute, ax=ax, discrete=True, hue='dec', multiple='dodge')
        #ax.set_ylabel(' ')
        ax.set_title(attribute)

    plt.tight_layout()
    #plt.show()


## Does age difference have an impact when matching with another subject?
df['age_diff'] = (df['age'] - df['age_o']).abs()

df['age_diff'] = pd.cut(df['age_diff'], bins=np.arange(0, 26, 5), right=False, labels=[f'{i}-{i+4}' for i in range(0, 25, 5)])
palette

plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='age_diff', hue='match', palette={0: "teal", 1: "darkblue"})
plt.title('Impact of Age Difference on Match')
plt.xlabel('Age Difference (5 - years)')
plt.ylabel('Count of Decisions')
plt.xticks(rotation=0)
plt.legend(title='Match', labels=['No', 'Yes'])
#plt.show()


## Percentage of Matches vs. No Matches
match = df['match'].value_counts(normalize=True).mul(100).reset_index()
match.columns = ['match', 'percentage']

with plt.style.context('seaborn-white'):
    plt.figure(figsize=(6, 4))
    sns.barplot(x='match', y='percentage', data=match, palette='mako')
    plt.title('Percentage of Matches vs. No Matches')
    plt.xlabel('Match (1) vs No Match (0)')
    plt.ylabel('Frequency')
    plt.xticks(rotation=0)
    plt.grid(axis='y', alpha=0.75)
    #plt.show()


## Number of likes men give vs number of likes women give
df_renamed['gender'] = df_renamed['gender'].map({0: 'Women', 1: 'Men'})
match_df = df_renamed.groupby('gender').agg(likes=('dec', 'sum'), matches=('match', 'sum')).reset_index()
match_df = match_df.melt(id_vars='gender', var_name='Match', value_name='Count')

plt.figure(figsize=(10, 6))
sns.barplot(x='gender', y='Count', hue='Match', data=match_df, palette='mako')
plt.title('Likes vs Matches by Gender')
plt.xlabel('Gender')
plt.ylabel('Total Count')
plt.legend(title='Outcome')
#plt.show()


# Based on the description of each column it will be more useful to drop some columns that will not be necessary for our experiment:
df_final = df.drop([
    'gender', 
    'idg', 
    'round',
    'condtn',
    'from',
    'position', 
    'positin1', 
    'order', 
    'samerace',
    'race_o',
    'field',
    'field_cd', 
    'undergra',
    'mn_sat',
    'tuition',
    'race',
    'imprace',
    'zipcode',
    'income',
    'career',
    'career_c',
    'length',
    'met',
    'met_o',
    'match_es',
    'you_call',
    'them_cal',
], axis=1)

df_final.to_csv('../data/input_file.csv')