import pandas as pd
import numpy as np

def convert_to_numeric(obj):
    """
    Convert object to an int, or float, or returns the object (string)
    if neither is possible.
    """
    try:
        # Attempt to convert directly to int, then float if ValueError is raised
        return int(obj)
    except ValueError:
        try:
            return float(obj)
        except ValueError:
            return obj

def nan_percentage(df: pd.DataFrame):
    """
    Returns the percentage of nan values per column in a dataframe
    """
    for col in df.columns:
        print("NaN % {}: {:0.2f}%".format(col, df[col].isna().mean() * 100))
        

def rename_columns(df, rename_suffix):
    """
    Modifies column names of a DataFrame based on specified renaming rules.
    Each rule in the `rename_suffix` specifies a suffix to look for and a 
    suffix to append.
    """
    new_columns = df.columns.tolist()
    for i, col in enumerate(df.columns):
        for suffix_change, suffix_add in rename_suffix:
            if col.endswith(suffix_change):
                new_columns[i] = col.replace(suffix_change, '') + suffix_add
                #print(new_columns)
    df.columns = new_columns
    return df

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
        
        # Rank the attributes, lower number means more important
        ranks = noisy_attributes.rank(method='first', ascending=False).astype(int)  # 'first' to handle ties by order of appearance
    
        row[attribute_columns] = ranks

    return row

def age_distribution(df):
    """
    Calculate the percentage of people's ages within 5-year gaps.

    Parameters:
    - df: Pandas DataFrame containing the age data.
    - age_column: The name of the column in the DataFrame that contains age data.

    Returns:
    - DataFrame with corresponding percentage of people within age range.
    """
    df['age'] = df['age'].dropna()
    
    bins = [0, 20, 25, 30, 35, 40, 45, 50, 60] 
    labels = ['<20', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49', '>50']

    # Use pandas cut to categorize each age into bins
    binned_ages = pd.cut(df['age'], bins=bins, labels=labels, right=False, include_lowest=True)

    age_distribution = binned_ages.value_counts(normalize=True).sort_index() * 100

    age_distribution_df = age_distribution.reset_index()
    age_distribution_df.columns = ['Age Range', 'Percentage']

    return age_distribution_df.round(2)

def drop_columns(df, suffix):
    cols = [col for col in df.columns if any(col.endswith(s) for s in suffix)]
    
    df = df.drop(columns=cols, errors='ignore')
    
    return df

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
    
def pr_threshold(model, model_name='Model'):
    """
    Calculate and plot the precision-recall curve for a given model and test data.
    Also, find the threshold that yields the highest F1 score.

    Parameters:
    model: The trained classifier to evaluate.
    model_name (optional): Name of the model to include in the plot title.

    Returns:
    optimal_threshold: The threshold value that yields the highest F1 score.
    optimal_report: Classification report using the optimal threshold.
    """
    probabilities = model.predict_proba(X_val)[:,1]

    precision, recall, thresholds = precision_recall_curve(y_val, probabilities)

    fscore = (2 * precision * recall) / (precision + recall + 1e-12)

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
    plt.show()

    optimal_report = classification_report(y_test, final_predictions, zero_division=0)
    return best_threshold, optimal_report

def generate_combinations_og(list_a, list_b):
    """
    Generates a list of combinations by rotating elements between two lists.

    Parameters:
    list_a (list): First list of elements.
    list_b (list): Second list of elements.

    Returns:
    list: A list of combinations where elements have been rotated between the two lists.
    """
    n = len(list_a)
    combinations = []
    
    # avoid modifying the original lists
    current_a = list_a[:]
    current_b = list_b[:]
    
    # n rounds to cycle through each list
    for _ in range(n):
        combinations.append([current_a[:], current_b[:]])
        
        # pop the last element from A and push it to the front of B
        element_a = current_a.pop()
        current_b.insert(0, element_a)
        
        # pop the last element from B and push it to the front of A
        element_b = current_b.pop()
        current_a.insert(0, element_b)

    # last step: mirrored lists
    combinations.append([current_a[:], current_b[:]])
    
    return combinations
    
def shap_importance(model):
    """
    Return a dataframe containing the features sorted by Shap importance.

    Parameters:
    model : The tree-based model (like RandomForest, XGBoost, etc.).
    X_train : pd.DataFrame
        Training set used to train the model (without the label).
    X_test : pd.DataFrame
        Test set or any set to compute SHAP values (without the label).

    Returns:
    pd.DataFrame
        A dataframe containing the features sorted by Shap importance.
    """
    explainer = shap.Explainer(model, X_train)
    
    shap_values = explainer.shap_values(X_test, check_additivity=False)
    
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
    return feature_importance.reset_index(drop=True).head(10)

def cosine_pairs_similarity(df, pairs, features):
    """
    Calculate cosine similarity for specified pairs in the DataFrame.

    Parameters:
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

def percentage_positive_negative_values(df, column_name):
    """
    Calculate the percentage of positive and negative values in a specified column of a DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame containing the data.
    column_name (str): The name of the column for which to calculate the percentages.
    
    Returns:
    pd.DataFrame: A DataFrame with the column name, percentage of positive values, and percentage of negative values.
    """
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

def wave_matches_display(df):
    """
    Processes a DataFrame to display wave-based match information.

    Parameters:
    df (pd.DataFrame): Input DataFrame with 'wave', 'iid', 'pid', and 'match' columns.

    Returns:
    pd.DataFrame: DataFrame with wave, participants, and matches information.
    """
    wave_data = {}

    # Group by wave and process each group
    for wave, group in df.groupby('wave'):
        participants = list(group['iid'].unique())
        
        matches = []
        for index, person in group.iterrows():
            if person['match'] == 1 and person['pid'] in group['iid'].values:
                matches.append([person['iid'], person['pid']])
        
        wave_data[wave] = {
            "participants": participants,
            "matches": matches
        }
    wave_df = pd.DataFrame([
        {'wave': wave, 'participants': data['participants'], 'matches': data['matches']}
        for wave, data in wave_data.items()
    ])

    return wave_df

def goal_mapping(df):
    """
    Maps and aggregates 'goal' values in the DataFrame.

    Parameters:
    df (pd.DataFrame): Input DataFrame with 'iid' and 'goal' columns.

    Returns:
    pd.DataFrame: Aggregated DataFrame with remapped 'goal' values.
    """

    # Step 1: Create a mapping dictionary for the 'goal' values
    mapping = {
        1: 1, 2: 1,
        3: 2, 4: 2,
        5: 3, 6: 3
    }

    # Step 2: Aggregate the DataFrame by 'iid' and take the first 'goal' from each group
    aggregated_df = df.groupby('iid').agg({'goal': 'first'}).reset_index()

    # Step 3: Apply the mapping to the 'goal' column
    # The 'map' function applies the mapping dictionary to the 'goal' column
    # The 'fillna' function ensures that any 'goal' values not in the mapping retain their original values
    aggregated_df['goal'] = aggregated_df['goal'].map(mapping).fillna(aggregated_df['goal'])

    return aggregated_df

def resolve_conflicts(row):
    """
    Resolves ranking conflicts in a row by assigning ranks based on sorted values.
    
    Parameters:
    row (pd.Series): A row of values to be ranked.

    Returns:
    pd.Series: A Series with ranks assigned to the values in the original row.
    """
    sorted_indices = np.argsort(row)
    ranks = np.arange(1, len(row) + 1)

    sorted_row = np.empty_like(ranks)
    sorted_row[sorted_indices] = ranks

    return pd.Series(sorted_row, index=row.index)

def individual_preference(iid, wave, attributes_averages, importance, shared, date_rankings):
    """
    Computes and ranks potential partners for a specified individual based on their preference weights, matching dating goals,
    individual rankings given during past dates, common interests, and the significance of those goals.
    
    Args:
    - attributes_averages (DataFrame): DataFrame containing averaged attributes and dating goals for each person.
    - importance (DataFrame): DataFrame containing the importance rankings for each attribute and dating goals for each person.
    - iid (int): The individual ID for whom to compute the scores.
    - date_rankings (DataFrame): DataFrame containing the rankings given by subjects to their dates across various attributes.
    - shared (DataFrame): DataFrame containing ratings for various activities and interests from 1 to 10.
    
    Returns:
    - DataFrame: A DataFrame of other persons ranked by compatibility for the specified individual.
    """
    if iid not in importance['iid'].values:
        return "Invalid iid. This individual does not exist in the importance dataset."

    # importance rankings and dating goal for the specified individual
    person_importance = importance.loc[importance['iid'] == iid].iloc[0]
    person_goal = person_importance['goal']

    # adjust importance rankings (now 6 most important, 1 least important) to be used as weights
    for attr in person_importance.index:
        if 'important' in attr:
            person_importance[attr] = 7 - person_importance[attr]

    # merging the date first impressions for the iid into attributes_averages
    relevant_rankings = date_rankings[date_rankings['iid'] == iid]
    if not relevant_rankings.empty:
        relevant_rankings = relevant_rankings.set_index('pid')
        attributes_averages = attributes_averages.join(relevant_rankings, on='iid', how='left', rsuffix='_ranked')

    # wave data for interests calculation
    shared = shared.set_index('iid')
    subject_interests = shared.loc[iid]
    partner_interests = shared.drop([iid], axis=0)

    # calculate common interests
    common_interests_columns = shared.columns.to_list()
    common_interests_score = (partner_interests[common_interests_columns] >= 8) & (subject_interests[common_interests_columns] >= 8)
    common_interests_score = common_interests_score.sum(axis=1) * 5

    scores_df = attributes_averages.copy()
    scores_df = scores_df.set_index('iid')
    scores_df['total_score'] = 0

    # Check and add match predictions directly from wave without merging
    match_bonus = 20
    wave_filtered = wave[wave['iid'] == iid][['pid', 'match_predictions']].set_index('pid')
    scores_df = scores_df.merge(wave_filtered, left_on='iid', right_index=True, how='left')
    scores_df['match_predictions']= scores_df['match_predictions'].fillna(0)
    scores_df['total_score'] += scores_df['match_predictions'] * match_bonus

    # weighted scores for attributes and replace NaN values where necessary
    for important_attr in person_importance.index:
        if 'important' in important_attr:
            observed_attr = important_attr.replace('_important', '') + '_o'
            ranked_attr = observed_attr.replace('_o', '') + '_ranked'
            if observed_attr in scores_df.columns:
                weighted_score = scores_df[observed_attr] * person_importance[important_attr]
                scores_df['total_score'] += weighted_score

    if not relevant_rankings.empty:
        scores_df['attr_ranked'] = relevant_rankings['attr']
        importance_attr = 'attr_important'
        scores_df['attr_ranked'] = scores_df['attr'].fillna(scores_df['attr_o'])
        #if importance_attr in person_importance:
        weighted_ranking = scores_df['attr_ranked']        
        scores_df['total_score'] += weighted_ranking
               

    # Add common interests score from the correctly indexed partner interests
    scores_df['common_interests_score'] = common_interests_score.fillna(0)
    scores_df['total_score'] += scores_df['common_interests_score']

    # Apply goal matching boost
    goal_boost = 10
    scores_df['total_score'] += (scores_df['goal'] == person_goal) * goal_boost

    # Remove the individual from the DataFrame, sort by total_score
    scores_df = scores_df.reset_index()
    scores_df = scores_df[scores_df['iid'] != iid]
    scores_df.sort_values('total_score', ascending=False, inplace=True)
    preference_list = scores_df['iid'].tolist()

    return preference_list
    