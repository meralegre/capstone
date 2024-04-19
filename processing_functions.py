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
    - A DataFrame with each 5-year age gap and the corresponding percentage 
    of people within that range.
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
    
def pr_threshold(model, X_test, y_test, model_name='Model'):
    """
    Calculate and plot the precision-recall curve for a given model and test data.
    Also, find the threshold that yields the highest F1 score and plot the classification report based on this threshold.

    Parameters:
    model: The trained classifier to evaluate.
    X_test: The test features.
    y_test: The true labels for the test set.
    model_name (optional): Name of the model to include in the plot title.

    Returns:
    optimal_threshold: The threshold value that yields the highest F1 score.
    optimal_report: Classification report using the optimal threshold.
    """
    
    # Predict probabilities for the positive class
    probabilities = model.predict_proba(X_test)[:, 1]

    # Calculate precision, recall, and thresholds using the predicted probabilities
    precision, recall, thresholds = precision_recall_curve(y_test, probabilities)

    # Calculate the F1 score for each threshold
    # Add a tiny constant to avoid division by zero
    fscore = (2 * precision * recall) / (precision + recall + 1e-12)

    # Find the index of the maximum F1 score
    optimal_index = np.argmax(fscore)
    optimal_threshold = thresholds[optimal_index]

    # Make final predictions using the optimal threshold
    final_predictions = (probabilities >= optimal_threshold).astype(int)

    # Plot precision, recall, and F1 score as functions of the threshold
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, precision[:-1], 'b--', label='Precision')
    plt.plot(thresholds, recall[:-1], 'g-', label='Recall')
    plt.plot(thresholds, fscore[:-1], 'r-.', label='F1 Score')
    plt.axvline(x=optimal_threshold, color='k', linestyle='--', label='Optimal Threshold')
    plt.xlabel('Threshold')
    plt.legend(loc='best')
    plt.ylim([0, 1])
    plt.title(f'{model_name}: Precision, Recall, and F1 Score for different thresholds')
    plt.show()

    # Generate and print classification report
    optimal_report = classification_report(y_test, final_predictions, zero_division=0)
    return optimal_threshold, optimal_report

def generate_combinations(list_a, list_b):
    n = len(list_a)
    combinations = []
    
    # Initial lists are copied to avoid modifying the original lists
    current_a = list_a[:]
    current_b = list_b[:]
    
    # We need n rounds to fully cycle through each list
    for _ in range(n):
        # Append the current configuration
        combinations.append([current_a[:], current_b[:]])
        
        # Pop the last element from A and push it to the front of B
        element_from_a = current_a.pop()
        current_b.insert(0, element_from_a)
        
        # Pop the last element from B and push it to the front of A
        element_from_b = current_b.pop()
        current_a.insert(0, element_from_b)

    combinations.append([current_a[:], current_b[:]])
    
    return combinations
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    