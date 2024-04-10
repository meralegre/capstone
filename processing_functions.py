import pandas as pd

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

def attributes_ranking(row, attribute_columns, wave_column, scale_min=1, scale_max=10):
    """
    Scales attribute points directly in the original columns based on specified conditions.
    
    Parameters:
    - row: The row of the DataFrame being processed.
    - attribute_columns: List of columns containing attribute points to be scaled.
    - wave_column: The column to check for the specified condition.
    - scale_min, scale_max: The minimum and maximum of the new scale (default 1 to 10).
    
    Returns:
    - A modified row with scaled attributes if conditions are met.
    """
    # Check if the 'wave' column's value is within the specified ranges
    if row[wave_column] in list(range(1, 6)) + list(range(10, 22)):
        for col in attribute_columns:
            # Scale the attribute directly, updating the original column value
            row[col] = round(((row[col] / 100) * (scale_max - scale_min)) + scale_min, 2)
    return row
    