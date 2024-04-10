import pandas as pdimport numpy as npimport matplotlib.pyplot as pltimport seaborn as snsfrom processing_functions import *df = pd.read_csv('SpeedDatingData.csv', encoding='latin1')# Let's start by changing the empty spaces by nan values to have a clear image# of the dataset. Since most of these columns have '?' as the nan value, they # have mismatched datatypes, being objects where they should be ints or floats.# Therefore I decided to create a function to change the column to its# corresponding datatype.df.applymap(convert_to_numeric).replace('?', np.nan, inplace=True)### Look into the datasetdf.info(verbose=True)### Check percentage of nan values per column. # To make life easier I decided to define a function to return the # corresponding percentage for each column.nan_percentage(df) # will probably delete some rows if the provided information is null# During the experiment they devided the speed dating events in three steps. # First step consisted on asking the participants questions based on their # initial thoughts on the selected attributes, how they perceive themselves, # how they perceive the others, what is the most important attribute, etc.# Since I will be only be taking into account first impressions (simulating a# dating app) I will remove the data observed after the dates themselves.### explain the column names and why the selected ones for this case are ### being filtered_columns = [    col for col in df.columns if not col.endswith('_2')     and not col.endswith('_3')    and not col.endswith('_s')    ]# Select only the filtered columnsdf_filtered = df[filtered_columns]# Now we have in total 111 columns that center more around the actual dates# I am going to rename the columns associated to rankings to have a clear # understanding of which is which isntead of having it associated with numbersdf_filtered.columns = [    col.replace('1_1', '') + '_important'     if col.endswith('1_1') else col for col in df_filtered.columns    ]