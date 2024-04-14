import pandas as pdimport numpy as npimport matplotlib.pyplot as pltimport seaborn as snsfrom processing_functions import *df = pd.read_csv('data/SpeedDatingData.csv', encoding='latin1')# Let's start by changing the empty spaces by nan values to have a clear image# of the dataset. Since most of these columns have '?' as the nan value, they # have mismatched datatypes, being objects where they should be ints or floats.# Therefore I decided to create a function to change the column to its# corresponding datatype.df.applymap(convert_to_numeric).replace('?', np.nan, inplace=True)### Look into the datasetdf.info(verbose=True)### Check percentage of nan values per column. # To make life easier I decided to define a function to return the # corresponding percentage for each column.nan_percentage(df) # will probably delete some rows if the provided information is null# During the experiment they devided the speed dating events in three steps. # First step consisted on asking the participants questions based on their # initial thoughts on the selected attributes, how they perceive themselves, # how they perceive the others, what is the most important attribute, etc.# Since I will be only be taking into account first impressions (simulating a# dating app) I will remove the data observed after the dates themselves.### explain the column names and why the selected ones for this case are ### being filtered_columns = [    col for col in df.columns if not col.endswith('_2')     and not col.endswith('_3')    and not col.endswith('_s')]# Select only the filtered columnsdf_filtered = df[filtered_columns]# Now we have in total 111 columns that center more around the actual dates# I am going to rename the columns associated to rankings to have a clear # understanding of which is which isntead of having it associated with numbersrename_suffix = [    ('1_1', '_important'),    ('2_1', '_o_want'),    ('3_1', '_self'),    ('4_1', '_fellow_want'),    ('5_1', '_perceived')]df_renamed = rename_columns(df_filtered, rename_suffix)# since 'field' has too many unique values, we'll drop it. It is not that # important either for this research#df_renamed['field'].value_counts(normalize=True) df_renamed.drop(columns='field', inplace=True)attribute_groups = [    ['attr_important', 'sinc_important', 'intel_important', 'fun_important', 'amb_important', 'shar_important'],    ['attr_fellow_want', 'sinc_fellow_want', 'intel_fellow_want', 'fun_fellow_want', 'amb_fellow_want', 'shar_fellow_want'],    ['attr_o_want', 'sinc_o_want', 'intel_o_want', 'fun_o_want', 'amb_o_want', 'shar_o_want']]df_example = df_renamed.apply(lambda row: attribute_ranking(row, attribute_groups), axis=1)### Plots# Let's see first the gender and age distribution## Gendergender_counts = df['gender'].value_counts()# Create a pie chartplt.figure(figsize=(8, 8))plt.pie(gender_counts, labels=['Female', 'Male'], autopct='%1.1f%%')plt.title('Proportion of Gender')plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.plt.show()age_dist_df = age_distribution(df)print(age_dist_df)suffix = ['_fellow_want', 'perceived', '_o_want']dropped = drop_columns(df_renamed, suffix)df_csv = pd.read_csv('data/input_file.csv')df_parquet = pd.read_parquet('data/input_file.parquet')