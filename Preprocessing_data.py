# This program is directed towards pre-processing the Alzeimher's Disease dataset.

# Import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

# pathe to folder
base_path = "path/to/files/"
# Read in the dataset
df = pd.read_csv(base_path +'/oasis_cross-sectional.csv')
df.head()

df.isna().sum() # Sums the total number of NaNs in each column
df= df.drop(['Delay', 'Hand', 'ID'], axis = 1) #drops the 'Delay' and 'Hand' columns
df

df.dropna(how='any', inplace=True) # drops all rows where NaN is True
df.head()
df.describe() # used to compare with central tendencies of the original dataset to make sure cleaned data is not altered

# Visualization: Heat map for correlations within the df using Spearman correlation.
plt.figure(figsize=(10,5))
sns.heatmap(df[['Age','CDR','SES','Educ','nWBV']].corr(method='spearman'), annot=True,cmap='crest')

# Visualization: Density plot for 'Education'
sns.histplot(df['Educ'], kde = True, color = 'green', label ='Education')
plt.title('Density Plot for Education')
plt.xlabel('Education Level')
plt.ylabel('Number')

# Visualization: Density plot for 'CDR'
sns.histplot(df['CDR'], kde = True, label ='CDR')
plt.title('Density Plot for Clinical Dementia Rating')
plt.xlabel('CDR Value')
plt.ylabel('Frequency')

# Visualization: Density plot for 'WBV'
sns.histplot(df['nWBV'], kde = True, color = 'skyblue', label ='Whole Brain Volume')
plt.title('Density Plot for the Normalized Whole Brain Volume')
plt.xlabel('Whole Brain Volume')
plt.ylabel('Number')

# Data Engineering: Create a new column by changing the 0.0 and 1.0 in the CDR column to 'No' and 'Yes', respectively.
new_column= []
for value in df['CDR']:
    if value < 0.0001:
        new_column.append('No')
    else:
        new_column.append('Yes')
df['CDR Y/N'] = new_column
new_df = df.drop(['CDR', 'eTIV', 'ASF'], axis=1)
print(new_df)

# Change the 'M' and 'F' in the 'M/F' column, to 0 and 1, respectively. This allows for the ML model to perform a more accurate analysis.
new_sex = []
for value in new_df['M/F']:
    if value == 'M':
        new_sex.append(0)
    else:
        new_sex.append(1)
new_df['M/F'] = new_sex

# Changes the N and Y values in 'CDR Y/N' to 0 and 1, respectively.
new_valCDR = []
for value in new_df['CDR Y/N']:
    if value == 'Yes':
        new_valCDR.append(1)
    else:
        new_valCDR.append(0)
new_df['CDR Y/N'] = new_valCDR
print(new_df)

# save cleaned data
out_put_file = base_path + "AD_Cleaned_data.csv"
new_df.to_csv 
