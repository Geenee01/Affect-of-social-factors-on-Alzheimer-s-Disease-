import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

base_path = "path/to/data/"
file_path = base_path + 'AD_Cleaned_data.csv'

df = pd.read_csv(file_path)
for i in range(len(df)):
    if df.loc[i, 'M/F'] == 'F':
        df.loc[i, 'M/F'] = 0
    elif df.loc[i, 'M/F'] == 'M':
        df.loc[i, 'M/F'] = 1

df.drop('Unnamed: 0', axis=1, inplace=True)
new_df = df.drop('CDR_bin', axis=1)

headers = new_df.columns

combos = list(combinations(headers, 2))

num = 0
for i in combos:
    two_columns_df = df[[i[0], i[1]]].copy()
    version_num = str(num)
    tail = 'updated_columns_{}.csv'.format(version_num)
    path = base_path + str(tail)
    two_columns_df.to_csv(path)
    num += 1
print("done")

# training and testing the model for each combo
j = 0
accuracy_lst = []
feature_combinations = []
svc = SVC(kernel='sigmoid')
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=12)
target = df['CDR_bin']

while j < 15:
    tail = 'updated_columns_{}.csv'.format(j)
    path = base_path + tail
    two_column_df = pd.read_csv(path)
    X_train, X_test, y_train, y_test = train_test_split(two_column_df, target, random_state=5, test_size=0.2)
    rf_classifier.fit(X_train, y_train)
    y_predicted = rf_classifier.predict(X_test)
    accuracy_score = rf_classifier.score(X_test, y_test)
    # print('accuracy', f'{rf_classifier.score(X_test, y_test):.2%}')
    accuracy_lst.append(accuracy_score * 100)
    feature_combinations.append(two_column_df.columns)
    print("The accuracy for the feature combination of {} and {} is {:.2%}.".format(combos[j][0], combos[j][1],
                                                                                    accuracy_score))

    j += 1

print(accuracy_lst)
print(feature_combinations)

combinations = [[col for col in combo if col != 'Unnamed: 0'] for combo in feature_combinations]

names = ['Combo' + str(i + 1) for i in range(len(combinations))]

accuracies = accuracy_lst

# visualize results with plots
# Add labels and title
plt.bar(names, accuracies)
#plt.figure(figsize = (5, 3))
plt.xlabel('Feature Combinations')
plt.xticks(rotation='vertical', fontsize = 6)
plt.ylabel('Accuracy')
plt.title('Random Forest Accuracy with Different Feature Combinations')
plt.tight_layout()

# Display the plot
plt.show()

