import numpy as np
import pandas as pd


def get_df():
    file = open("./Dataset/new.data")
    df = pd.DataFrame(columns=[x for x in range(1, 90)])
    for j in range(1541):
        temp = []
        for i in range(12):
            line = file.readline().split(" ")
            temp += line
        index = int(temp[0])
        df.loc[index] = temp[1:]
    return df


print("The description of the df\n")
df = get_df()
print(df.head())
print(df.describe())

df = df.replace("\\n", "", regex=True)  # removing the newlines from data
df = df.replace("-9", np.nan, regex=False)  # replacing -9 with NaN
print(df.head())

# df_vals = df.apply(lambda x:x.unique()==['0',])
# print(df_vals)
df = df[[2, 3, 8, 9, 11, 15, 18, 31, 37, 39, 40, 43, 50, 57]]  # dropped unwanted features
print(df.head())

df = df.astype(float)  # converting string to numeric values
# print(type(df[2].iloc[0]))
print("\nThe number of null values\n")
print(df.isnull().sum())
print(df.describe())

# Drop rows with invalid entries in target variable
df = df[df[57] <= 1]

# Finding NaN distribution for each feature
df_NaN = df.apply(lambda x: x.isnull().sum(), axis=1)
index = df_NaN[df_NaN > 3].index
df = df.drop(index=index)
print("\nThe number of null values after dropping some rows\n")
print(df.isnull().sum())
print(df.describe())

# Renaming the columns
col = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal',
       'num']
temp_col = df.columns
col_dict = dict(zip(temp_col, col))
df = df.rename(columns=col_dict)
print("\ndf after renaming the columns\n")
print(df.head())

print("\nThe number of different target values\n")
print(df['num'].value_counts())

print(df.isnull().sum())
cat_val = ['fbs', 'restecg', 'exang', 'slope', 'ca']

# Imputation by mode
df[cat_val] = df[cat_val].apply(lambda x: x.replace(np.nan, x.mode()[0]))
print("\nNumber of missing values after imputation by mode\n")
print(df.isnull().sum())

# Imputation by regression
from sklearn.linear_model import LinearRegression

missing_columns = ['trestbps', 'chol', 'thalach', 'oldpeak', 'thal']


def random_imputation(df, feature):
    number_missing = df[feature].isnull().sum()
    observed_values = df.loc[df[feature].notnull(), feature]
    df.loc[df[feature].isnull(), feature + '_imp'] = np.random.choice(observed_values, number_missing, replace=True)

    return df


df_temp = df.copy()
for feature in missing_columns:
    df_temp[feature + '_imp'] = df[feature]
    df_temp = random_imputation(df_temp, feature)

random_data = pd.DataFrame(columns=["Ran" + name for name in missing_columns])

for feature in missing_columns:
    random_data["Ran" + feature] = df_temp[feature + '_imp']
    parameters = list(set(df.columns) - set(missing_columns) - {feature + '_imp'})

    model = LinearRegression()
    model.fit(X=df_temp[parameters], y=df_temp[feature + '_imp'])

    # Standard Error of the regression estimates is equal to std() of the errors of each estimates
    predict = model.predict(df[parameters])
    std_error = (predict[df[feature].notnull()] - df_temp.loc[df_temp[feature].notnull(), feature + '_imp']).std()

    # observe that I preserve the index of the missing data from the original dataframe
    random_predict = np.random.normal(size=df_temp[feature].shape[0],
                                      loc=predict,
                                      scale=std_error)
    random_data.loc[(df_temp[feature].isnull()) & (random_predict > 0), "Ran" + feature] = random_predict[
        (df_temp[feature].isnull()) & (random_predict > 0)]

df[missing_columns] = random_data
print("\nThe description after imputation by regression\n")
print(df.describe())

# Visualising the corelation between features
import matplotlib.pyplot as plt
import seaborn as sns

corrmat = df.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20, 20))
sns.heatmap(df[top_corr_features].corr(), annot=True, cmap='RdYlGn')
plt.show()
print("\nThe correlation of features wrt the target\n")
print(corrmat['num'].sort_values(ascending=False))

X = df.drop(columns="num")
y = df["num"]

# Splitting the dataset
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardising the data
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import RandomizedSearchCV

# Extra trees Classifier
from sklearn.ensemble import ExtraTreesClassifier

et = ExtraTreesClassifier()
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
param_grid = {
    'bootstrap': [True, False],
    'max_features': [
        'sqrt', 'log2'],
    'min_samples_leaf': np.linspace(0.001, 0.5, 10),
    'min_samples_split': np.linspace(0.001, 0.5, 10),
    'n_estimators': np.linspace(10, 1000, 20, dtype=int),
    'criterion': ['gini', 'entropy']
}

search = RandomizedSearchCV(et, param_grid, n_iter=10, scoring='accuracy', n_jobs=-1, cv=cv, random_state=1)
# execute search
result = search.fit(X, y)
# summarize result
print("\nExtraTrees Classifier\n")
print('Best Score: %s' % result.best_score_)
print('Best Hyperparameters: %s' % result.best_params_)

# Random forest classifier
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
param_grid = {
    'bootstrap': [True, False],
    'max_features': ['sqrt', 'log2'],
    'min_samples_leaf': np.linspace(0.001, 0.5, 10),
    'min_samples_split': np.linspace(0.001, 0.5, 10),
    'n_estimators': np.linspace(10, 1000, 20, dtype=int),
    'criterion': ['gini', 'entropy']
}

search = RandomizedSearchCV(rf, param_grid, n_iter=10, scoring='accuracy', n_jobs=-1, cv=cv, random_state=1)
# execute search
result = search.fit(X, y)
# summarize result
print("\nRandom Forest Classifier\n")
print('Best Score: %s' % result.best_score_)
print('Best Hyperparameters: %s' % result.best_params_)

from sklearn.tree import DecisionTreeClassifier

param_grid = {
    'splitter': ['best', 'random'],
    'max_features': ['sqrt', 'log2'],
    'min_samples_leaf': np.linspace(0.001, 0.5, 10),
    'min_samples_split': np.linspace(0.001, 0.5, 10),
    'max_depth': [int(x) for x in np.linspace(10, 1000, 20, dtype=int)],
    'criterion': ['gini', 'entropy']
}

dt = DecisionTreeClassifier()
search = RandomizedSearchCV(dt, param_grid, n_iter=10, scoring='accuracy', n_jobs=-1, cv=cv, random_state=1)
# execute search
result = search.fit(X, y)
# summarize result
print("\nDecision Tree Classifier\n")
print('Best Score: %s' % result.best_score_)
print('Best Hyperparameters: %s' % result.best_params_)

from sklearn.svm import SVC

param_grid = {
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'gamma': ['auto', 'scale'],
    'degree': [x for x in range(3, 15)],
    "C": np.arange(2, 10, 2)
}

svm = SVC()
search = RandomizedSearchCV(svm, param_grid, n_iter=10, scoring='accuracy', n_jobs=-1, cv=cv, random_state=1)
# execute search
result = search.fit(X, y)
# summarize result
print("\nSVM Classifier\n")
print('Best Score: %s' % result.best_score_)
print('Best Hyperparameters: %s' % result.best_params_)

# Genetic algorithm
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from random import shuffle

pca = PCA(n_components=10).fit(X)
xtr_new = pca.transform(X_train)
xte_new = pca.transform(X_test)
scaler = StandardScaler().fit(xtr_new)
xtr_new = scaler.transform(xtr_new)
xte_new = scaler.transform(xte_new)
initial_population_size = 100
population = [np.random.randint(1, 15, (3,)) for i in range(initial_population_size)]
scores = []

for generation in range(initial_population_size // 2):
    del scores[:]
    for sample in population:
        nn = MLPClassifier(activation='relu', solver='sgd', max_iter=800,
                           alpha=1e-5, hidden_layer_sizes=sample, random_state=85)
        nn.fit(xtr_new, y_train)
        scores.append([nn.score(xte_new, y_test), sample])

    scores.sort(key=lambda x: x[0])
    if len(scores) == 2:
        break
    # create new population
    del scores[:2]
    shuffle(scores)
    population = [model[1] for model in scores]
    new_population = []
    for index in range(len(population))[0:-1:2]:
        new_population.append(np.concatenate((population[index][:1], population[index + 1][1:])))
        new_population.append(np.concatenate((population[index + 1][:1], population[index][1:])))
    population = list(new_population)
    print("Generation %d out of %d: done!" % (generation + 1, initial_population_size / 2))

print("\nMultiLayer Perceptron with Genetic Algorithm\n")
print(scores)
