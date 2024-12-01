import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.utils import resample
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report




#====================    10-YEAR HEART DISEASE PREDICTION    ====================#



random_state = 55
train_file = r"train.csv"
test_file = r"test.csv"
pd.set_option('display.max_rows', None)


mydata_train = pd.read_csv(train_file)
df1 = pd.DataFrame(mydata_train)



#---------------------  Prepare Data  --------------------#



plt.figure(figsize=(11,8))


# Plot no.1
plt.subplot(2, 2, 1)
colours = ['#B29177', 'white'] 
sns.heatmap(mydata_train.isnull(), cmap=sns.color_palette(colours), cbar=False)
plt.title("Missing Data")


# Convert categorical values to binary values
df1['sex'] = df1['sex'].map({'M':1, 'F':0})
df1['is_smoking'] = df1['is_smoking'].map({'YES':1, 'NO':0})
unique, count = np.unique(df1['TenYearCHD'], return_counts=True)
# print("before oversampling: ", unique, count)


# Plot no.2
plt.subplot(2, 2, 2)
sns.countplot(x='TenYearCHD',data=df1,hue='TenYearCHD',palette='coolwarm',legend=True)
plt.title("Data Before OverSampling")
plt.ylabel("Count")


# Find mean of missing data and fill null values
cig_mean1 = df1.loc[:, 'cigsPerDay'].mean()
edu_mean1 = df1.loc[:, 'education'].mean()
df1.fillna({'cigsPerDay':round(cig_mean1)}, inplace=True)
df1.fillna({'education':round(edu_mean1)}, inplace=True)


mean_values1 = df1.groupby('sex')[['BMI', 'BPMeds', 'totChol', 'heartRate', 'glucose']].mean()
m,n = mean_values1.shape


for i in range(m):
    for j in range(n):
        row_name1    = mean_values1.index[i]
        column_name1 = mean_values1.columns[j]
        value1 = round(mean_values1.iat[i,j], 4)
        df1.loc[(df1['sex'] == i) & (df1[column_name1].isna()), column_name1] = value1


# Separate majority and minority classes
majority = df1[df1['TenYearCHD'] == 0]
minority = df1[df1['TenYearCHD'] == 1]


# Oversample the minority class
minority_oversampled = resample(minority,replace=True, n_samples=len(majority),random_state=random_state)


df_balanced = pd.concat([majority, minority_oversampled])


unique, count = np.unique(df_balanced['TenYearCHD'], return_counts=True)
# print("this is after oversampling: ", unique, count)


# Plot no.3
plt.subplot(2, 2, 3)
sns.countplot(x='TenYearCHD',data=df_balanced,hue='TenYearCHD',palette='coolwarm',legend=True)
plt.title("Data After OverSampling")
plt.ylabel("Count")



#----------------------  Train Data  ---------------------#



X = df_balanced.iloc[:,:-1]  # All rows, all columns except the last
Y = df_balanced.iloc[:,-1]   # All rows, only the last column


# Scale the data
X = preprocessing.StandardScaler().fit(X).transform(X)
X_train,X_val,y_train,y_val = train_test_split(X,Y,train_size=0.8,random_state=random_state,shuffle=True)


model = {}
accuracy = {}
precision = {}
recall = {}


model['logreg1'] = LogisticRegression(solver='lbfgs', max_iter=400)
# model['logreg2'] = LogisticRegression(solver='liblinear', max_iter=100)
# model['logreg3'] = LogisticRegression(solver='newton-cg', max_iter=100)
# model['logreg4'] = LogisticRegression(solver='newton-cholesky', max_iter=400)
# model['logreg5'] = LogisticRegression(solver='sag', max_iter=100)
# model['logreg6'] = LogisticRegression(solver='saga', max_iter=100)
model['xgboost'] = XGBClassifier()
model['randfor1'] = RandomForestClassifier()
model['randfor2'] = RandomForestClassifier(min_samples_split=2, max_depth=16,n_estimators=50)
# model['randfor3'] = RandomForestClassifier(min_samples_split=2, max_depth=16,n_estimators=500)
model['dtree']   = DecisionTreeClassifier(max_depth=50, random_state=random_state)


for key in model.keys():
    model[key].fit(X_train, y_train)
    predictions1 = model[key].predict(X_val)
    accuracy[key] = accuracy_score(predictions1,y_val)
    precision[key] = precision_score(predictions1,y_val)
    recall[key] = recall_score(predictions1, y_val)
    # print(key, classification_report(y_val,predictions1))


# Compare efficiency in models
df_model = pd.DataFrame(index=model.keys(), columns=['Accuracy','Precision','Recall'])
df_model['Accuracy'] = accuracy.values()
df_model['Precision'] = precision.values()
df_model['Recall'] = recall.values()
print(df_model)


# Plot no.4
plt.subplot(2, 2, 4)
df_model.plot.bar(rot=45, ax=plt.gca())
plt.title("Different Models")
plt.tight_layout()
plt.show()



#-------------------  Predict Test Data  -----------------#



mydata_test = pd.read_csv(test_file)
df2 = pd.DataFrame(mydata_test)


# Convert categorical values to binary values
df2['sex'] = df2['sex'].map({'M':1, 'F':0})
df2['is_smoking'] = df2['is_smoking'].map({'YES':1, 'NO':0})


# Find mean of missing data and fill null values
edu_mean2 = df2.loc[:, 'education'].mean()
cig_mean2 = df2.loc[:, 'cigsPerDay'].mean()
df2.fillna({'education':round(edu_mean2)}, inplace=True)
df2.fillna({'cigsPerDay':round(cig_mean2)}, inplace=True)


mean_values2 = df2.groupby('sex')[['BPMeds', 'totChol', 'BMI', 'glucose']].mean()
o,p = mean_values2.shape


for i in range(o):
    for j in range(p):
        row_name2    = mean_values2.index[i]
        column_name2 = mean_values2.columns[j]
        value = round(mean_values2.iat[i,j], 4)
        df2.loc[(df2['sex'] == i) & (df2[column_name2].isna()), column_name2] = value
   

# Scale the data
X2 = df2.iloc[:,:]  # All rows, all columns
X2 = preprocessing.StandardScaler().fit(X2).transform(X2)


predictions2 = model['randfor1'].predict(X2)
unique, count = np.unique(predictions2, return_counts=True)
# print(unique, count)