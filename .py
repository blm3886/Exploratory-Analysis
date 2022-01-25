# importing important libraries
import pandas as pd
import numpy as np 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

data_income= pd.read_csv("E:\pds\income.csv")
df=data_income.copy()
print(df.info())

#summary of the numerical variable
df.describe()
df.isnull().sum()

# summary of the categorical variables
summary_cate=df.describe(include="O")
print(summary_cate)

#frequency of each category
df['JobType'].value_counts() # the output of this command gives an idea that 
                             #there are missing values existing as '?'

df['EdType'].value_counts()
df['occupation'].value_counts() #this also has '?' as missing value


# directly reading the ? as a blank value while reading the file itself
df= pd.read_csv("E:\pds\income.csv", na_values= " ?")
df.describe()
df.info()
df.isnull().sum()

missing= df[df.isnull().any(axis=1)]

#dropping all the values having any missing values
df1= df.dropna(axis=0)
df1.info()

#getting the correlation of the int variables
correlation= df1.corr()
print(correlation)
#==============================================================================
# Cross tables and data visualization

#  Gender proportion table
list(df1.columns)
gender= pd.crosstab(index= df1['gender'], columns= 'count', normalize= True)
print(gender)

# Gender vs salary status
gen_salstat= pd.crosstab(index= df1['gender'], columns= df1['SalStat'], normalize= True)
print(gen_salstat)

# Age vs salary status
age_salstat= pd.crosstab(index= df1['age'], columns= df1['SalStat'], normalize= True)
print(age_salstat)

# jobtype and salarystatus
sns.countplot(y=df1['JobType'], hue= df1['SalStat'])
jobtype_salstat= pd.crosstab(index= df1['JobType'], columns= df1['SalStat'], normalize= True)
print(jobtype_salstat)
sns.countplot(y=df1['EdType'], hue= df1['SalStat'])
sns.countplot(y=df1['maritalstatus'], hue= df1['SalStat'])
sns.countplot(y=df1['occupation'], hue= df1['SalStat'])
sns.countplot(y=df1['relationship'], hue= df1['SalStat'])
sns.countplot(y=df1['race'], hue= df1['SalStat'])
sns.countplot(y=df1['gender'], hue= df1['SalStat'])
sns.boxplot(y=df1['hoursperweek'], x= df1['SalStat'])


#==============================================================================
#visualizing the numerical variables
sns.distplot(df1.age, bins= 10)
sns.distplot(df1.capitalgain, bins= 10)
sns.distplot(df1.capitalloss)
sns.distplot(df1.hoursperweek)

#==============================================================================
# Building the model: using the Logistic regression technique as the output is a categorical
# LOGISTIC REGRESSION
#==============================================================================
from sklearn.metrics import accuracy_score, confusion_matrix
#reindexing the salary status names to 0 and 1
df1.loc[(df1['SalStat']== " less than or equal to 50,000"), 'SalStat']= 0
df1['SalStat']
df1.loc[(df1['SalStat']== " greater than 50,000"), 'SalStat']=1
df1['SalStat']

new_data= pd.get_dummies(df1, drop_first=True)

#Storing the column names
columns_list= list(new_data.columns)
print(columns_list)

#Separating the input variables from the data
features= list(set(columns_list)-set(['SalStat']))
print(features)

#Storing the values in y
y= new_data['SalStat'].values
print(y)

#Storing the values in x
x= new_data[features].values
print(x)

#Splitting the data into train and test set
train_x, test_x, train_y, test_y= train_test_split(x,y,train_size=0.7, test_size=0.3, random_state=0)

#Make instance of the model
logistic= LogisticRegression()

#fitting the values for x and y
logistic.fit(train_x, train_y)
logistic.coef_
logistic.intercept_

#prediction from test data
prediction= logistic.predict(test_x)
print(prediction)

#Confusion matrix: used to evaluate the performance of an evalation model
# It gives the number of correct predictions and the number of incorrect predictions

confusion_matrix= confusion_matrix(test_y, prediction)
print(confusion_matrix)

#Calculating the accuracy
accuracy_score= accuracy_score(test_y, prediction)
print(accuracy_score)

#printing the misclassified values from prediction
print('Misclassified samples: %d' %(test_y !=prediction).sum())

#==============================================================================
# Building the model: using the Logistic regression technique as the output is a categorical
# LOGISTIC REGRESSION: Restructuring the model to improve the performance by removing insignificant variables
#==============================================================================
from sklearn.metrics import accuracy_score, confusion_matrix
#reindexing the salary status names to 0 and 1
df1.loc[(df1['SalStat']== " less than or equal to 50,000"), 'SalStat']= 0
df1['SalStat']
df1.loc[(df1['SalStat']== " greater than 50,000"), 'SalStat']=1
df1['SalStat']

cols= ['gender', 'nativecountry', 'race', 'JobType']
new_data= df1.drop(cols, axis=1)
new_data= pd.get_dummies(new_data, drop_first=True)

#Storing the column names
columns_list= list(new_data.columns)
print(columns_list)

#Separating the input variables from the data
features= list(set(columns_list)-set(['SalStat']))
print(features)

#Storing the values in y
y= new_data['SalStat'].values
print(y)

#Storing the values in x
x= new_data[features].values
print(x)

#Splitting the data into train and test set
train_x, test_x, train_y, test_y= train_test_split(x,y,train_size=0.7, test_size=0.3, random_state=0)

#Make instance of the model
logistic= LogisticRegression()

#fitting the values for x and y
logistic.fit(train_x, train_y)

logistic.coef_
logistic.intercept_

#prediction from test data
prediction= logistic.predict(test_x)
print(prediction)

#Confusion matrix
confusion_matrix= confusion_matrix(test_y, prediction)
print(confusion_matrix)

#Calculating the accuracy
accuracy_score= accuracy_score(test_y, prediction)
print(accuracy_score)

#printing the misclassified values from prediction
print('Misclassified samples: %d' %(test_y !=prediction).sum())

# ==============================================================================
# KNN Classifier
# ==============================================================================
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# import library for plotting
import matplotlib.pyplot as plt

#Storing the K nearest neighbors classifier
KNN_classifier= KNeighborsClassifier(n_neighbors= 5)

#fitting X and Y
KNN_classifier.fit(train_x, train_y)

#prediction
pred= KNN_classifier.predict(test_x)
print(pred)

#performance checking
confusion_matrix= confusion_matrix(test_y, pred)
print("\t", "Predicted values")
print("Original values","\n", confusion_matrix)
 
accuracy_score= accuracy_score(test_y,pred)
print(accuracy_score)

Misclassified_sample= []
# Calculating errors for K values between 1 and 20
for i in range(1,20):
    knn= KNeighborsClassifier(n_neighbors=i)
    knn.fit(train_x, train_y)
    pred_i= knn.predict(test_x)
    Misclassified_sample.append((test_y != pred_i).sum())
    
print(Misclassified_sample)