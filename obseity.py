#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
data=pd.read_csv('ObesityDataSet_raw_and_data_sinthetic.csv')
pd.set_option('display.float_format',lambda x : '%.2f' % x)
data.info()


# In[2]:


data.describe(include = 'all')


# In[3]:


from pandas_profiling import ProfileReport
profile = ProfileReport(data, title="Pandas Profiling Report")
profile.to_widgets()
profile.to_notebook_iframe()


# In[4]:


import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
import collections
from collections import Counter

import sklearn
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import SGDClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


# In[5]:


df = pd.read_csv('ObesityDataSet_raw_and_data_sinthetic.csv')


# In[4]:


df


# In[5]:


df.shape


# In[6]:


df.info()


# In[7]:


df.describe()


# In[8]:


df.columns


# In[9]:


df.columns = ['Gender', 'Age', 'Height', 'Weight', 'Family History with Overweight',
       'Frequent consumption of high caloric food', 'Frequency of consumption of vegetables', 'Number of main meals', 'Consumption of food between meals', 'Smoke', 'Consumption of water daily', 'Calories consumption monitoring', 'Physical activity frequency', 'Time using technology devices',
       'Consumption of alcohol', 'Transportation used', 'Obesity']

df


# In[10]:


df['Obesity'] = df['Obesity'].apply(lambda x: x.replace('_', ' '))
df['Transportation used'] = df['Transportation used'].apply(lambda x: x.replace('_', ' '))
df['Height'] = df['Height']*100
df['Height'] = df['Height'].round(1)
df['Weight'] = df['Weight'].round(1)
df['Age'] = df['Age'].round(1)
df


# In[11]:


for x in ['Frequency of consumption of vegetables', 'Number of main meals', 'Consumption of water daily', 'Physical activity frequency', 'Time using technology devices']:
    value = np.array(df[x])
    print(x,':', 'min:', np.min(value), 'max:', np.max(value))


# ## Exploratory Data Analysis

# In[12]:


for x in ['Frequency of consumption of vegetables', 'Number of main meals', 'Consumption of water daily', 'Physical activity frequency', 'Time using technology devices']:
    df[x] = df[x].apply(round)
    value = np.array(df[x])
    print(x,':', 'min:', np.min(value), 'max:', np.max(value), df[x].dtype)
    print(df[x].unique())
    


# In[13]:


df1 = df.copy()


# In[14]:


mapping0 = {1:'Never', 2:'Sometimes', 3:'Always'}
mapping1 = {1: '1', 2:'2' , 3: '3', 4: '3+'}
mapping2 = {1: 'Less than a liter', 2:'Between 1 and 2 L', 3:'More than 2 L'}
mapping3 = {0: 'I do not have', 1: '1 or 2 days', 2: '2 or 4 days', 3: '4 or 5 days'}
mapping4 = {0: '0–2 hours', 1: '3–5 hours', 2: 'More than 5 hours'}


# In[15]:


df['Frequency of consumption of vegetables'] = df['Frequency of consumption of vegetables'].replace(mapping0)
df['Number of main meals'] = df['Number of main meals'].replace(mapping1)
df['Consumption of water daily'] = df['Consumption of water daily'].replace(mapping2)
df['Physical activity frequency'] = df['Physical activity frequency'].replace(mapping3)
df['Time using technology devices'] = df['Time using technology devices'].replace(mapping4)


# In[16]:


df


# ### Age, Height and Weight

# In terms of height, male and female are similarly distributed according to the box plot below. While male are generally taller than female, both male and female share a similar average in weight, with female having a much larger range of weight (as well as BMI) compared to male. This is further illustrated by the steeper line plot between weight and height of female than male.

# In[18]:


sns.set()
fig = plt.figure(figsize=(20,10))
plt.subplot(1, 2, 1)
sns.boxplot(x='Gender', y='Height', data=df)
plt.subplot(1, 2, 2)
sns.boxplot(x='Gender', y='Weight', data=df)


# In[760]:


sns.set()
g = sns.jointplot("Height", "Weight", data=df,
                  kind="reg", truncate=False,
                  xlim=(125, 200), ylim=(35, 180),
                  color="m", height=10)
g.set_axis_labels("Height (cm)", "Weight (kg)")


# In[761]:


g = sns.lmplot(x="Height", y="Weight", hue="Gender",
               height=10, data=df)
g.set_axis_labels("Height (cm)", "Weight (kg)")


# ### Obesity

# In[762]:


c = Counter(df['Obesity'])
print(c)


# In[763]:


fig = plt.figure(figsize=(8,8))
plt.pie([float(c[v]) for v in c], labels=[str(k) for k in c], autopct=None)
plt.title('Weight Category') 
plt.tight_layout()


# In[764]:


filt = df['Gender'] == 'Male'
c_m = Counter(df.loc[filt, 'Obesity'])
print(c_m)
c_f = Counter(df.loc[~filt, 'Obesity'])
print(c_f)


# A bigger proportion of female with a higher BMI is reflected by the large slice of Obesity Type III in the pie chart below, while Obesity Type II is the most prevalent type of obesity in make. Interestingly, there is also a higher proportion of Insufficient Weight in female compared to male, this could be explained by a heavier societal pressure on women to go on diets.

# In[765]:


fig = plt.figure(figsize=(20,8))
plt.subplot(1, 2, 1)
plt.pie([float(c_m[v]) for v in c_m], labels=[str(k) for k in c_m], autopct=None)
plt.title('Weight Category of Male') 
plt.tight_layout()

plt.subplot(1, 2, 2)
plt.pie([float(c_f[v]) for v in c_f], labels=[str(k) for k in c_f], autopct=None)
plt.title('Weight Category of Female') 
plt.tight_layout()


# ### Eating and Exercise Habits

# In[766]:


for a in df.columns[4:-1]:
    data = df[a].value_counts()
    values = df[a].value_counts().index.to_list()
    counts = df[a].value_counts().to_list()
    
    plt.figure(figsize=(12,5))
    ax = sns.barplot(x = values, y = counts)
    
    plt.title(a)
    plt.xticks(rotation=45)
    print(a, values, counts)


# ## Data Preprocessing

# In[767]:


df1.head()


# #### Since classifier cannot operate with label data directly, One Hot Encoder and Label Encoding will be used to assign numeric values to each category

# In[768]:


# identity categorical variables (data type would be 'object')
cat = df1.dtypes == object

print(cat)

# When dtype == object is 'true'
print(cat[cat])
cat_labels = cat[cat].index
print('Categorical variables:', cat_labels)

# When dtype == object is 'false'
false = cat[~cat]
non_cat = false.index
print('Non Categorical variables:', non_cat)


# In[769]:


# identify categorical variables with more than 2 values/answers
col = [x for x in labels]
multiple = [df1[x].unique() for x in labels]

multi_col = {col: values for col, values in zip(col, multiple) if len(values)>2}
print(multi_col)
print('\n')
print('Categorical variables with more than 2 values/answers:', multi_col.keys())


# In[770]:


df1.head(3)


# In[771]:


df1.columns

def col_no(x):
    d = {}
    d[df1.columns[x]] = x
    return(d)

print([col_no(x) for x in range(0, len(df1.columns))])


# In[772]:


x = df1[df1.columns[:-1]]
y = df['Obesity']

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)


# In[773]:


le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_train


# In[774]:


Scale_features = ['Age', 'Height', 'Weight']
Scale_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('Scaling', StandardScaler())
])

Ordi_features = ['Consumption of food between meals', 'Consumption of alcohol']
Ordi_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('Ordi', OrdinalEncoder())
])

NonO_features = ['Gender', 'Family History with Overweight', 'Frequent consumption of high caloric food', 'Smoke', 'Calories consumption monitoring', 'Transportation used']
NonO_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('Non-O', OneHotEncoder())
])

Preprocessor = ColumnTransformer(transformers=[
    ('Scale', Scale_transformer, Scale_features),
    ('Ordinal', Ordi_transformer, Ordi_features),
    ('Non-Ordinal', NonO_transformer, NonO_features)
], remainder = 'passthrough')
    
clf = Pipeline(steps=[('preprocessor', Preprocessor)])


# In[775]:


clf.fit(x_train, y_train)


# In[776]:


trans_df = clf.fit_transform(x_train)
print(trans_df.shape)


# In[777]:


# Column name of first two steps in pipeline

cols = [y for x in [Scale_features, Ordi_features] for y in x]
cols


# In[778]:


# Column names of OneHotEncoder step in pipeline

ohe_cols = clf.named_steps['preprocessor'].transformers_[2][1]\
    .named_steps['Non-O'].get_feature_names(NonO_features)
ohe_cols = [x for x in ohe_cols]
ohe_cols


# In[779]:


# Column names of remainder='Passthrough' - remaining columns that didn't get processed
non_cat


# In[780]:


transformed_x_train = pd.DataFrame(trans_df, columns= ['Age', 'Height',
 'Weight',
 'Consumption of food between meals',
 'Consumption of alcohol','Gender_Female',
 'Gender_Male',
 'Family History with Overweight_no',
 'Family History with Overweight_yes',
 'Frequent consumption of high caloric food_no',
 'Frequent consumption of high caloric food_yes',
 'Smoke_no',
 'Smoke_yes',
 'Calories consumption monitoring_no',
 'Calories consumption monitoring_yes',
 'Transportation used_Automobile',
 'Transportation used_Bike',
 'Transportation used_Motorbike',
 'Transportation used_Public Transportation',
 'Transportation used_Walking', 'Frequency of consumption of vegetables',
 'Number of main meals',
 'Consumption of water daily',
 'Physical activity frequency',
 'Time using technology devices'])


# In[781]:


# transformed/processed features

transformed_x_train


# In[782]:


le = LabelEncoder()
y_test = le.fit_transform(y_test)
le_name_mapping = dict(zip(le.transform(le.classes_), le.classes_))
print(le_name_mapping)


# ## Model Selection

# Classifiers are selected and stored in a list, each classifier will be looped through and the preprocessor will be applied each time. The accuracy score of every classifier will be printed out.

# In[783]:


classifiers = [
    KNeighborsClassifier(n_neighbors = 5),
    SVC(kernel="rbf", C=0.025, probability=True),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    SGDClassifier()
    ]

top_class = []

for classifier in classifiers:
    pipe = Pipeline(steps=[('preprocessor', Preprocessor),
                      ('classifier', classifier)])
    
    # training model
    pipe.fit(x_train, y_train)   
    print(classifier)
    
    acc_score = pipe.score(x_test, y_test)
    print("model score: %.3f" % acc_score)
    
    # using the model to predict
    y_pred = pipe.predict(x_test)
    
    target_names = [le_name_mapping[x] for x in le_name_mapping]
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    if acc_score > 0.8:
        top_class.append(classifier)


# In[1]:


import seaborn as sns
import matplotlib.pyplot as plt
# 数据
models = ['KNeighborsClassifier', 'SVC', 'GradientBoostingClassifier', 'SGDClassifier']
scores = [0.830, 0.500, 0.962, 0.726]
categories = ['Insufficient Weight', 'Normal Weight', 'Obesity Type I', 'Obesity Type II', 'Obesity Type III', 'Overweight Level I', 'Overweight Level II']

# 模型得分条形图
plt.figure(figsize=(10, 6))
sns.barplot(x=models, y=scores)
plt.title('Model Scores')
plt.ylabel('Score')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 每个模型的精确度、召回率和F1分数条形图
precision_data = {
    'KNeighborsClassifier': [0.71, 0.76, 0.86, 0.96, 1.00, 0.71, 0.79],
    'SVC': [1.00, 0.72, 0.25, 0.83, 1.00, 0.00, 0.00],
    'GradientBoostingClassifier': [0.96, 1.00, 0.94, 1.00, 1.00, 0.89, 0.95],
    'SGDClassifier': [1.00, 0.50, 0.71, 1.00, 1.00, 0.48, 0.55]
}

# 模型的召回率数据
recall_data = {
    'KNeighborsClassifier': [0.92, 0.55, 0.91, 0.92, 1.00, 0.74, 0.75],
    'SVC': [0.23, 0.45, 0.97, 0.76, 1.00, 0.00, 0.00],
    'GradientBoostingClassifier': [1.00, 0.90, 1.00, 0.92, 1.00, 0.93, 0.97],
    'SGDClassifier': [1.00, 0.90, 0.44, 0.96, 1.00, 0.41, 0.47]
}

# 模型的F1分数数据
f1_data = {
    'KNeighborsClassifier': [0.80, 0.64, 0.89, 0.94, 1.00, 0.73, 0.77],
    'SVC': [0.38, 0.55, 0.40, 0.79, 1.00, 0.00, 0.00],
    'GradientBoostingClassifier': [0.98, 0.95, 0.97, 0.96, 1.00, 0.91, 0.96],
    'SGDClassifier': [1.00, 0.64, 0.55, 0.98, 1.00, 0.44, 0.51]
}
for metric, data in zip(['Precision', 'Recall', 'F1-Score'], [precision_data, recall_data, f1_data]):
    plt.figure(figsize=(12, 8))
    for model in models:
        sns.lineplot(x=categories, y=data[model], label=model)
    plt.title(f'{metric} for Different Models')
    plt.ylabel(metric)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.legend()
    plt.show()


# In[ ]:




