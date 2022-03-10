#!/usr/bin/env python
# coding: utf-8

# **30E03000 - Data Science for Business I (2021)**

# # Assignment 2: Credit Risk Modeling

# ## Import libraries

# In[3]:


import pandas as pd

#add all necessary libraries here
import numpy as np #scientific computing
import pandas as pd #data management
import itertools

#matplotlib for plotting
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.ticker as mtick #for percentage ticks

#sklearn for modeling
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier #Decision Tree algorithm
from sklearn.model_selection import train_test_split #Data split function
from sklearn.preprocessing import LabelEncoder #OneHotEncoding
from sklearn.metrics import classification_report 
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_curve, auc
from sklearn.decomposition import PCA

#Decision tree plot
import pydotplus
from IPython.display import Image
from collections import Counter


# ## Import data

# In[4]:


#import the data into a Pandas dataframe and show it
data = pd.read_csv('credit.csv')
data.head(10).style


# ## Data exploration

# In[5]:


data.shape 
# The dataset has 1000 rows (entries/instances/observations) and 32 columns (features).
data.info()


# In[6]:


data.describe().round().style


# ## Data visualization

# In[6]:


ax = data['RESPONSE'].value_counts().plot(kind='bar')
ax.set_xlabel('is good applicant?')
ax.set_ylabel('Count')


# In[7]:


# plot fraud vs. non-fraud 
keys, counts = np.unique(data.RESPONSE, return_counts=True)
counts_norm = counts/counts.sum()

fig = plt.figure(figsize=(8, 5)) #specify figure size
gs = gridspec.GridSpec(1, 2, width_ratios=[3,1]) #specify relative size of left and right plot

#Absolute values
ax0 = plt.subplot(gs[0])
ax0 = plt.bar(['bad', 'good'], counts, color=['#1f77b4','#ff7f0e']) #left bar plot
ax0 = plt.title('bad vs. good applicants:\n Absolute distribution') 
ax0 = plt.ylabel('frequency')
ax0 = plt.text(['bad'], counts[0]/2, counts[0]) #add text box with count of non-fraudulent cases
ax0 = plt.text(['good'], counts[1]/2, counts[1]) #add text box with count of fraudulent cases

#Normalized values
ax1 = plt.subplot(gs[1])
ax1 = plt.bar(['Data'], [counts_norm[0]], label='bad')
ax1 = plt.bar(['Data'], [counts_norm[1]], bottom=counts_norm[0], label='good')
ax1 = plt.legend(bbox_to_anchor=(1, 1))
ax1 = plt.title('bad vs. good:\n Relative distribution')
ax1 = plt.ylabel('frequency')
ax1 = plt.text(['Data'],counts_norm[0]/2, '{}%'.format((counts_norm[0]*100).round(1)))
ax1 = plt.text(['Data'],(counts_norm[1]/2)+counts_norm[0], '{}%'.format((counts_norm[1]*100).round(1)))

plt.tight_layout()
plt.show()


# In[8]:


ax = data.groupby(['RESPONSE', 'EMPLOYMENT'])['EMPLOYMENT'].count().unstack().plot.bar() 
#present employment since


# In[9]:


ax = data.groupby(['RESPONSE', 'JOB'])['JOB'].count().unstack().plot.bar() 
# 0 : unemployed/ unskilled  - non-resident
# 1 : unskilled - resident
# 2 : skilled employee / official
# 3 : management/ self-employed/highly qualified employee/ officer


# ## Data preprocessing

# In[10]:


data.corr().style


# In[9]:


# X, y = data[['CHK_ACCT', 'DURATION', 'HISTORY','NEW_CAR','USED_CAR','RADIOTV','AMOUNT',
#              'SAV_ACCT','EMPLOYMENT','REAL_ESTATE','PROP_UNKN_NONE','OTHER_INSTALL','OWN_RES']], data['RESPONSE'] #define feature matrix X and labels y
# X.head()

X, y = data.loc[:, data.columns != 'RESPONSE'], data['RESPONSE'] #define feature matrix X and labels y


# ## Data split

# In[10]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1234) #split data 70:30


# In[11]:


train_dist = y_train.value_counts() / len(y_train) #normalize absolute count values for plotting
test_dist = y_test.value_counts() / len(y_test)
data_dist = data['RESPONSE'].value_counts() / len(data)

fig, ax = plt.subplots()

ax.barh(['Test','Train','Data'], [test_dist[0], train_dist[0], data_dist[0]], color='#1f77b4', label='not good')
ax.barh(['Test','Train','Data'], [test_dist[1], train_dist[1], data_dist[1]], left=[test_dist[0], train_dist[0], data_dist[0]], color='#ff7f0e', label='is good')
ax.set_title('Split visualization')
ax.legend(loc='upper left')
plt.xlabel('Proportion')
plt.ylabel('Partition')

#plot bar values
for part, a, b in zip(['Test', 'Train','Data'], [test_dist[0], train_dist[0], data_dist[0]], [test_dist[1], train_dist[1], data_dist[1]]):
    plt.text(a/2, part, str(np.round(a, 2)))
    plt.text(b/2+a, part, str(np.round(b, 2)));


# ## Build an (unbalanced) Decision Tree model

# In[14]:


#Define Decision tree classifier with some default parameters
clf = tree.DecisionTreeClassifier(criterion = "gini", random_state = 100,
                               max_depth=3, min_samples_leaf=3)

#Fit the training data 
clf.fit(X_train, y_train) #what do we need here?


# In[15]:


#Use classifier to predict labels
y_pred = clf.predict(X_test) #what do we need here?


# In[16]:


y_pred


# In[17]:


#probabilities
y_pred_probs = clf.predict_proba(X_test)


# In[18]:


'''
The graphviz library is used to visualize the tree. 
'''

#Decision tree plot
import pydotplus
from IPython.display import Image 

# Create DOT data
dot_data = tree.export_graphviz(clf, out_file=None, 
                                feature_names=X_train.columns, 
                                class_names=['not good', 'good'], filled=True) #or use y_train.unique()

# Draw graph
graph = pydotplus.graph_from_dot_data(dot_data)  

# Show graph
Image(graph.create_png())

# Create PNG 
#graph.write_png("clf.png") #uncomment this line to save the plot as a .png file


# ## Rebalancing with SMOTE

# In[12]:


smote = SMOTE(sampling_strategy='minority')
X_sm, y_sm = smote.fit_resample(X_train, y_train) #ONLY APPLIED TO TRAINING!!!


# In[13]:


X_sm


# In[14]:


X_train


# In[20]:


train_dist = y_train.value_counts() / len(y_train) #normalize absolute count values for plotting
test_dist = y_test.value_counts() / len(y_test)
data_dist = y.value_counts() / len(y)
smote_dist = pd.Series(y_sm).value_counts() / len(pd.Series(y_sm))

fig, ax = plt.subplots()

ax.barh(['X_train (SMOTE)','Test','Train','Data'], [smote_dist[0], test_dist[0], train_dist[0], data_dist[0]], color='#1f77b4', label='0 (no)')
ax.barh(['X_train (SMOTE)','Test','Train','Data'], [smote_dist[1], test_dist[1], train_dist[1], data_dist[1]], left=[smote_dist[0], test_dist[0], train_dist[0], data_dist[0]], color='#ff7f0e', label='1 (yes)')
ax.set_title('Split visualization')
ax.legend(loc='upper left')
plt.xlabel('Proportion')
plt.ylabel('Partition')

#plot bar values
for part, a, b in zip(['X_train (SMOTE)', 'Test', 'Train','Data'], [smote_dist[0], test_dist[0], train_dist[0], data_dist[0]], [smote_dist[1], test_dist[1], train_dist[1], data_dist[1]]):
    plt.text(a/2, part, str(np.round(a, 2)))
    plt.text(b/2+a, part, str(np.round(b, 2)));


# ## Build a balanced Decision Tree model

# In[21]:


clf_b = tree.DecisionTreeClassifier(criterion = "gini", random_state = 100,
                               max_depth=3, min_samples_leaf=3)

#Fit the training data 
clf_b.fit(X_sm, y_sm) #what do we need here?


# In[22]:


#Use classifier to predict labels
y_pred_b = clf_b.predict(X_test) #what do we need here?


# In[23]:


y_pred_b


# In[24]:


#probabilities
y_pred_probs_b = clf_b.predict_proba(X_test)


# In[25]:


'''
The graphviz library is used to visualize the tree. 
'''

#Decision tree plot
import pydotplus
from IPython.display import Image 

# Create DOT data
dot_data = tree.export_graphviz(clf, out_file=None, 
                                feature_names=X_train.columns, 
                                class_names=['not good', 'good'], filled=True) #or use y_train.unique()

# Draw graph
graph = pydotplus.graph_from_dot_data(dot_data)  

# Show graph
Image(graph.create_png())

# Create PNG 
#graph.write_png("clf.png") #uncomment this line to save the plot as a .png file


# ## Model evaluation
# 
# ### 1. Confusion Matrix
# ### 2. ROC and AUC
# ### 3. Expected value framework (Excel)

# In[26]:


print ("Accuracy is: ", (accuracy_score(y_test,y_pred)*100).round(2))


# In[27]:


print ("Accuracy is: ", (accuracy_score(y_test,y_pred_b)*100).round(2))


# In[58]:


#confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    #else:
    #    print('Confusion matrix, without normalization')

    #print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylim([1.5, -0.5]) #added to fix a bug that causes the matrix to be squished
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[59]:


# Compute confusion matrix
class_names = ['not good', 'good']
cnf_matrix_original = confusion_matrix(y_test, y_pred)
cnf_matrix_balanced = confusion_matrix(y_test, y_pred_b)
np.set_printoptions(precision=2)
##imbalanced
# Plot non-normalized confusion matrix
plt.figure(figsize=(13, 5))
plt.subplot(121) 
plot_confusion_matrix(cnf_matrix_original, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.subplot(122) 
plot_confusion_matrix(cnf_matrix_original, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')
##balanced
# Plot non-normalized confusion matrix
plt.figure(figsize=(13, 5))
plt.subplot(121) 
plot_confusion_matrix(cnf_matrix_balanced, classes=class_names,
                      title='Confusion matrix (balanced), without normalization')

# Plot normalized confusion matrix
plt.subplot(122) 
plot_confusion_matrix(cnf_matrix_balanced, classes=class_names, normalize=True,
                      title='Normalized confusion matrix (balanced)')


plt.show()


# In[60]:


#AUC and ROC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_probs[:,1])
roc_auc = auc(fpr, tpr)
print("AUC score on Testing: " + str(roc_auc))


# In[61]:


fpr, tpr, thresholds = roc_curve(y_test, y_pred_probs_b[:,1])
roc_auc = auc(fpr, tpr)
print("AUC score on Testing: " + str(roc_auc))


# In[62]:


fig, axs = plt.subplots(1,1, figsize=(10,8))

plt.title('ROC (Receiver Operating Characteristic)')
plt.plot(fpr, tpr, 'b', label='AUC = %0.4f'% roc_auc)
plt.legend(loc='best')
plt.plot([0,1],[0,1],color='black', linestyle='--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.ylabel('True Positive Rate (TPR)')
plt.xlabel('False Positive Rate (FPR)');


# In[63]:


plt.figure(figsize=(12,10))

for test, pred, name in zip([y_test, y_test], [y_pred_probs[:,1], y_pred_probs_b[:,1]], ['Decision Tree', 'Decision Tree balanced (SMOTE)']):
    fpr, tpr, _ = roc_curve(test, pred)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label='{}: AUC = {}'.format(name, round(roc_auc, 3)))
    plt.legend(loc='best')
    plt.plot([0,1],[0,1],color='black', linestyle='--')

plt.title('ROC curve (Receiver Operating Characteristic)')    
plt.ylabel('True Positive Rate (TPR)')
plt.xlabel('False Positive Rate (FPR)')

plt.show()


# After comparing two modelsin Excel, I would suggest Decision tree with balanced data. Based on calculation, the expected benefit witht est set priors is $27, which is higher than the other one with $5.67
