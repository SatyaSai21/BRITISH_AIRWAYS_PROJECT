#!/usr/bin/env python
# coding: utf-8

# In[1]:


from bs4 import BeautifulSoup
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import nltk
from nltk.corpus import stopwords


# In[2]:


import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import seaborn as sns


# In[4]:


df= pd.read_csv("C:\\Users\\Sathya Sai\\Downloads\\customer_booking.csv", encoding="ISO-8859-1")
df.head()


# In[5]:


df["booking_complete"].value_counts()


# In[6]:


df.dtypes


# In[7]:


df.info()


# In[8]:


df.info(memory_usage='deep')


# In[9]:


#data cleaning


# In[10]:


df["flight_day"]=df["flight_day"].map({"Sat":6,"Wed":3,"Tue":2,"Mon":1,"Thu":4,"Fri":5,"Sun":7})


# In[11]:


df["flight_day"].value_counts()


# In[12]:


df.isna().sum() #data is clean


# In[13]:


df.select_dtypes(include=['object'])


# In[14]:


df["sales_channel"].value_counts()


# In[15]:


df[(df["sales_channel"]=="Mobile") & (df["booking_complete"]==0)]


#  # majority of the internet users and Mobile users are not completing the booking.Incomplete bookings via internet------------>37513.  No. of total bookings via internet---->44382.Incomplete bookings via Mobile------------>5009.  No. of total bookings via internet---->5618.This can be a useful.

# In[16]:


df["trip_type"].value_counts() # mojority are roundtrips


# In[17]:


plt.hist(temp_d["flight_day"],rwidth=0.96)
plt.xlabel('DAY')
plt.ylabel('COUNT OF PASSENGERS')
plt.title('PASSENGERS TRAVELLING ON EACH DAY')
plt.show()


# In[18]:


plt.figure(figsize=(8,8))
fig,ax=plt.subplots(1,2)
ax[0].hist(temp_d["trip_type"])
ax[0].set_title('TYPE OF TRIP')
ax[1].pie(temp_d["sales_channel"].value_counts(),labels=["Internet","Mobile"],autopct="%.2f%%",explode=[0.1,0.1])
ax[1].set_title('Different Sales Channels')
plt.show()


# In[19]:


sns.countplot(x=temp_d["num_passengers"],data=temp_d,hue=temp_d["booking_complete"])
plt.title("GROUP OR FAMILY_SIZE")
plt.show()


# In[20]:


sns.scatterplot(x=temp_d["wants_extra_baggage"],y=temp_d["booking_complete"])
plt.show()


# In[21]:


temp_d.agg(['mean','max','min','std','count'])


# In[22]:


index_c=(temp_d["booking_origin"].value_counts()[temp_d["booking_origin"].value_counts()>5]).index#dele[dele["booking_origin"]=="others"]


# In[23]:


index_c=index_c
for i in temp_d.index:
    if temp_d.loc[i,"booking_origin"] not in index_c:
        temp_d.loc[i,"booking_origin"]="others"
        
     #   print(temp_d.loc[i,"booking_origin"])
      #  temp_d.loc[i,"booking_origin"]="others"


# In[24]:


temp_d[temp_d["booking_origin"]=="others"]


# In[38]:


plt.figure(figsize=(10,10))
plt.hist(temp_d["booking_origin"],orientation='horizontal',bins=41,rwidth=0.95)
plt.show()


# In[40]:


sns.countplot(x=temp_d["wants_preferred_seat"],hue=temp_d["booking_complete"])
plt.xticks([0,1],labels=["NO","YES"])
plt.title("CHOICE OF BOOKING CORRESPONDING TO PREFERENCE OF SEAT")
plt.show()


# In[41]:





# In[42]:


print("MEAN ={}".format(temp_d["length_of_stay"].mean()))
print("STD ={}".format(temp_d["length_of_stay"].std()))
print("upper limit ={}".format(temp_d["length_of_stay"].mean()+3*temp_d["length_of_stay"].std()))
print("lower limit ={}".format(temp_d["length_of_stay"].mean()-3*temp_d["length_of_stay"].std()))


# In[43]:


sns.countplot(x=temp_d["wants_in_flight_meals"],hue=temp_d["booking_complete"])
plt.xticks([0,1],labels=["NO","YES"])
plt.title("CHOICE OF BOOKING CORRESPONDING TO FLIGHT MEALS")
plt.show()


# In[44]:


df["route"].value_counts()


# In[45]:


df["booking_origin"].value_counts()


# In[46]:


df.drop(['sales_channel','route','trip_type','booking_origin'],axis='columns',inplace=True) #HIGH VARIANCE YET THE SAME TIME UNUSEFUL THEREFORE REMOVE THOSE FEATURES


# In[47]:


plt.figure(figsize=(16,8))
sns.heatmap(df.corr(),annot=True)


# In[48]:


df.info(memory_usage='deep') #memory uasge reduced This speeds up the operations on dataframe.


# In[49]:


from sklearn.feature_selection import SelectKBest, chi2


# In[82]:


target=df["booking_complete"]


# In[83]:


df.drop('booking_complete',axis=1,inplace=True)


# In[88]:





# In[89]:


x=SelectKBest(chi2,k=7)


# In[90]:


a=x.fit(temp_d,target1)


# In[91]:


features=x.get_feature_names_out()
features


# In[92]:


x.scores_


# In[93]:


X=pd.DataFrame(x.transform(df),columns=features)


# In[94]:


X


# In[60]:


sns.pairplot(X)


# In[62]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score


# In[95]:


from sklearn.model_selection import train_test_split 


# In[99]:


train_x,test_x,train_y,test_y=train_test_split(X,target,test_size=0.2)#


# In[154]:


def finding_best_model(train_x,train_y,test_x,test_y):
    Algorithms={
        'DecisionTree_classifier':{
            'model' : DecisionTreeClassifier(),
            'params' : {
                'criterion' :['gini', 'entropy'],
                'splitter' :['random','best'],
                'max_features':['auto','sqrt','log2']
            }
        },
        'XGB_classifier':
        {
            'model':GradientBoostingClassifier(),
            'params':{
                'loss':['exponential'],
                'criterion':['friedman_mse']     
            }
        },
        'RandomForest_Classifier':{
            'model':RandomForestClassifier(),
            'params':{
                'n_estimators':[100,125,150],
                'criterion':['gini','entropy'],
                'min_samples_split':[2,4]   
            }   
        },
        'KNeighbors_Classifier':{
            'model': KNeighborsClassifier(),
            'params':{
                'algorithm':['ball_tree', 'kd_tree','brute'],
                'n_neighbors':[3,5,7],
            }
        },
        'Gaussian_NB':{
            'model':GaussianNB(),
            'params':{}
        },
        'Bernouli_NB':{
            'model':BernoulliNB(),
            'params':{}
        },     
    }
    accuracy={}
    f1score={}
    recall={}
    precision={}
    roc_auc={}
    model={}
    for algo,parameter in Algorithms.items():
        gc=GridSearchCV(estimator=parameter['model'],param_grid=parameter['params'],cv=5)
        gc.fit(train_x,train_y)
        accuracy[algo]=gc.score(test_x,test_y)
        model[algo]=gc.best_params_
        f1score[algo]=f1_score(test_y,gc.predict(test_x))
        precision[algo]=precision_score(test_y,gc.predict(test_x))
        recall[algo]=recall_score(test_y,gc.predict(test_x))
        roc_auc[algo]=roc_auc_score(test_y,gc.predict(test_x))
        print(f"------{algo} confusion matrix--------------\n")
        print( confusion_matrix(test_y,gc.predict(test_x),labels=[0,1]))
        
    return accuracy,model,f1score,recall,precision,roc_auc
        


# In[155]:


score_dict,model_dict,f1score_dict,recall_dict,precision_dict,roc_auc_dict=finding_best_model(train_x,train_y,test_x,test_y)


# In[107]:


model_n=MultinomialNB()


# In[108]:


model_n.fit(train_x,train_y)


# In[73]:


fi=model_n.predict(test_x)


# In[76]:


model_n.score(test_x,test_y)


# In[145]:


new_m=LogisticRegression(solver='saga')


# In[146]:


new_m.fit(new_df,t_x)


# In[147]:


new_ans=new_m.predict(n_y)


# In[148]:


new_m.score(n_y,t_y)


# In[169]:


print("-----------------------{}-----------\nAccuracy = {}\n".format('Logistic Regression',0.8532))
print(classification_report(new_ans,t_y,zero_division=False)) 


# In[ ]:


score_dict,model_dict,f1score_dict,recall_dict,precision_dict,roc_auc_dict


# In[184]:


m={}
m['precision']=precision_dict


# In[185]:


precision_score=pd.DataFrame(m)


# In[187]:


score,models,f1_score,recall_score,precision_score,roc_auc_score


# In[101]:


s=StandardScaler()


# In[102]:


ad=s.fit_transform(n_x)
#h=GridSearchCV(estimator=SVC(),param_grid=params,cv=5)


# In[103]:


new_df=pd.DataFrame(ad,columns=head)


# In[77]:


head=ad.feature_names_in_


# In[55]:





# In[86]:


------DecisionTree_classifier confusion matrix--------------

[[7442 1098]
 [1125  335]]

------XGB_classifier confusion matrix--------------

[[8537    3]
 [1460    0]]

------RandomForest_Classifier confusion matrix--------------

[[8177  363]
 [1301  159]]

------KNeighbors_Classifier confusion matrix--------------
[[8400  140]
 [1406   54]]

------Gaussian_NB confusion matrix--------------

[[8475   65]
 [1449   11]]

------Bernouli_NB confusion matrix--------------

[[8540    0]
 [1460    0]]

------Logistic_Regression confusion matrix--------------

[[8540    0]
 [1460    0]]


# In[63]:


roc_auc_score


# In[68]:


score[score.scores==score.scores.max()] #best model according to accuracy.


# In[ ]:


0.8216 ------>STANDARDSCALER + KNN
             precision    recall  f1-score   support

           0       0.96      0.85      0.90      9608
           1       0.03      0.10      0.04       392

    accuracy                           0.82     10000
   macro avg       0.49      0.47      0.47     10000
weighted avg       0.92      0.82      0.87     10000
#------------------------------------------------------------------------------------------------------
0.8532 --------->STANDARDSCALER + DECISIONTREE # DON'T AFFECT MUCH

              precision    recall  f1-score   support

           0       1.00      0.85      0.92     10000
           1       0.00      0.00      0.00         0

    accuracy                           0.85     10000
   macro avg       0.50      0.43      0.46     10000
weighted avg       1.00      0.85      0.92     10000
#------------------------------------------------------------------------------------------------------
0.8532-------------->STANDARDSCALER + GAUSSIANNB()
             precision    recall  f1-score   support

           0       1.00      0.85      0.92     10000
           1       0.00      0.00      0.00         0

    accuracy                           0.85     10000
   macro avg       0.50      0.43      0.46     10000
weighted avg       1.00      0.85      0.92     10000

#------------------------------------------------------------------------------------------------------
0.8532---------->STANDARDSCALER + MULTINOMIALNB()
 precision    recall  f1-score   support

           0       1.00      0.85      0.92     10000
           1       0.00      0.00      0.00         0

    accuracy                           0.85     10000
   macro avg       0.50      0.43      0.46     10000
weighted avg       1.00      0.85      0.92     10000

#------------------------------------------------------------------------------------------------------
0.8529--------->STANDARDSCALER + RandomForestClassifier(min_samples_split=4, n_estimators=150) # DON'T AFFECT MUCH
     precision    recall  f1-score   support

           0       1.00      0.85      0.92      9989
           1       0.00      0.36      0.01        11

    accuracy                           0.85     10000
   macro avg       0.50      0.61      0.46     10000
weighted avg       1.00      0.85      0.92     10000

#------------------------------------------------------------------------------------------------------
0.8532--------->STANDARDSCALER + LogisticRegression(solver='saga')
        precision    recall  f1-score   support

           0       1.00      0.85      0.92     10000
           1       0.00      0.00      0.00         0

    accuracy                           0.85     10000
   macro avg       0.50      0.43      0.46     10000
weighted avg       1.00      0.85      0.92     10000


# In[105]:


models.values


# # CLASSIFICATION REPORT OF EACH ALGORITHM

# In[157]:


print("----------"+str(mod)+"---------\n")
print(classification_report(test_y,fi,zero_division=0))


# In[156]:


print("----------"+str(g)+"---------\n")
print(classification_report(test_y,ans,zero_division=0))


# In[150]:


print("----------"+str(g)+"---------\n")
print(classification_report(test_y,ans,zero_division=0))


# In[143]:


print("----------"+str(g)+"---------\n")
print(classification_report(test_y,ans,zero_division=0))


# In[135]:


print("----------"+str(g)+"---------\n")
print(classification_report(test_y,ans,zero_division=0))
        precision    recall  f1-score   support


# In[113]:


print("----------"+str(g)+"---------\n")
print(classification_report(test_y,ans,zero_division=0))


# In[99]:


print("----------"+str(g)+"---------\n")
print(classification_report(test_y,ans,zero_division=0))


# In[94]:


print("----------"+str(g)+"---------\n")
print(classification_report(test_y,ans,zero_division=0))


# # BEST MODEL ARE MULTINOMIAL NAIVE BAYES,
# 
# # KNeighborsClassifier(),
# 
# # DecisionTreeClassifier(max_features='log2', splitter='random'),
# 
# # RandomForestClassifier(min_samples_split=4, n_estimators=150).

# # BUT  TO  REDUCE  COMPUTATION  WE  CAN  CHOOSE   MULTINOMIAL NAIVE  BAYES  (AS  KNN  CLASSIFIER   IS  A  SLOW  LEARNER).

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




