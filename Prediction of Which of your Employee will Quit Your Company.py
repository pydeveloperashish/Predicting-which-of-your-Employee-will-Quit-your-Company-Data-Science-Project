#!/usr/bin/env python
# coding: utf-8

# In[136]:


import numpy as np
import pandas as pd

hr_df=pd.read_csv(r'C:\Python37\Projects\ALL ML-DL-DS Projects from Udemy and other Sources\Data_Science\Data Science and Deep Learning for Business\datascienceforbusiness-master\hr_data.csv')


# In[5]:


hr_df


# In[ ]:





# In[6]:


#Numerical Analysis


# In[7]:


hr_df.shape


# In[8]:


hr_df.size


# In[ ]:





# In[9]:


hr_df.info()


# In[ ]:





# In[10]:


hr_df['department'].unique()


# In[11]:


hr_df['salary'].unique()


# In[ ]:





# In[ ]:





# In[ ]:





# In[12]:


#Loading our Employee Satisfaction Data


# In[13]:


s_df=pd.read_excel(r'C:\Python37\Projects\ALL ML-DL-DS Projects from Udemy and other Sources\Data_Science\Data Science and Deep Learning for Business\datascienceforbusiness-master\employee_satisfaction_evaluation.xlsx')


# In[14]:


s_df


# In[ ]:





# In[ ]:





# In[15]:


#Merging and Joining


# In[16]:


main_df= hr_df.set_index('employee_id').join(s_df.set_index('EMPLOYEE #'))


# In[17]:


main_df=main_df.reset_index()


# In[18]:


main_df


# In[ ]:





# In[19]:


main_df.info()


# In[ ]:





# In[ ]:





# In[20]:


main_df[main_df.isnull().any(axis=1)]


# In[ ]:





# In[ ]:





# In[21]:


main_df.describe()


# In[ ]:





# In[ ]:





# In[22]:


main_df.fillna(main_df.mean(),inplace=True)


# In[23]:


main_df[main_df.isnull().any(axis=1)]


# In[ ]:





# In[ ]:





# In[24]:


main_df.loc[main_df['employee_id']==1340]


# In[ ]:





# In[ ]:





# In[25]:


main_df.drop(columns='employee_id',inplace=True)


# In[26]:


main_df


# In[ ]:





# In[28]:


#main_df['department'].values_counts()


# In[29]:


main_df.groupby('department').sum()


# In[ ]:





# In[ ]:





# In[30]:


main_df.groupby('department').mean()


# In[ ]:





# In[ ]:





# In[31]:


main_df['left'].value_counts()


# In[ ]:





# In[ ]:





# In[32]:


#Data Visualization


# In[33]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:





# In[34]:


def plot_corr(df,size=10):
    
    corr=df.corr()
    fig,ax=plt.subplots(figsize=(size,size))
    ax.legend()
    cax=ax.matshow(corr)
    fig.colorbar(cax)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation='vertical')
    plt.yticks(range(len(corr.columns)), corr.columns)
    

plot_corr(main_df)


# In[ ]:





# In[ ]:





# In[35]:


plt.bar(x=main_df['left'],height=main_df['satisfaction_level'])


# In[36]:


sns.barplot(x='left',y='satisfaction_level',data=main_df)


# In[ ]:





# In[37]:


sns.barplot(x='promotion_last_5years',y='satisfaction_level',data=main_df,hue='left')


# In[ ]:





# In[38]:


sns.pairplot(main_df,hue='left')


# In[ ]:





# In[ ]:





# In[ ]:





# In[39]:


#Data Preprocessing


# In[40]:


y=main_df[['department','salary']]


# In[41]:


y


# In[42]:


from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

k=le.fit_transform(main_df['salary'])


# In[43]:


k


# In[44]:


main_df


# In[45]:


main_df['salary_num']=k


# In[46]:


main_df


# In[47]:


main_df.loc[main_df['salary']=='high']


# In[48]:


main_df.drop(['salary'],axis=1,inplace=True)


# In[49]:


main_df


# In[ ]:





# In[ ]:





# In[50]:


z=le.fit_transform(main_df['department'])


# In[51]:


z


# In[52]:


main_df['department_num']=z


# In[53]:


main_df


# In[54]:


main_df.loc[main_df['department']=='IT']


# In[55]:


main_df.drop(['department'],axis=1,inplace=True)


# In[56]:


main_df


# In[ ]:





# In[57]:


X=main_df.drop(['left'],axis=1)


# In[58]:


X


# In[ ]:





# In[59]:


y=main_df['left']


# In[60]:


y.size


# In[ ]:





# In[ ]:





# In[61]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,)


# In[62]:


X_test


# In[63]:


y_test


# In[64]:


# Standard Scaler


# In[ ]:





# In[65]:


# Model Classification


# In[66]:


# Decision Tree


# In[67]:


from sklearn.metrics import accuracy_score


# In[68]:


from sklearn.tree import DecisionTreeClassifier

dt=DecisionTreeClassifier()

dt.fit(X_train,y_train)


# In[69]:


prediction_dt=dt.predict(X_test)


# In[70]:


prediction_dt


# In[71]:


y_test


# In[152]:


accuracy_dt=accuracy_score(y_test,prediction_dt)*100


# In[153]:


accuracy_dt


# In[73]:


X_test


# In[ ]:





# In[ ]:





# In[74]:


Catagory=['Employee will stay','Employee will Leave']


# In[ ]:





# In[75]:


custom_dt=[[1,500,3,6,0,0.90,0.89,1,8]]


# In[76]:


print(int(dt.predict(custom_dt)))


# In[77]:


Catagory[int(dt.predict(custom_dt))]


# In[ ]:





# In[ ]:





# In[78]:


dt.feature_importances_


# In[79]:


feature_importance=pd.DataFrame(dt.feature_importances_,index=X_train.columns,columns=['Importance']).sort_values('Importance',ascending=False)


# In[80]:


feature_importance


# In[ ]:





# In[87]:


X_train


# In[ ]:





# In[82]:


#KNN


# In[83]:


# Data Processing of KNN


# In[84]:


from sklearn.preprocessing import StandardScaler


# In[86]:


sc=StandardScaler().fit(X_train)
X_train_std=sc.transform(X_train)
X_test_std=sc.transform(X_test)


# In[88]:


X_train_std


# In[90]:


X_test_std


# In[ ]:





# In[130]:


from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_std,y_train)


# In[ ]:





# In[131]:


prediction_knn=knn.predict(X_test_std)


# In[132]:


accuracy_knn=accuracy_score(y_test,prediction_knn)*100


# In[133]:


accuracy_knn


# In[134]:


prediction_knn


# In[100]:


y_test


# In[ ]:





# In[ ]:





# In[126]:


k_range=range(1,26)
scores={}
scores_list=[]


for k in k_range:
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_std,y_train)
    prediction_knn=knn.predict(X_test_std)
    scores[k]=accuracy_score(y_test,prediction_knn)*100
    scores_list.append(accuracy_score(y_test,prediction_knn))

    
    


# In[127]:


scores


# In[128]:


scores_list


# In[129]:


plt.plot(k_range,scores_list)


# In[ ]:





# In[146]:


X_test.head(1)


# In[147]:


X_knn=np.array([[20,500,10,6,0,0.10,0.30,1,8]])
X_knn_std=sc.transform(X_knn)


# In[148]:


X_knn_std


# In[149]:


X_knn_prediction=knn.predict(X_knn_std)


# In[150]:


X_knn_prediction


# In[151]:


Catagory[int(dt.predict(custom_dt))]


# In[ ]:





# In[ ]:





# In[156]:


algorithms=['Decision Tree','KNN']
scores=[accuracy_dt,accuracy_knn]
plt.xlabel("Algorithms")
plt.ylabel("Accuracy Score")
sns.barplot(algorithms,scores)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




