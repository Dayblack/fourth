#!/usr/bin/env python
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv(r"C:\Users\ASUS\Downloads\train_ctrUa4K.csv")


# In[3]:


df.head(10)


# In[4]:


df.describe()


# In[5]:


df["Property_Area"].value_counts()


# In[6]:


df["ApplicantIncome"].hist(bins=100)


# In[7]:


df.boxplot(column="ApplicantIncome",by = "Education")


# In[8]:


df['LoanAmount'].hist(bins=50)


# In[9]:


df.boxplot(column='LoanAmount')


# In[10]:


#分类变量分析
#数据透视与交叉表格
temp1 = df["Credit_History"].value_counts(ascending=True)
temp2 = df.pivot_table(values="Loan_Status",index = ["Credit_History"],aggfunc=lambda x:x.map({"Y":1,"N":0}).mean())
print(temp1)
print(temp2)
temp3 = df.pivot_table(values=["Married"],index = ["Credit_History"],aggfunc=lambda x:x.map({"Yes":1,"No":0}).mean())
print(temp3)

temp4 = pd.crosstab(df['Credit_History'], df['Loan_Status'])
print(temp4)
temp5 = pd.pivot_table(df,values =[ 'Loan_Status'],index = ["Married","Credit_History"],aggfunc=[lambda x:x.map({"Y":1,"N":0}).mean()])
print(temp5)


# In[11]:


fig = plt.figure(figsize=(8,4),dpi = 100)
temp1.plot(kind='bar')
temp2.plot(kind = 'bar')
temp3.plot(kind="bar")
temp4.plot(kind="bar",stacked=False,color=['red','blue'],grid=True)
temp4.plot(kind="bar",stacked=True,color=['red','blue'],grid=False)


# In[ ]:





# In[12]:


# tmpd7 = pd.crosstab([df["Education"],df["Self_Employed"]],df['LoanAmount'])
# print(tmpd7)
# tmpd7.plot(kind = "box")
# tmpd8 = pd.pivot_table(df,index=["Education","Self_Employed"],values =['LoanAmount'])
# print(tmpd8)


# In[13]:


df.boxplot(column="LoanAmount",by = ["Education","Self_Employed"])


# In[14]:


df['Self_Employed'].fillna('No',inplace=True)


# In[15]:


#数据转换
#1.有些变量的值丢失了，我们应该根据缺失值的数量和变量的预期重要性来估计这些值。
table = df.pivot_table(values='LoanAmount', index='Self_Employed' ,columns='Education', aggfunc=np.median)
print(table)
# Define function to return value of this pivot_table
def fage(x):
    return table.loc[x['Self_Employed'],x['Education']]
# Replace missing values
df['LoanAmount'].fillna(df[df['LoanAmount'].isnull()].apply(fage, axis=1), inplace=True)


# In[16]:


table.loc["No","Graduate"]


# In[17]:


#2.处理极值
df['LoanAmount_log'] = np.log(df['LoanAmount'])
df['LoanAmount_log'].hist(bins=20)

df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
df['TotalIncome_log'] = np.log(df['TotalIncome'])
df['LoanAmount_log'].hist(bins=30)


# In[18]:


#构建预测模型
#填充数据集中丢失的所有值
df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
df['Married'].fillna(df['Married'].mode()[0], inplace=True)
df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)
df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)


# In[19]:


from sklearn.preprocessing import LabelEncoder


# In[20]:


var_mod = ['Gender','Married','Dependents','Education','Self_Employed','Property_Area','Loan_Status']
le = LabelEncoder()


# In[21]:


for i in var_mod:
    df[i] = le.fit_transform(df[i])
df.dtypes


# In[ ]:


#为保证每次循环的训练集与验证集的有效性，保持每次循环训练集与验证集数据结构的一致
#为了模型的有效性，只需保证验证集数据分布结构与测试集数据分布结构一致，即可认为通过验证集选出的模型与超参数对检验集也有提升作用
#针对非平衡数据分层采样使用StratifiedKFold
# from sklearn.model_selection import StratifiedKFold
# X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
# y = np.array([0, 0, 1, 1])
# # X is the feature set and y is the target
# skf = StratifiedKFold(n_splits=2,random_state = None)
# print(skf.get_n_splits(X, y))
# print(skf)

# for train_index, test_index in skf.split(X, y):
#     print("TRAIN:", train_index, "TEST:", test_index)



#重复Kfold,n_repeats = n（n表示在不重复的情况下随机划分的次数）,n_splits = k
# from sklearn.model_selection import RepeatedKFold
# X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
# y = np.array([0, 0, 1, 1])
# rkf = RepeatedKFold(n_splits=4, n_repeats=4, random_state=2652124)
# for train_index, test_index in rkf.split(X):
#     print("TRAIN:", train_index, "TEST:", test_index)
#     X_train, X_test = X[train_index], X[test_index]
#     y_train, y_test = y[train_index], y[test_index]



#Leave-One-Out Cross Validation
# from sklearn.model_selection import LeaveOneOut
# import numpy as np
# X = np.array([[1, 2], [3, 4],[5,6],[7, 8]])
# y = np.array([1, 2, 2, 1])
# loo = LeaveOneOut()
# loo.get_n_splits(X)
# print(loo.get_n_splits(X))
# for train_index, test_index in loo.split(X):
#         print("train:", train_index, "validation:", test_index)
#         X_train, X_test = X[train_index], X[test_index]
#         y_train, y_test = y[train_index], y[test_index]



#对抗验证，适用于测试集与训练集分布存在显著差异
#一般根据特征分布来检查 训练和测试之间的相似度
#对抗验证就是要，根据测试集的数据分布状况，来选择与测试样本数据分布一致的训练样本，进而拿到数据分布一致的验证集
#如何达到这个过程：1.合并训练集和测试集，并且将训练集和测试集的标签分别设置为0和1；
#2.构建一个分类器（CNN,RNN或者决策树等），用于学习the different between testing and training data
#3.找到训练集中与测试集最相似的样本（most resemble data），作为验证集，其余的作为训练集
#4.构建一个用于训练的模型（CNN,RNN或者决策树等）


# In[22]:


#用于训练集的数据越多，bias越小，f（^x）越接近f(x)
#
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier,export_graphviz
from sklearn import metrics







# In[40]:


#Import models from scikit learn module:
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier,export_graphviz
from sklearn import metrics


#建立分类模型与性能函数
def classification_model(model, data, predictors, outcome):
    #用训练集拟合模型
    model.fit(data[predictors],data[outcome])
    #对训练集进行预测
    predictions = model.predict(data[predictors])
    
    accuracy = metrics.accuracy_score(predictions,data[outcome])
    print ("Accuracy : %s" % "{0:.3%}".format(accuracy))
    
    #进行5次k-folds交叉验证
    kf = KFold(shuffle=False,n_splits=5,random_state=None)
    error = []
    for train_index, test_index in kf.split(data[predictors]):
        # 过滤训练数据
        train_predictors = data[predictors].iloc[train_index]
        #训练算法的target
        train_target = data[outcome].iloc[train_index]
        #训练算法
        model.fit(train_predictors, train_target)
        #记录 every folds 误差
        error.append(model.score(data[predictors].iloc[test_index], data[outcome].iloc[test_index]))
    print ("Cross-Validation Score : %s" % "{0:.3%}".format(np.mean(error)))
    #再次拟合模型，以便在函数外引用
    model.fit(data[predictors],data[outcome]) 
    


# In[49]:


#Logistic Regression回归模型
#依据数据探索结果，做出假设1：申请人的信用记录将会影响贷款额2.申请人及合申请人收入较高的申请人3.申请人拥有较高的学历4.具有高增长前景的城市房地产

#创建第一个 Credit_History 模型
outcome_var = "Loan_Status"
model = LogisticRegression()
predictor_var = ['Credit_History']
classification_model(model,df,predictor_var,outcome_var)


# In[59]:


#尝试不同的变量组合
predictor_var = ['Credit_History','Education','Married','Self_Employed','Property_Area']
classification_model(model, df,predictor_var,outcome_var)


# In[60]:


#特征工程
#更好的建模 Decision Tree

model = DecisionTreeClassifier()
predictor_var = ['Credit_History','Gender','Married','Education']
classification_model(model, df,predictor_var,outcome_var)


# In[61]:


predictor_var = ['Credit_History','Loan_Amount_Term','LoanAmount_log']
classification_model(model, df,predictor_var,outcome_var)
#相较于Accuracy，交叉验证的得分减少，显示出数据过度拟合


# In[62]:


#尝试更复杂的模型Random Forest
model = RandomForestClassifier(n_estimators=100)
predictor_var = ['Gender', 'Married', 'Dependents', 'Education',
       'Self_Employed', 'Loan_Amount_Term', 'Credit_History', 'Property_Area',
        'LoanAmount_log','TotalIncome_log']
classification_model(model, df,predictor_var,outcome_var)


# In[63]:


#对于过度拟合，1.减少预测器的数量2.调整模型参数
#提取重要性特征：
featimp = pd.Series(model.feature_importances_, index=predictor_var).sort_values(ascending=False)
print (featimp)


# In[64]:


# 使用前五个变量创建模型，同时对参数修改
model = RandomForestClassifier(n_estimators=25, min_samples_split=25, max_depth=7, max_features=1)
predictor_var = ['TotalIncome_log','LoanAmount_log','Credit_History','Dependents','Property_Area']
classification_model(model, df,predictor_var,outcome_var)


# In[ ]:


#交叉验证的分数提高，表明模型的泛化能力效果很好
#说明了，1.更复杂的模型不一定能够得到更好的结果
#2.在不理解底层黑盒，算法基础的情况下，应该尽量少得使用复杂模型，避免过拟合出现而无法调整模型和解释原因
# 3.特征工程是重要的

