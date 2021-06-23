import pandas as pd

pd.set_option("display.max_rows", 50)
pd.set_option("expand_frame_repr", True)
pd.set_option('display.max_columns', 500)

import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

# importing libraries
import statsmodels.api as sm

#read data file
data = pd.read_csv(r'SE_STOREFRONT_DATA_reducced.csv', encoding='latin1')
print(data.shape)
print(data.columns)
print(data.head())
print(data.describe())

print(data['handling_fee']. mean())
count_no_sub = len(data[data['Is_Churned']==0])
count_sub = len(data[data['Is_Churned']==1])
pct_of_no_sub = count_no_sub/(count_no_sub+count_sub)
print("percentage of not churned is", pct_of_no_sub*100)
pct_of_sub = count_sub/(count_no_sub+count_sub)
print("percentage of churned", pct_of_sub*100)
pd.crosstab(data.gmv_rank,data.Is_Churned).plot(kind='bar')
plt.title('Churn Frequency for GMV Category')
plt.xlabel('gmv_rank')
plt.ylabel('Frequency of Churn')
plt.savefig('Churn')
plt.show()

df = pd.DataFrame(data,columns=['Is_Churned','handling_fee', 'total_aov','gmv_rank', '>Mean_SF_Handling_Fee', ])

 #create confusion matrix
sns.set_theme(style="white")
corrMatrix = df.corr()
sns.heatmap(corrMatrix, annot=True)
plt.show()



#define train data
Xtrain= data[[
'days_since_last_order','>_than_mean_GMV','>_Mean_Handling_Fee','>Mean_SF_Handling_Fee','haslogo','hastax','hasproducts','total_product_count','gmv_rank','total_avg_handling','total_aov','handling_fee','Handling_fee%', '>_aggr_handling_fee']]



ytrain=data['Is_Churned']
log_reg=sm.Logit(ytrain,sm.add_constant(Xtrain))
log_reg = sm.Logit(ytrain, Xtrain).fit()


#print summary output
print(log_reg.summary2())

X=Xtrain
y=ytrain

model = LogisticRegression(fit_intercept = False, max_iter=1000,penalty='none' ,C=1e9)
mdl = model.fit(X, y)
model.coef_
print(model.coef_)



#odds ratio


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X_train,X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

logisticRegr = LogisticRegression(max_iter= 1000)
logisticRegr.fit(X_train, y_train)

df=pd.DataFrame({'odds_ratio':(np.exp(logisticRegr.coef_).T).tolist(),'variable':X.columns.tolist()})
df['odds_ratio'] = df['odds_ratio'].str.get(0)

df=df.sort_values('odds_ratio', ascending=False)
print(df)
#
# correlation matrix churn and days since last order


df = pd.DataFrame(data,columns=['Is_Churned','days_since_last_order' ])

sns.set_theme(style="white")
corrMatrix = df.corr()
sns.heatmap(corrMatrix, annot=True)
plt.show()
