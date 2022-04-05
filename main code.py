import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import os

#%%
# Reading the data

df = pd.read_csv('./Project/141b_final_project/unemploymentdata.csv')
df

#%%
# VISUALIZATION CODE!

#%%
# splitting the data into a training and testing set
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
x = df.iloc[::,2:]
y = df['Unemployment']
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.27,random_state=0)

#%%
# fitting the full model
fullmodel = sm.OLS(y_train, x_train).fit()
fullmodel.summary()

#%%
# prediction accuracy:

yhat_test = fullmodel.predict(x_test)
MSE = sum((y_test.values - yhat_test.values)**2)/len(y_test)

#%%
# Residual plots

#%%
# Reduced model

reducedmodel = sm.OLS(y_train,x_train[['EmployServ','EmployInd','Education']]).fit()
reducedmodel.summary()

#%%
# prediction accuracy

yhat_test_reduced = fullmodel.predict(x_test)
MSE = sum((y_test.values - yhat_test_reduced.values)**2)/len(y_test)
MSE

#%%

#full pca to see how many PCs to pick
scaler = StandardScaler()
std_df = scaler.fit_transform(df)
pca = PCA()
pca.fit(std_df)
pca_variance =  pca.explained_variance_ratio_
cum_sum_eigenvalues = np.cumsum(pca_variance)

plt.bar(list(range(1,10)), pca_variance, align="center", label="Individual variance")
plt.step(list(range(1,len(cum_sum_eigenvalues)+1)), cum_sum_eigenvalues, where='mid',label='Cumulative explained variance')
plt.legend()
plt.ylabel("Variance ratio")
plt.xlabel("Principal components")
plt.title("Variance captured by the principle components")
plt.savefig("pca variance explained plot")
plt.show()

#%%

# the first four principle components for the analysis and transform the data
x_full = pd.concat([x_train,x_test])
std_df = scaler.fit_transform(x_full)
pca4 = PCA(n_components=4)
pca4.fit(x_full)
x_4d = pca4.transform(x_full)

#%%
# replit
p_train = x_4d[:53,::]
p_test = x_4d[53:,::]

#%%

# regression with the transformed data
#pca_model = sm.OLS(y_train, p_train).fit()
pca_model = sm.OLS(y_train, sm.add_constant(p_train)).fit()
pca_model.summary()

#%%
yhat_test_pca = pca_model.predict(sm.add_constant(p_test))
MSE = sum((y_test.values - yhat_test_reduced.values)**2)/len(y_test)
MSE

#%%
# looking at the loadings
loadings = pd.DataFrame(pca4.components_.T, index=df.columns[2:])

loadings
