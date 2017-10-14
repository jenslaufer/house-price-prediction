import pandas as pd
from plotnine import *
from plotnine.data import *
from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model
from altair import Chart, X

df = pd.read_csv('../input/train.csv')

# Plots of variables
#chart = Chart(df).mark_bar().encode(
#    x=X('LotFrontage', bin=True),
#    y='count(*):Q',
#)
chart = Chart(df).mark_point().encode(
    x='LotArea',
    y='SalePrice',
)
html = chart.to_html()

with open('chart.html', 'w') as f:
    f.write(html)

df.fillna(0, inplace=True)
ggplot(df, aes('LotFrontage')) + geom_histogram()

# Data Cleaning
#df.fillna(0, inplace=True)
df.fillna(0, inplace=True)


keys = list(df.keys())

quantitative_keys = ['MSSubClass', 'LotFrontage', 'LotArea',
                     'OverallQual', 'OverallCond', 'YearBuilt',
                     'YearRemodAdd',
                     'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2',
                     'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF',
                     '2ndFlrSF', 'LowQualFinSF',
                     'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath',
                     'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr',
                     'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt',
                     'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',
                     'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea',
                     'MiscVal', 'MoSold', 'YrSold'
                     ]

qualitative_keys = keys
for key in quantitative_keys:
    qualitative_keys.remove(key)


# Feature Engineering

# Feature Selection

y = df['SalePrice']
X = df.drop('SalePrice', axis=1)
X = df.drop(qualitative_keys, axis=1)

import seaborn as sns

# _r reverses the normal order of the color map 'RdYlGn'
#sns.heatmap(X, cmap='RdYlGn_r', linewidths=0.5, annot=True)
# sns.plt.show()


X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.33, random_state=42)


reg = linear_model.LinearRegression()

# Train the model using the training sets
reg.fit(X_train, y_train)

train_score = reg.score(X_train, y_train)


test_score = reg.score(X_test, y_test)
print "Training set Accuracy: {} Test set Accuracy: {} ".format(train_score, test_score)


from sklearn.metrics import mean_squared_error
predictions = reg.predict(X_test)
print 'RMSE is: {}'.format(mean_squared_error(y_test, predictions))