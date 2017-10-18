# House Prediction Model

```{.python .input}
import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn import datasets
from sklearn.preprocessing import LabelEncoder

from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoLars
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
```

```{.python .input}
train_df = pd.read_csv('../input/train.csv')

train_df = train_df[train_df.GrLivArea < 4700]


test_df = pd.read_csv('../input/test.csv')
test_id = test_df['Id']

Y_train = train_df['SalePrice']
Y_train = np.log1p(Y_train)
train_df.drop(['SalePrice'], inplace=True, axis=1)

df = train_df.append(test_df, ignore_index=True)
df.drop(['Id'], inplace=True, axis=1)

X = df
X['Alley'].fillna('NA', inplace=True)
X['PoolQC'].fillna('NP', inplace=True)
X['Fence'].fillna('NF', inplace=True)
X['MasVnrType'].fillna('None', inplace=True)
X['BsmtQual'].fillna('NB', inplace=True)
X['BsmtCond'].fillna('NB', inplace=True)
X['BsmtExposure'].fillna('NB', inplace=True)
X['BsmtFinType1'].fillna('NB', inplace=True)
X['BsmtFinType2'].fillna('NB', inplace=True)
X['Electrical'].fillna('SBrkr', inplace=True)
X['FireplaceQu'].fillna('NF', inplace=True)
X['GarageType'].fillna('NG', inplace=True)
X['GarageYrBlt'].fillna(0, inplace=True)
X['GarageFinish'].fillna('NG', inplace=True)
X['GarageQual'].fillna('NG', inplace=True)
X['GarageCond'].fillna('NG', inplace=True)
X['MSZoning'].fillna('RL', inplace=True)
X['LotFrontage'].fillna(X['LotFrontage'].mean(), inplace=True)
X['MasVnrArea'].fillna(0, inplace=True)
X['Utilities'].fillna('AllPub', inplace=True)
X['Exterior1st'].fillna('VinylSd', inplace=True)
X['Exterior2nd'].fillna('VinylSd', inplace=True)
X['BsmtFinSF1'].fillna(X['BsmtFinSF1'].mean(), inplace=True)
X['BsmtFinSF2'].fillna(X['BsmtFinSF2'].mean(), inplace=True)
X['BsmtUnfSF'].fillna(X['BsmtUnfSF'].mean(), inplace=True)
X['TotalBsmtSF'].fillna(X['TotalBsmtSF'].mean(), inplace=True)
X['BsmtFullBath'].fillna(X['BsmtFullBath'].mean(), inplace=True)
X['BsmtHalfBath'].fillna(X['BsmtHalfBath'].mean(), inplace=True)
X['KitchenQual'].fillna('TA', inplace=True)
X['Functional'].fillna('Typ', inplace=True)
X['GarageCars'].fillna(2, inplace=True)
X['GarageArea'].fillna(X['GarageArea'].mean(), inplace=True)
X['SaleType'].fillna('WD', inplace=True)



X.drop(['MiscFeature'], inplace=True, axis=1)
X.drop(['PoolQC'], inplace=True, axis=1)
X.drop(['Alley'], inplace=True, axis=1)
X.drop(['Fence'], inplace=True, axis=1)
X.drop(['FireplaceQu'], inplace=True, axis=1)
X.drop(['LotFrontage'], inplace=True, axis=1)


# label and one-hot encode data
cat_cols = X.dtypes.index[X.dtypes == 'object']
cont_cols = X.dtypes.index[X.dtypes != 'object']
for column in cat_cols:

	le = LabelEncoder()
	X[column] = le.fit_transform(X[column])

	one_hot = pd.get_dummies(X[column], prefix=column)
	X.drop(column, axis=1, inplace=True)
	X = X.join(one_hot)

# replace numerical columns
#X['LotFrontage'].fillna(X['LotFrontage'].mean(), inplace=True)
X['MasVnrArea'].fillna(0, inplace=True)

# Skew correct numerical columns
X['MSSubClass'] = np.log1p(X['MSSubClass'])
#X['LotFrontage'] = np.log1p(X['LotFrontage'])
X['LotArea'] = np.log1p(X['LotArea'])
X['OverallCond'] = np.log1p(X['OverallCond'])
X['MasVnrArea'] = np.log1p(X['MasVnrArea'])
X['BsmtFinSF1'] = np.log1p(X['BsmtFinSF1'])
X['BsmtFinSF2'] = np.log1p(X['BsmtFinSF2'])
X['BsmtUnfSF'] = np.log1p(X['BsmtUnfSF'])
X['TotalBsmtSF'] = np.log1p(X['TotalBsmtSF'])
X['1stFlrSF'] = np.log1p(X['1stFlrSF'])
X['2ndFlrSF'] = np.log1p(X['2ndFlrSF'])
X['GrLivArea'] = np.log1p(X['GrLivArea'])
X['WoodDeckSF'] = np.log1p(X['WoodDeckSF'])
X['OpenPorchSF'] = np.log1p(X['OpenPorchSF'])
X['EnclosedPorch'] = np.log1p(X['EnclosedPorch'])
X['ScreenPorch'] = np.log1p(X['ScreenPorch'])
X['MiscVal'] = np.log1p(X['MiscVal'])

X_train = X[:train_df.shape[0]]
```

```{.python .input}
pipe = Pipeline([
    ('reduce_dim', None),
    ('regression', None)
])


refcv_params = {
    'reduce_dim': [RFECV(LinearRegression(), step=1, cv=StratifiedKFold(2))]
}

kbest_params = {
    'reduce_dim': [SelectKBest()],
    'reduce_dim__k': list(range(100, 150, 1))
}

pca_params = {
    'reduce_dim': [PCA(iterated_power=7)],
    'reduce_dim__whiten': [False],
    'reduce_dim__copy': [True, False]
}


linear_regression_params = {
    'regression': [LinearRegression()]
}


ridge_params = {
    'regression': [Ridge()],
    'regression__alpha': [1, 5, 8, 10, 20, 30]
}


lasso_params = {
    'regression': [Lasso()],
    'regression__alpha': [0.1, 0.5, 1]
}

lasso_lars_params = {
    'regression': [LassoLars()],
    'regression__alpha': [0.1, 0.5, 1, 2, 10, 20]
}





reducer_params = [pca_params]
classifier_params = [ridge_params]

param_grid = []
for classifier_param in classifier_params:
    for reducer_param in reducer_params:
        params = dict(reducer_param.items() + classifier_param.items())
        param_grid.append(params)


grid = GridSearchCV(pipe, param_grid=param_grid, cv=KFold(n_splits=5, random_state=42), 
                    scoring='neg_mean_squared_error')

grid.fit(X_train, Y_train)

print(math.sqrt(-grid.best_score_))
print(grid.best_estimator_)



predicted = grid.predict(X_train)
X_test = X[train_df.shape[0]:]
test_pred = np.expm1(grid.predict(X_test))

submission = pd.DataFrame(test_pred, index=test_id.values, columns=['SalePrice'])
submission.index.name = 'Id'
submission.to_csv('submission.csv')
```
