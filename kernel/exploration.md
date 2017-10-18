# Data Exploration House Price Dataset

```{.python .input  n=135}
import warnings
warnings.filterwarnings('ignore')
```

```{.python .input  n=136}
import pandas as pd
import numpy as np
from plotnine import *
from plotnine.data import *
from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model
from altair import Chart, X
import seaborn as sns
import matplotlib.pyplot as plt

```

```{.python .input  n=137}
%matplotlib inline
```

```{.python .input  n=138}
%%javascript
IPython.OutputArea.prototype._should_scroll = function(lines) {
    return false;
}
```

```{.python .input  n=157}
def quantitive_plot(variable_name):
    fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=False, figsize=(20,10))
    sns.distplot(df[variable_name], ax=ax1)
    sns.regplot(variable_name,'SalePrice', data=df[pd.isnull(df[variable_name])], ax=ax2)
    plt.show()
    
    
def qualitive_plot(variable_name):
    fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=False, figsize=(20,10))
    sns.countplot(df[variable_name], ax=ax1)
    sns.boxplot(variable_name,'SalePrice', data=df, ax=ax2)
    plt.show()


def var_plot(variable_name):
    if df[variable_name].dtype != np.object:
        quantitive_plot(variable_name)
    else:
        qualitive_plot(variable_name)
```

```{.python .input  n=140}
df = pd.read_csv('../input/train.csv')
```

## Data Exploration

Before modeling we need to do a exploratory data analysis to understand the data
and clean the data.

```{.python .input  n=141}
df.shape
```

The data set has 1460 observations on 81 variables.

```{.python .input  n=142}
quantitive = df.select_dtypes(exclude=['object']).drop(['Id', 'SalePrice'], axis=1).keys()
qualitive = df.select_dtypes(include=['object']).keys()
```

Quanitative variables:

```{.python .input  n=143}
quantitive
```

Qualitative variables:

```{.python .input  n=144}
qualitive
```

### Data Cleaning

```{.python .input  n=145}
missing = df.isnull().sum()
missing = missing[missing > 0]
missing.sort_values(inplace=True)
missing
```

I check the values that occur for the different values:

```{.python .input  n=18}
df.PoolQC.fillna('NoPool', inplace=True)
df.MiscFeature.fillna('None', inplace=True)
df.Alley.fillna('NoAlley', inplace=True)
df.Fence.fillna('NoFence', inplace=True)
df.FireplaceQu.fillna('NoFireplace', inplace=True)
df.LotFrontage.fillna(0, inplace=True)
df.GarageCond.fillna('TA', inplace=True)
df.GarageType.fillna('Attchd', inplace=True)
df.GarageFinish.fillna('Unknown', inplace=True)
df.GarageQual.fillna('TA', inplace=True)
df.BsmtExposure.fillna('No', inplace=True)
df.BsmtFinType2.fillna('Unknown', inplace=True)
df.BsmtFinType1.fillna('Unknown', inplace=True)
df.BsmtCond.fillna('TA', inplace=True)
df.BsmtQual.fillna('None', inplace=True)
df.MasVnrArea.fillna(0, inplace=True)
df.MasVnrType.fillna('None', inplace=True)
df.Electrical.fillna('SBrkr', inplace=True)
```

We replace the missing year when a garage was build with

```{.python .input  n=19}
df.GarageYrBlt.fillna(df.YearBuilt, inplace=True)
```

#### Outliers

```{.python .input  n=20}
df_quantitive = df.select_dtypes(exclude=['object'])
df_quantitive.drop('Id', axis=1, inplace=True)
df_qualitive = df.select_dtypes(include=['object'])
```

```{.python .input  n=21}
f = pd.melt(df_quantitive, value_vars=df_quantitive.keys())

g = sns.FacetGrid(f, col="variable",  col_wrap=2, sharex=False, sharey=False, size=6)
g = g.map(sns.distplot, "value");
```

I am removing the variables 3SsnPorch, LowQualFinSF, Screeas all values in
values are 0.

```{.python .input  n=22}
f = pd.melt(df_qualitive, value_vars=df_qualitive.keys())

g = sns.FacetGrid(f, col="variable",  col_wrap=2, sharex=False, sharey=False, size=6)
g = g.map(sns.countplot, "value")
```

### Correlations

```{.python .input  n=25}
corr = df.corr()

plt.figure(figsize=(20, 20))
sns.heatmap(corr);
```

```{.python .input}
f = pd.melt(df, id_vars=['SalePrice'], value_vars=quantitive)


g = sns.FacetGrid(f, col="variable",  col_wrap=2, sharex=False, sharey=False, size=5)
g = g.map(plt.scatter, "value", "SalePrice", alpha=0.1)
```
