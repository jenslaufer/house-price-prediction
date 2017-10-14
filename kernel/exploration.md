# Data Exploration House Price Dataset

```{.python .input  n=15}
import pandas as pd;
from plotnine import *
from plotnine.data import *
from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model
from altair import Chart, X
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
```

```{.python .input}
plt.figure(figsize=(20, 10))
```

```{.python .input  n=16}
df = pd.read_csv('../input/train.csv')
```

## Data Exploration

Before doing any modeling I want to explore the data and clean it.

```{.python .input  n=17}
df.info()
```

The dataset contains 1460 rows with 81 columns

```{.python .input  n=18}
df.head()
```

### Data Cleaning

#### Null values

Variables with null values:

```{.python .input  n=19}
num = df.isnull().sum().sort_values(ascending=False)
num[num > 0]
```

I check the values that occur for the different values:

```{.python .input  n=20}
for key in num[num > 0].keys():
    print "{}\n".format(df[key].value_counts())
```

```{.python .input  n=21}
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

```{.python .input  n=22}
df.GarageYrBlt.fillna(df.YearBuilt, inplace=True)
```

```{.python .input  n=23}
num = df.isnull().sum().sort_values(ascending=False)
num[num > 0]
```

#### Outliers

```{.python .input  n=24}
df_quantitive = df.select_dtypes(exclude=['object'])
df_qualitive = df.select_dtypes(include=['object'])
```

```{.python .input  n=25}
df.hist()
plt.show()
```

```{.python .input  n=27}
#ggplot(df, aes('LotFrontage')) + geom_histogram()
# ggplot(df, aes('LotArea')) + geom_histogram()

# ggplot(df, aes('LotArea','SalePrice')) + geom_point()
```

### Correlations

```{.python .input}
corr = df_quantitive.corr()

# plot the heatmap
sns.heatmap(corr)
```

```{.python .input}

```
