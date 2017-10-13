```{.python .input  n=1}
import pandas as pd
from plotnine import *
from plotnine.data import *
from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model
from altair import Chart, X
```

```{.python .input  n=2}
df = pd.read_csv('../input/train.csv')
```

## Data Cleaning

### Null values

```{.python .input  n=3}
num = df.isnull().sum().sort_values(ascending=False)
num[num > 0]
```

```{.python .input  n=4}
df.GarageYrBlt       
```

```{.python .input}
df.LotFrontage.hist()
show()

df.LotArea.hist()
show()
```

```{.python .input  n=5}
df.fillna(0, inplace=True)
ggplot(df, aes('LotFrontage')) + geom_histogram()
ggplot(df, aes('LotArea')) + geom_histogram()

ggplot(df, aes('LotArea','SalePrice')) + geom_point()
```
