
# Linear Regression Project

In this project we're going to be a freelance data scientist for a day! We've got some contract work with an NYC ecommerce company that sells clothing online. What is interesting is that this company offers in-store consultations to help the customers in their buying decisions, which can be done on either their mobile app or their website.

The question is simple: should the company focus their efforts on their mobile app or their website? Let's find out.

## Data

Let's import the necessary libraries



```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
```

For our data, we've got a file called 'Ecommerce Customers'. 


```
from google.colab import drive
drive.mount('./gdrive')
```


```
filePath = './gdrive/My Drive/Google Colaboratory/Colab Notebooks/Machine Learning Projects/Linear Regression Project (Ecommerce)/data'
```


```
customers = pd.read_csv(filePath + '/Ecommerce Customers')
```


```
customers.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Email</th>
      <th>Address</th>
      <th>Avatar</th>
      <th>Avg. Session Length</th>
      <th>Time on App</th>
      <th>Time on Website</th>
      <th>Length of Membership</th>
      <th>Yearly Amount Spent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>mstephenson@fernandez.com</td>
      <td>835 Frank Tunnel\nWrightmouth, MI 82180-9605</td>
      <td>Violet</td>
      <td>34.497268</td>
      <td>12.655651</td>
      <td>39.577668</td>
      <td>4.082621</td>
      <td>587.951054</td>
    </tr>
    <tr>
      <th>1</th>
      <td>hduke@hotmail.com</td>
      <td>4547 Archer Common\nDiazchester, CA 06566-8576</td>
      <td>DarkGreen</td>
      <td>31.926272</td>
      <td>11.109461</td>
      <td>37.268959</td>
      <td>2.664034</td>
      <td>392.204933</td>
    </tr>
    <tr>
      <th>2</th>
      <td>pallen@yahoo.com</td>
      <td>24645 Valerie Unions Suite 582\nCobbborough, D...</td>
      <td>Bisque</td>
      <td>33.000915</td>
      <td>11.330278</td>
      <td>37.110597</td>
      <td>4.104543</td>
      <td>487.547505</td>
    </tr>
    <tr>
      <th>3</th>
      <td>riverarebecca@gmail.com</td>
      <td>1414 David Throughway\nPort Jason, OH 22070-1220</td>
      <td>SaddleBrown</td>
      <td>34.305557</td>
      <td>13.717514</td>
      <td>36.721283</td>
      <td>3.120179</td>
      <td>581.852344</td>
    </tr>
    <tr>
      <th>4</th>
      <td>mstephens@davidson-herman.com</td>
      <td>14023 Rodriguez Passage\nPort Jacobville, PR 3...</td>
      <td>MediumAquaMarine</td>
      <td>33.330673</td>
      <td>12.795189</td>
      <td>37.536653</td>
      <td>4.446308</td>
      <td>599.406092</td>
    </tr>
  </tbody>
</table>
</div>




```
customers.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Avg. Session Length</th>
      <th>Time on App</th>
      <th>Time on Website</th>
      <th>Length of Membership</th>
      <th>Yearly Amount Spent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>500.000000</td>
      <td>500.000000</td>
      <td>500.000000</td>
      <td>500.000000</td>
      <td>500.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>33.053194</td>
      <td>12.052488</td>
      <td>37.060445</td>
      <td>3.533462</td>
      <td>499.314038</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.992563</td>
      <td>0.994216</td>
      <td>1.010489</td>
      <td>0.999278</td>
      <td>79.314782</td>
    </tr>
    <tr>
      <th>min</th>
      <td>29.532429</td>
      <td>8.508152</td>
      <td>33.913847</td>
      <td>0.269901</td>
      <td>256.670582</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>32.341822</td>
      <td>11.388153</td>
      <td>36.349257</td>
      <td>2.930450</td>
      <td>445.038277</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>33.082008</td>
      <td>11.983231</td>
      <td>37.069367</td>
      <td>3.533975</td>
      <td>498.887875</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>33.711985</td>
      <td>12.753850</td>
      <td>37.716432</td>
      <td>4.126502</td>
      <td>549.313828</td>
    </tr>
    <tr>
      <th>max</th>
      <td>36.139662</td>
      <td>15.126994</td>
      <td>40.005182</td>
      <td>6.922689</td>
      <td>765.518462</td>
    </tr>
  </tbody>
</table>
</div>




```
customers.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 500 entries, 0 to 499
    Data columns (total 8 columns):
    Email                   500 non-null object
    Address                 500 non-null object
    Avatar                  500 non-null object
    Avg. Session Length     500 non-null float64
    Time on App             500 non-null float64
    Time on Website         500 non-null float64
    Length of Membership    500 non-null float64
    Yearly Amount Spent     500 non-null float64
    dtypes: float64(5), object(3)
    memory usage: 31.3+ KB


## Exploratory Data Analysis

Let's explore this data to find some interesting patterns.

We'll use seaborn to compare the Time on Website and Yearly Amount Spent columns.


```
sns.set_palette('GnBu_d')
sns.set_style('whitegrid')
```


```
sns.jointplot(x='Time on Website', y='Yearly Amount Spent', data=customers)
```




    <seaborn.axisgrid.JointGrid at 0x7ff39a687320>




![png](images/Linear%20Regression%20%28Ecommerce%29_12_1.png)


Now we'll do the same thing with Time on App.


```
sns.jointplot(x='Time on App', y='Yearly Amount Spent', data=customers)
```




    <seaborn.axisgrid.JointGrid at 0x7ff3985812b0>




![png](images/Linear%20Regression%20%28Ecommerce%29_14_1.png)


Now let's compare Time on App with Length of Membership.


```
sns.jointplot(x='Time on App', y='Length of Membership', kind='hex', data=customers)
```




    <seaborn.axisgrid.JointGrid at 0x7ff398441898>




![png](images/Linear%20Regression%20%28Ecommerce%29_16_1.png)


Let's explore these types of relationships across the entire data set.


```
sns.pairplot(customers)
```




    <seaborn.axisgrid.PairGrid at 0x7ff3982aab38>




![png](images/Linear%20Regression%20%28Ecommerce%29_18_1.png)


Based on this plot it appears that the feature that correlates most with Yearly Amount Spent is Length of Membership, so let's create a linear model plot.


```
sns.lmplot(x='Length of Membership', y='Yearly Amount Spent', data=customers)
```




    <seaborn.axisgrid.FacetGrid at 0x7ff397dcfe10>




![png](images/Linear%20Regression%20%28Ecommerce%29_20_1.png)


## Training and Testing Data

Now let's split the data into training and testing sets.


```
y = customers['Yearly Amount Spent']
```


```
X = customers[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]
```


```
from sklearn.model_selection import train_test_split
```


```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
```

## Training the Model


```
from sklearn.linear_model import LinearRegression
```


```
lm = LinearRegression()
```


```
lm.fit(X_train, y_train)
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)




```
print('Coefficients: \n', lm.coef_)
```

    Coefficients: 
     [25.98154972 38.59015875  0.19040528 61.27909654]


## Predicting Test Data


```
predictions = lm.predict(X_test)
```


```
plt.scatter(y_test, predictions)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
```




    Text(0, 0.5, 'Predicted Y')




![png](images/Linear%20Regression%20%28Ecommerce%29_33_1.png)


## Evaluating the Model

Let's see how well our model performed.


```
from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
```

    MAE: 7.2281486534308295
    MSE: 79.8130516509743
    RMSE: 8.933815066978626


## Residuals

Let's make sure that everything is right with our data.


```
sns.distplot((y_test-predictions), bins=50)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7ff3922922e8>




![png](images/Linear%20Regression%20%28Ecommerce%29_37_1.png)


Looks like a normal distribution to me!

## Answering the Question

We still didn't answer our question, which is to decide which to focus on: mobile app or website.

Let's use the coefficients that were calculated and interpret them to find the answer.


```
coefficients = pd.DataFrame(lm.coef_, X.columns)
coefficients.columns = ['Coefficient']
print(coefficients)
```

                          Coefficient
    Avg. Session Length     25.981550
    Time on App             38.590159
    Time on Website          0.190405
    Length of Membership    61.279097


Looking at these numbers, which is the coefficient between these variables and the yearly amount of money spent. 

Knowing this, we can see that a 1 unit increase in Avg. Session Length is associated with an increase of 25.98 total dollars spent.

A 1 unit increase in Time on App is associated with an increase of 38.59 total dollars spent.

A 1 unit increase in Time on Website is associated with an increase of 0.19 total dollars spent.

A 1 unit increase in Length of Membership is associated with an increase of 61.28 total dollars spent. 

### The answer is not so cut and dry.

The company could either develop the website to catch up to the performance of the mobile app or the company could focus on the mobile app because it's already doing so well. Either way, we've got some great insights!


```

```
