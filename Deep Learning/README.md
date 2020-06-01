
In this project, we will test to see how powerful Neural Networks really are.

## Data


```
import pandas as pd
import numpy as np
url = 'https://raw.githubusercontent.com/jerrytigerxu/Deep-Learning/master/bank_note_data.csv'
data = pd.read_csv(url)
```


```
data.head()
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
      <th>Image.Var</th>
      <th>Image.Skew</th>
      <th>Image.Curt</th>
      <th>Entropy</th>
      <th>Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3.62160</td>
      <td>8.6661</td>
      <td>-2.8073</td>
      <td>-0.44699</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.54590</td>
      <td>8.1674</td>
      <td>-2.4586</td>
      <td>-1.46210</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3.86600</td>
      <td>-2.6383</td>
      <td>1.9242</td>
      <td>0.10645</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3.45660</td>
      <td>9.5228</td>
      <td>-4.0112</td>
      <td>-3.59440</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.32924</td>
      <td>-4.4552</td>
      <td>4.5718</td>
      <td>-0.98880</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



## EDA


```
import seaborn as sns
%matplotlib inline
```


```
sns.countplot(x='Class', data=data)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f39a9b89908>




![png](images/Deep%20Learning%20Project_6_1.png)



```
sns.pairplot(data, hue='Class', diag_kind='hist')
```




    <seaborn.axisgrid.PairGrid at 0x7f39ad5db5c0>




![png](images/Deep%20Learning%20Project_7_1.png)



```
from sklearn.preprocessing import StandardScaler
```


```
scaler = StandardScaler()
```


```
scaler.fit(data.drop('Class', axis=1))
```




    StandardScaler(copy=True, with_mean=True, with_std=True)




```
scaled_features = scaler.fit_transform(data.drop('Class', axis=1))
```


```
df_feat = pd.DataFrame(scaled_features, columns=data.columns[:-1])
df_feat.head()
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
      <th>Image.Var</th>
      <th>Image.Skew</th>
      <th>Image.Curt</th>
      <th>Entropy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.121806</td>
      <td>1.149455</td>
      <td>-0.975970</td>
      <td>0.354561</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.447066</td>
      <td>1.064453</td>
      <td>-0.895036</td>
      <td>-0.128767</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.207810</td>
      <td>-0.777352</td>
      <td>0.122218</td>
      <td>0.618073</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.063742</td>
      <td>1.295478</td>
      <td>-1.255397</td>
      <td>-1.144029</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.036772</td>
      <td>-1.087038</td>
      <td>0.736730</td>
      <td>0.096587</td>
    </tr>
  </tbody>
</table>
</div>



## Train Test Split


```
X = df_feat
```


```
y = data['Class']
```


```
from sklearn.model_selection import train_test_split
```


```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
```

## TensorFlow


```
import tensorflow as tf
```


```
df_feat.columns
```




    Index(['Image.Var', 'Image.Skew', 'Image.Curt', 'Entropy'], dtype='object')




```
image_var = tf.feature_column.numeric_column('Image.Var')
image_skew = tf.feature_column.numeric_column('Image.Skew')
image_curt = tf.feature_column.numeric_column('Image.Curt')
entropy = tf.feature_column.numeric_column('Entropy')
```


```
feat_cols = [image_var, image_skew, image_curt, entropy]
```


```
classifier = tf.estimator.DNNClassifier(hidden_units=[10, 20, 10], n_classes=2, feature_columns=feat_cols)
```

    WARNING: Logging before flag parsing goes to stderr.
    W0625 17:13:06.752336 139886474659712 estimator.py:1811] Using temporary folder as model directory: /tmp/tmpy_imy2y4



```
input_func = tf.estimator.inputs.pandas_input_fn(x=X_train, y=y_train, batch_size=20, shuffle=True)
```


```
classifier.train(input_fn=input_func, steps=500)
```

    W0625 17:14:08.558118 139886474659712 deprecation.py:323] From /usr/local/lib/python3.6/dist-packages/tensorflow/python/training/training_util.py:236: Variable.initialized_value (from tensorflow.python.ops.variables) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use Variable.read_value. Variables in 2.X are initialized automatically both in eager and graph (inside tf.defun) contexts.
    W0625 17:14:08.587572 139886474659712 deprecation.py:323] From /usr/local/lib/python3.6/dist-packages/tensorflow_estimator/python/estimator/inputs/queues/feeding_queue_runner.py:62: QueueRunner.__init__ (from tensorflow.python.training.queue_runner_impl) is deprecated and will be removed in a future version.
    Instructions for updating:
    To construct input pipelines, use the `tf.data` module.
    W0625 17:14:08.590641 139886474659712 deprecation.py:323] From /usr/local/lib/python3.6/dist-packages/tensorflow_estimator/python/estimator/inputs/queues/feeding_functions.py:500: add_queue_runner (from tensorflow.python.training.queue_runner_impl) is deprecated and will be removed in a future version.
    Instructions for updating:
    To construct input pipelines, use the `tf.data` module.
    W0625 17:14:08.611769 139886474659712 deprecation.py:506] From /usr/local/lib/python3.6/dist-packages/tensorflow/python/ops/init_ops.py:1251: calling VarianceScaling.__init__ (from tensorflow.python.ops.init_ops) with dtype is deprecated and will be removed in a future version.
    Instructions for updating:
    Call initializer instance with the dtype argument instead of passing it to the constructor
    W0625 17:14:09.706165 139886474659712 deprecation.py:323] From /usr/local/lib/python3.6/dist-packages/tensorflow_estimator/python/estimator/canned/head.py:437: to_float (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use `tf.cast` instead.
    W0625 17:14:09.790671 139886474659712 deprecation.py:323] From /usr/local/lib/python3.6/dist-packages/tensorflow/python/ops/nn_impl.py:180: add_dispatch_support.<locals>.wrapper (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use tf.where in 2.0, which has the same broadcast rule as np.where
    W0625 17:14:09.897806 139886474659712 deprecation.py:506] From /usr/local/lib/python3.6/dist-packages/tensorflow/python/training/adagrad.py:76: calling Constant.__init__ (from tensorflow.python.ops.init_ops) with dtype is deprecated and will be removed in a future version.
    Instructions for updating:
    Call initializer instance with the dtype argument instead of passing it to the constructor
    W0625 17:14:10.906214 139886474659712 deprecation.py:323] From /usr/local/lib/python3.6/dist-packages/tensorflow/python/training/monitored_session.py:875: start_queue_runners (from tensorflow.python.training.queue_runner_impl) is deprecated and will be removed in a future version.
    Instructions for updating:
    To construct input pipelines, use the `tf.data` module.





    <tensorflow_estimator.python.estimator.canned.dnn.DNNClassifier at 0x7f3992e75da0>



## Model Evaluation


```
pred_fn = tf.estimator.inputs.pandas_input_fn(x=X_test, batch_size=len(X_test), shuffle=False)
```


```
note_predictions = list(classifier.predict(input_fn=pred_fn))
```

    W0625 17:15:41.045409 139886474659712 deprecation.py:323] From /usr/local/lib/python3.6/dist-packages/tensorflow/python/training/saver.py:1276: checkpoint_exists (from tensorflow.python.training.checkpoint_management) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use standard file APIs to check for files with this prefix.



```
note_predictions[0]
```




    {'all_class_ids': array([0, 1], dtype=int32),
     'all_classes': array([b'0', b'1'], dtype=object),
     'class_ids': array([1]),
     'classes': array([b'1'], dtype=object),
     'logistic': array([0.98087597], dtype=float32),
     'logits': array([3.9374974], dtype=float32),
     'probabilities': array([0.01912409, 0.98087597], dtype=float32)}




```
final_preds = []
for pred in note_predictions:
  final_preds.append(pred['class_ids'][0])
```


```
from sklearn.metrics import classification_report, confusion_matrix
```


```
print(confusion_matrix(y_test, final_preds))
```

    [[225   3]
     [  0 184]]



```
print(classification_report(y_test, final_preds))
```

                  precision    recall  f1-score   support
    
               0       1.00      0.99      0.99       228
               1       0.98      1.00      0.99       184
    
        accuracy                           0.99       412
       macro avg       0.99      0.99      0.99       412
    weighted avg       0.99      0.99      0.99       412
    



```

```
