# 🚀 Sampling

*Submitted by:*   
*Name: **YUVRAJ KALSI***   
*Roll No: **102017081***  
*Group: **3CS4*** 

Sampling is process of selecting a portion or subset (*known as **sample***), of the population to represent the entire population.

## Implementation of sampling consists of 6 major steps:

Note: Libraries used in the program: [pandas](https://pandas.pydata.org/docs/), [numpy](https://numpy.org/doc/stable/), [imblearn](https://imbalanced-learn.org/stable/).

### Step - 1

Dataset used in the program: 🔗[Credit Card Dataset](https://github.com/AnjulaMehto/Sampling_Assignment/blob/main/Creditcard_data.csv)

```bash
fd = pd.read_csv("Creditcard_data.csv")
```

### Step - 2

Balancing the data using **Over-Sampling** Technique.
Oversampling involves increasing the number of samples in the minority class by synthesizing new samples. This is done by duplicating existing samples in the minority class.

```bash
from imblearn.over_sampling import RandomOverSampler
import pandas as pd

X = df.drop('Class', axis=1)
y = df['Class']

ros = RandomOverSampler()

X_resampled, y_resampled = ros.fit_resample(X, y)

df_resampled = pd.concat([pd.DataFrame(X_resampled), pd.DataFrame(y_resampled)], axis=1)
df_resampled.columns = df.columns
```

### Step - 3

**Sampling Techniques** used in the program:  
* Simple Random Sampling
* Systematic Sampling
* Stratified Sampling
* Cluster Sampling
* Quota Sampling

**1)** **Simple Random Sampling** is a statistical sampling technique in which every unit or item in the population has an equal chance of being selected for the sample. This means that each member of the population is selected independently and randomly, with no bias or preference.

```bash
def simple_random_sample(df, z_score, margin_error, p):

    # Calculate Sample-Size
    n = (z_score**2 * p * (1-p)) / margin_error**2
    n = int(np.ceil(n))   

    # Select the rows corresponding to the random indices to create the sample  
    sample_indices = random.sample(range(len(df)), n)
    sample_df = df.iloc[sample_indices, :]    

    return sample_df
```
**2)** **Systematic Sampling** is a statistical sampling technique where a random starting point is selected from a population and then every kth element is selected from the population until the desired sample size is achieved and the value of k is known as the sampling interval.


```bash
def systematic_sample(df, k):
   
    # Create a list of starting indices for each interval
    start_indices = np.arange(0, len(df), k)
    
    # Select the rows corresponding to the starting indices to create the sample
    sample_df = df.iloc[start_indices, :]
    
    return sample_df    
```
**3)** **Stratified Sampling** is a statistical sampling technique in which the population is divided into non-overlapping subgroups or strata, based on a particular characteristic or attribute, and samples are randomly selected from each stratum. 

```bash
def stratified_sample(df, col, z, e, p):
    
    # Calculate Sample-Size
    t = df[col].value_counts()
    s = len(t)
    n = (z ** 2) * (p * (1 - p)) // ((e / s) ** 2)
    
    n_rows = t[0] + t[1]
    
    # Separate the data into groups based on 'Class' i.e 0 and 1 and generate random sample from each group.
    sample_df = df.groupby(col, group_keys=False).apply(lambda x: x.sample(frac = n/n_rows))
    
    return sample_df
```

**4)** **Cluster Sampling** is a statistical sampling technique where the population is divided into smaller groups called clusters. The clusters are then randomly selected and all individuals within the selected clusters are included in the sample. 

```bash
def cluster_sampling(df, z, e, p, c):
    
    # Calculate Sample-Size
    n = ((z ** 2) * (p * (1 - p)) // ((e) ** 2)) / (df.shape[0] - c)

    # Take a random cluster with size of 'n' 
    cluster_sample_df = df.sample(frac = n)
    
    return cluster_sample_df
```
**5)** **Quota Sampling** is a non-probability sampling method that relies on the non-random selection of a predetermined number or proportion of units called a quota. We first divide the population into mutually exclusive subgroups (called strata) and then recruit sample units until you reach your quota.

```bash
def quota_sampling(df, strata, quotas):

    sample = pd.DataFrame(columns = df.columns)

    for stratum, quota in quotas.items():

        # Generating all rows which has 'strata = stratum'
        stratum_df = df[df[strata] == stratum]

        # Randomly extracting only 'n = quota' number of enteries    
        stratum_sample = stratum_df.sample(n = quota, random_state = 1)
        
        # Concatenating in data-frame
        sample = pd.concat([sample, stratum_sample], ignore_index = True)
    
    return sample
```

### Step - 4

Calling each function to create five respective samples.

```bash
sample_1 = simple_random_sample(df_resampled, 1.96, 0.05, 0.5)

sample_2 = systematic_sample(df_resampled, 5)

sample_3 = stratified_sample(df_resampled, 'Class', 0.95, 0.05, 0.5)

sample_4 = cluster_sampling(df_resampled, 0.95, 0.05, 0.5, 300)

quotas = {0: 50, 1: 50}
sample_5 = quota_sampling(df_resampled, 'Class', quotas)
```

### Step - 5

**Models** used in the program:

* Logistic Regression
* Random Forest Classifier
* SVM
* Gradient Boosting Classifier
* Naive Bayes

```bash
# Testing on whole dataset

X_test = np.array(df.drop('Class', axis=1))
y_test = np.array(df['Class']).reshape(-1,).astype('int')
```

**1)** **Logistic Regression**

```bash
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

M1 = []
for sample in all_samples:

    # Extracting training datset
    X_train = np.array(sample.iloc[:,:-1])
    y_train = np.array(sample.iloc[:,-1:].values).reshape(-1,).astype('int')
    
    # Fitting the model
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    
    # Predicting the target
    y_pred = clf.predict(X_test)

    # Calculating accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Appending to a list
    M1.append(accuracy)

```

**2)** **Random Forest Classifier**

```bash
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

M2 = []
for sample in all_samples:
    
    # Extracting training datset
    X_train = np.array(sample.iloc[:,:-1])
    y_train = np.array(sample.iloc[:,-1:].values).reshape(-1,).astype('int')
    
    # Fitting the model
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)

    # Predicting the target
    y_pred = clf.predict(X_test)

    # Calculating accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Appending to a list
    M2.append(accuracy)
```

**3)** **SVM**

```bash
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

M3 = []
for sample in all_samples:

    # Extracting training datset
    X_train = np.array(sample.iloc[:,:-1])
    y_train = np.array(sample.iloc[:,-1:].values).reshape(-1,).astype('int')
    
    # Fitting the model
    clf = SVC()
    clf.fit(X_train, y_train)
    
    # Predicting the target
    y_pred = clf.predict(X_test)

    # Calculating accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Appending to a list
    M3.append(accuracy)

```

**4)** **Gradient Boosting Classifier**

```bash
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

M4 = []
for sample in all_samples:

    # Extracting training datset
    X_train = np.array(sample.iloc[:,:-1])
    y_train = np.array(sample.iloc[:,-1:].values).reshape(-1,).astype('int')
    
    # Fitting the model
    clf = GradientBoostingClassifier()
    clf.fit(X_train, y_train)
    
    # Predicting the target
    y_pred = clf.predict(X_test)
    
    # Calculating accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Appending to a list
    M4.append(accuracy)

```

**5)** **Naive Bayes**

```bash
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

M5 = []
for sample in all_samples:
    
    # Extracting training datset
    X_train = np.array(sample.iloc[:,:-1])
    y_train = np.array(sample.iloc[:,-1:].values).reshape(-1,).astype('int')
    
    # Fitting the model
    clf = GaussianNB()
    clf.fit(X_train, y_train)

    # Predicting the target
    y_pred = clf.predict(X_test)

    # Calculating accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Appending to a list
    M5.append(accuracy)

```

### Step - 6

Creating **Comparison Table** showing the accuracies of samples created from different *sampling techniques* with different *models*.

```bash
models = [M1, M2, M3, M4, M5]

Comparison = pd.DataFrame(models, columns = ['Simple Random Sampling', 'Systematic Sampling', 'Stratified Sampling', 'Cluster Sampling',
'Quota Sampling'])

Comparison.index = ['Logistic Regression', 'Random Forest', 'SVM', 'Gradient Boosting', 'Naive Bayes']

print(Comparison)
```

The **Comparison Table** looks like:

|| Simple Random Sampling | Systematic Sampling | Stratified Sampling | Cluster Sampling | Quota Sampling | 
| :---------------: | :---------------: | :---------------: | :---------------: | :---------------: | :---------------: | 
| Logistic Regression | 0.835492 | 0.809585 | 0.843264 | 0.768135 | 0.822539 |
| Random Forest	 | 0.996114 | 0.996114 | 0.997409 | 0.976684 | 0.974093 |
| SVM | 0.623057 | 0.702073 | 0.694301 | 0.704663 | 0.755181 |
| Gradient Boosting	 | 0.970207 | 0.992228 | 0.990933 | 0.954663 | 0.918394 |
| Naive Bayes | 0.889896 | 0.791451 | 0.893782 |  0.716321 | 0.926166 |

Finding which **Sampling Technique** gives higher accuracy on which **Model**.

```bash
# Calculating the maximum value
max_value = Comparison.max().max()

# Finding corresponding row-name and column-name
row, col = Comparison.stack().idxmax()

print(f"The Sample created from '{col}' Technique gives the highest accuracy on model '{row}' of {max_value}")
```

The **Final Output** looks like:   
```bash
The Sample created from 'Stratified Sampling' Technique gives the highest accuracy on model 'Random Forest' of 0.9974093264248705.
```
