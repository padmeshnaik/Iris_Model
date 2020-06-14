from sklearn import datasets
import numpy as np
import pandas as pd
from io import StringIO

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler



from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier



import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


"""

iris=datasets.load_iris()

X = iris.data[:, [0,1,2,3]]

print(X)
y = iris.target

print('Class labels:', np.unique(y))

"""
iris=pd.read_csv('C:/Users/Padmesh/Desktop/Flask/My_First_Model/iris.data',header=None)

iris.columns=['Sepal Length','Sepal Width','Petal Length','Petal Width','Species']

class_le = LabelEncoder()
iris['Species'] = class_le.fit_transform(iris['Species'].values)


X=iris.drop('Species',axis=1)


y = iris.Species



# Splitting data into 70% training and 30% test data:

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=1,stratify=y)


print('Class labels:', np.unique(y_train))
print('Class labels:', np.unique(iris['Species']))




#print('Labels counts in y:', np.bincount())
print('Labels counts in y_train:', np.bincount(y_train))
print('Labels counts in y_test:', np.bincount(y_test))


sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)



###### Choosing the best algorithm according to the features.


#forest = RandomForestClassifier(criterion='gini',n_estimators=25,random_state=1,n_jobs=2)
#forest.fit(X_train_std, y_train)

#lr = LogisticRegression(C=100.0, random_state=1)
#lr.fit(X_train_std, y_train)

#svm = SVC(kernel='rbf', random_state=0, gamma=0.20, C=10.0)
#svm.fit(X_train_std, y_train)

#tree = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=1)
#tree.fit(X_train_std, y_train)

#ppn = Perceptron(max_iter=100, eta0=0.1, random_state=1)
#ppn.fit(X_train_std, y_train)

#knn = KNeighborsClassifier(n_neighbors=5, p=2,metric='minkowski')
#knn.fit(X_train_std, y_train)

#y_pred=knn.predict(X_test_std)

#print("Accuracy : %.2f" %accuracy_score(y_test,y_pred))


"""
###### Dealing with missing data.

# Eliminating Missing Data.

#1). If the data array is stored using pandas library.

csv_data = '''A,B,C,D
1.0,2.0,3.0,4.0
5.0,6.0,,8.0
10.0,11.0,12.0,'''




df = pd.read_csv(StringIO(csv_data))
print(df)

print()
print(df.isnull().sum())

# remove rows that contain missing values
print(df.dropna(axis=0))

# remove columns that contain missing values
print(df.dropna(axis=1))


# Imputing Missing data with mean or median values.

imr = SimpleImputer(missing_values=np.nan, strategy='most_frequent')  # strategy=['mean','median','constant']
imr = imr.fit(df.values)
imputed_data = imr.transform(df.values)
print(imputed_data)
"""

"""

###### Handling Categorical Data.

# By the process of mapping.

size_mapping = {'XL': 3,
                'L': 2,
                'M': 1}

df['size'] = df['size'].map(size_mapping)
print("After Mapping")
print(df)

# The above code will replace 'XL' with 3 and so on.



inv_size_mapping = {v: k for k, v in size_mapping.items()}
df['size']=df['size'].map(inv_size_mapping)
print("Inverse Mapping")
print(df)


# Encoding class labels

class_le = LabelEncoder()
y = class_le.fit_transform(df['classlabel'].values)
print(y)

# The above code will replace the string values in 'class label' column with integer values[0,1,2,...]


# ## Performing one-hot encoding on nominal features

X = df[['color', 'size', 'price']].values   # Store all the the nominal values in X

ohe = OneHotEncoder()

print(ohe.fit_transform(X).toarray())

# The above code will replace 'red' with 0 0 1, 'blue' with 0 1 0, 'green' with 1 0 0 .

"""



"""

###### Feature Scaling


# Bringing features onto the same scale

# Nomalization

mms = MinMaxScaler()
X_train_norm = mms.fit_transform(X_train)
X_test_norm = mms.transform(X_test)

# Standardization

stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)



##### Regularization (To prevent our model from overfitting)



# For regularized models in scikit-learn that support L1 regularization,
# we can simply set the `penalty` parameter to `'l1'` to obtain a sparse solution.


lr = LogisticRegression(penalty='l1', C=10.0, solver='liblinear')

# Note that C=1.0 is the default. You can increase or decrease it to make the regulariztion effect
# stronger or weaker, respectively.

lr.fit(X_train_std, y_train)
print('Training accuracy:', lr.score(X_train_std, y_train))
print('Test accuracy:', lr.score(X_test_std, y_test))

"""
svm = SVC(kernel='rbf', random_state=0, gamma=0.20, C=100.0)
svm.fit(X_train_std, y_train)

print('Training accuracy:', svm.score(X_train_std, y_train))
print('Test accuracy:', svm.score(X_test_std, y_test))





###### Dimensionality Reduction















