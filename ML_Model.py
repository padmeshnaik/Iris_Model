import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
import pickle



iris=pd.read_csv('C:/Users/Padmesh/Desktop/Flask/My_First_Model/iris.data',header=None)

iris.columns=['Sepal Length','Sepal Width','Petal Length','Petal Width','Species']

class_le = LabelEncoder()
iris['Species'] = class_le.fit_transform(iris['Species'].values)


X = iris.iloc[:,[0,1,2,3]].values
y = iris.iloc[:, 4].values





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



svm = SVC(kernel='rbf', random_state=0, gamma=0.20, C=100.0)
svm.fit(X_train, y_train)

pickle.dump(svm, open('model.pkl','wb'))


final_features = []

    if request.method=='POST':
        SL=''
        SW=''
        PL=''
        PW=''
        SL=request.form["SL"]
        SW=request.form["SW"]
        PL=request.form["PL"]
        PW=request.form["PW"]
        final_features.append(float(SL))
        final_features.append(float(SW))
        final_features.append(float(PL))
        final_features.append(float(PW))

        final_features1=[final_features]

        prediction=model.predict(final_features)

    

    return render_template('Home.html', prediction=final_features)

