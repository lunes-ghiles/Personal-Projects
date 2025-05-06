import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn import datasets
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

noise_test = False
noise_labels = False
PR_NOISE = (0.20, 0.10, 0.03, 0.07, 0.06, 0.19, 0.10, 0.1, 0.00, 0.15)


X, y = datasets.fetch_openml('mnist_784', version=1, return_X_y=True)
X = np.asarray(X)

Y = np.asarray(y)
print('X shape', X.shape)
print('Y shape', y.shape)

def show_digit(x_, y_):
    X_reshape = x_.reshape(28, 28) # reshape it to have 28*28
    plt.imshow(X_reshape, 'gray')
    plt.title('Label is: ' + y_)
    plt.show()

test_percentage = 0.2

X_train, X_test, y_train, y_test = model_selection.train_test_split(X,Y, test_size= test_percentage, random_state=12) 

if noise_test or noise_labels:
    X_test =  X_test + np.random.uniform(0,255,size = X_test.shape)

max_x_train = np.max(X_train)
X_train = X_train/(max_x_train) 
X_test = X_test/(max_x_train)

if noise_labels:
    Noise = [str(index) for index in np.random.choice(10,20000, p = PR_NOISE)]
    y_train[:20000] = Noise

show_digit(X_train[5],y_train[5])

R_train_KNN = []
R_test_KNN = []
Neighbours = np.arange(1,100,5)
for k in Neighbours:
  model = KNeighborsClassifier(n_neighbors = k, weights = 'uniform') 
  model.fit(X_train, y_train) 
  y_res_train = model.predict(X_train)
  y_res_test = model.predict(X_test)  
  R_train_KNN.append(sklearn.metrics.accuracy_score(y_train, y_res_train))
  R_test_KNN.append(sklearn.metrics.accuracy_score(y_test, y_res_test))



plt.clf()
plt.figure(figsize=(8, 6))
plt.plot(Neighbours, R_train_KNN, color='b', label='Train Accuracy')
plt.plot(Neighbours, R_test_KNN, color='g', label='Test Accuracy')
plt.title(f"Effect of K on KNN Model Accuracy")
plt.xlabel("K")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
plt.show()


R_train_RF = []
R_test_RF = []
Min_sample = np.arange(5,51,5)
for s in Min_sample:
  model_RF = RandomForestClassifier(criterion='entropy', min_samples_leaf=s, bootstrap = True) 
  model_RF.fit(X_train,y_train)                     
  y_res_train = model_RF.predict(X_train)   
  y_res_test = model_RF.predict(X_test)    
  R_train_RF.append(sklearn.metrics.accuracy_score(y_train, y_res_train))
  R_test_RF.append(sklearn.metrics.accuracy_score(y_test, y_res_test))

plt.clf()
plt.figure(figsize=(8, 6))
plt.plot(Min_sample, R_train_RF, color='b', label='Train Accuracy')
plt.plot(Min_sample, R_test_RF, color='g', label='Test Accuracy')
plt.title(f"Effect of Min Sample Size on RF Model Accuracy")
plt.xlabel("s")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
plt.show()
