from keras import Sequential
from keras.layers import Dense
import numpy as np
from numpy import loadtxt

#loading of the dataset
dataset=loadtxt('D:/Shubham P/Diabetes prediction project/pima-indians-diabetes.data.txt', delimiter=',')

#splitting of the data and output

#print(dataset.shape[0])
#print(dataset.shape[1])

X=dataset[:,0:dataset.shape[1]-1]
v1=int(0.7*dataset.shape[0])
v2=dataset.shape[0]-v1
xtrain=X[0:v1,]
xtest=X[v1:,]
y=dataset[:,dataset.shape[1]-1]
#y=np.reshape(y,(y.shape[0],1))
#print(xtrain.shape)


#Splitting the data for training and testing in 7:3 ratio
ytrain=y[0:v1,]
ytest=y[v1:,]
#print(ytest.shape)


#Now ,we will create a sequential model using Keras
model=Sequential()
model.add(Dense(12, input_dim=8, activation='relu')) #shape of the input layer is defined here, in the first hidden layer. This is doing two things, defining the first(i/p) layer and first hidden layer.
model.add(Dense(8,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

#Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#Now, we will fit the model, ie we will train(execute on some data) it.
model.fit(xtrain,ytrain, epochs=150, batch_size=10)

#evaluate(how well it is performing) the model
p,accuracy=model.evaluate(xtest,ytest)


#model.predict()
predictions=model.predict(xtest)

print(predictions)
#print(type(predictions))
print('Accuracy is %.2f', accuracy*100)



