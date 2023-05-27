# diabetic-predict
#simple project for training model to #classify the patient if he had diabetes #or not

import numpy as np 
from pandas import read_csv
url = "file:///G:/nural%20network/datasets/archive/diabetes.csv"
data= read_csv(url)

input_data = data.values[:,:-1]
output_data= data.values[:,-1]
test_data= data.values[600:,:-1]




from keras.models  import Sequential
from keras.layers import Dense

model= Sequential()
model.add(Dense(12,input_dim=8,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(1,activation='sigmoid'))



model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])


model.fit(input_data,output_data,epochs=620,batch_size=70)



accuracy=model.evaluate(input_data,output_data)


predections= (model.predict(input_data)>0.5).astype(float)


print(predections)



for i in range(20):
    print(predections[i],output_data[i])

