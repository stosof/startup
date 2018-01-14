from keras.models import load_model
import pandas as pd
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint
import numpy as np
from keras.models import model_from_json
from sklearn import preprocessing
from keras import optimizers
from keras import backend as K

#Import data from CSV
df_orig_data = pd.read_csv("data.csv")
#Drop CAX_ID column
df_orig_data = df_orig_data.iloc[:,1:]
#Create lists with the names of the numerical and categorical varaibles in the dataset
numerical_col_names = list(df_orig_data.select_dtypes(include=[np.number]).columns.values)
non_numerical_col_names = list(df_orig_data.select_dtypes(exclude=[np.number]).columns.values)

#One-hot encoding for categorical variables
cat_vars = non_numerical_col_names
data = df_orig_data
for i in range(len(non_numerical_col_names)):
    #     cat_list = non_numerical_col_names[i]+'_'+str(i)
    cat_list = pd.get_dummies(data.loc[:, non_numerical_col_names[i]], prefix=non_numerical_col_names[i])
    #     print (cat_list)
    data1 = pd.concat([data, cat_list], axis=1)
    data = data1

cat_vars = non_numerical_col_names
data_vars = data.columns.values.tolist()
to_keep = [i for i in data_vars if i not in cat_vars]

data_final = data[to_keep]

#Select numerical columns in order to perform normalization data
data = data_final
data_vars = data.columns.values.tolist()
to_normalize = [i for i in data_vars if i in numerical_col_names[1:]]
data_to_normalize = data[to_normalize]

#Drop original numerical vars
num_vars = numerical_col_names
data_vars = data.columns.values.tolist()
to_keep = [i for i in data_vars if i not in num_vars]
data_no_numerical = data[to_keep]

#Perform min-max normalization on numerical values
min_max_scaler = preprocessing.MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(data_to_normalize)
df_normalized = pd.DataFrame(np_scaled)
df_normalized.columns = to_normalize
data_final = pd.concat([data_no_numerical, df_normalized], axis=1)
print(data_final)

#Exclude Dependent
data_final_vars = data.columns.values.tolist()
y = ['Dependent']
X = [i for i in data_final if i not in y]

data_final = data[X]
y_final = data[y]

X = data_final
Y = y_final
Y_1 = pd.get_dummies(Y.loc[:, "Dependent"], prefix="Dependent")

input_dim = X.shape[1]
print (input_dim)
model = Sequential()
model.add(Dropout(0.5, input_shape=(input_dim,)))
model.add(Dense(1024, activation='relu', kernel_initializer='normal')) #input_shape=(input_dim,
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu', kernel_initializer='normal'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu', kernel_initializer='normal'))
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu', kernel_initializer='normal'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu', kernel_initializer='normal'))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu', kernel_initializer='normal'))
model.add(Dropout(0.3))
model.add(Dense(16, activation='relu', kernel_initializer='normal'))
model.add(Dropout(0.3))
model.add(Dense(2, activation='sigmoid', kernel_initializer='normal'))


X = X.iloc[0:len(X),:]
Y = Y.iloc[0:len(X),:]
Y = pd.get_dummies(Y.loc[:, "Dependent"], prefix="Dependent")

model.load_weights("0weights-improvement-43-0.74.hdf5")
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
scores = model.evaluate(np.array(X), np.array(Y_1), verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
predictions_0 = model.predict(np.array(X))
# print (predictions_0)
# print(len)

model.load_weights("1weights-improvement-27-0.71.hdf5")
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
scores = model.evaluate(np.array(X), np.array(Y_1), verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
predictions_1 = model.predict(np.array(X))
# print (predictions)

model.load_weights("2weights-improvement-351-0.60.hdf5")
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
scores = model.evaluate(np.array(X), np.array(Y_1), verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
predictions_2 = model.predict(np.array(X))
# print (predictions)

model.load_weights("3weights-improvement-90-0.79.hdf5")
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
scores = model.evaluate(np.array(X), np.array(Y_1), verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
predictions_3 = model.predict(np.array(X))
# print (predictions)

model.load_weights("4weights-improvement-49-0.81.hdf5")
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
scores = model.evaluate(np.array(X), np.array(Y_1), verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
predictions_4 = model.predict(np.array(X))
# print (predictions)

#Combine predictions of the 5 NNs
combined_predictions = []
for i in range(len(predictions_0)):

    sum_0 = 0
    sum_1 = 0

    sum_0 =(predictions_0[i,0] + predictions_1[i,0] + predictions_2[i,0] + predictions_3[i,0] + predictions_4[i,0])/5
    sum_1 = (predictions_0[i,1] + predictions_1[i,1] + predictions_2[i,1] + predictions_3[i,1] + predictions_4[i,1])/5

    combined_predictions.append([sum_0, sum_1])

print(combined_predictions)

#Check accuracy of combined preditions
Y_arr = np.array(Y_1)
Comb_arr = np.array(combined_predictions)
correct = 0
incorrect = 0
total = len(predictions_0)
for i in range(len(predictions_0)):

    if(Y_arr[i].argmax(axis=0) == Comb_arr[i].argmax(axis=0)):
        correct += 1
    else:
        incorrect += 1

total_combined_acc = correct/total
print (total_combined_acc)

K.clear_session()