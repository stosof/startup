import pandas as pd
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint
import numpy as np
from keras.models import model_from_json
from sklearn import preprocessing
from keras import optimizers

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

# Logger for model accuracy
class BatchLogger(Callback):

    def on_train_begin(self, epoch, logs={}):
        self.log_values = {}
        for k in self.params['metrics']:
            self.log_values[k] = []

    def on_epoch_end(self, batch, logs={}):
        for k in self.params['metrics']:
            if k in logs:
                self.log_values[k].append(logs[k])

    def get_values(self, metric_name, window):
        d = pd.Series(self.log_values[metric_name])
        return d.rolling(window, center=False).mean()

bl = BatchLogger()

#5-Fold Cross-validation
k = 5
a = len(X)/k
test_start = -a
train_len = len(X)
total_score = []
for i in range(k):

    test_start = test_start + a
    test_end = test_start + a
    if train_len - test_end < a :
        test_end = train_len

    train_01_start = 0
    train_02_start = test_end
    train_01_end = test_start
    train_02_end = train_len

    if test_end == train_len :
        train_02_end = test_start

    X_Train_01 = X.iloc[int(train_01_start):int(train_01_end),:]
    X_Train_02 = X.iloc[int(train_02_start):int(train_02_end),:]
    X_Train = X_Train_01.append(X_Train_02,ignore_index=True)
    Y_Train_01 = Y.iloc[int(train_01_start):int(train_01_end),:]
    Y_Train_02 = Y.iloc[int(train_02_start):int(train_02_end),:]
    Y_Train = Y_Train_01.append(Y_Train_02,ignore_index=True)
    Y_Train = pd.get_dummies(Y_Train.loc[:, "Dependent"], prefix="Dependent")

    X_Test = X.iloc[int(test_start):int(test_end),:]
    Y_Test = Y.iloc[int(test_start):int(test_end),:]
    Y_Test = pd.get_dummies(Y_Test.loc[:, "Dependent"], prefix="Dependent")

#Write train and test sets to excel
#         writer = ExcelWriter("LogReg_CrossVal_X_Train" + str(i) + ".xlsx")
#         X_Train.to_excel(writer,str(i))
#         writer.save()
#         writer = ExcelWriter("LogReg_CrossVal_Y_Train" + str(i) + ".xlsx")
#         Y_Train.to_excel(writer,str(i))
#         writer.save()
#         writer = ExcelWriter("LogReg_CrossVal_X_Test" + str(i) + ".xlsx")
#         X_Test.to_excel(writer,str(i))
#         writer.save()
#         writer = ExcelWriter("LogReg_CrossVal_Y_Test" + str(i) + ".xlsx")
#         Y_Test.to_excel(writer,str(i))
#         writer.save()

    input_dim = X_Train.shape[1]
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

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    filepath = str(i) + "weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
    # checkpoint = ModelCheckpoint(filepath, monitor='acc', verbose=1, save_best_only=True, mode='max')
    checkpoint_val = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint_val] #[checkpoint, checkpoint_val]

    history = model.fit(
        np.array(X_Train), np.array(Y_Train),
        batch_size=25, epochs=500, verbose=0, callbacks=callbacks_list,
        validation_data=(np.array(X_Test), np.array(Y_Test)))




