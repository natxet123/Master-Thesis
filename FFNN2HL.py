import numpy as np
import pandas as pd
import keras as K
from keras.layers import Dense
import matplotlib.pyplot as plt
# changed the patience to 20, had test rel rmse of  ..

#####################################
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
#####################################
mv = 1      #Model Version


def rel_rms_np(y_true, y_sim):
    return np.sqrt(np.mean(np.square(y_true-y_sim)))/np.sqrt(np.mean(np.square(y_true-np.mean(y_true))))  # goodness of fit: divided by rms(detrended targets)

# Load data
path = 'C:\\Users\\Ignacio\\Desktop\\TFM\\FFNN_Test'    #'../../'
Utrain = pd.read_csv('C:\\Users\\Ignacio\\Desktop\\TFM\\FFNN_Test\\Utrain.csv', header=None).to_numpy()
Utest = pd.read_csv('C:\\Users\\Ignacio\\Desktop\\TFM\\FFNN_Test\\Utest.csv', header=None).to_numpy()
Udev = pd.read_csv('C:\\Users\\Ignacio\\Desktop\\TFM\\FFNN_Test\\Udev.csv', header=None).to_numpy()

cytrain = pd.read_csv('C:\\Users\\Ignacio\\Desktop\\TFM\\FFNN_Test\\ftrain.csv', header=None).to_numpy()
cytest = pd.read_csv('C:\\Users\\Ignacio\\Desktop\\TFM\\FFNN_Test\\ftest.csv', header=None).to_numpy()
cydev = pd.read_csv('C:\\Users\\Ignacio\\Desktop\\TFM\\FFNN_Test\\fdev.csv', header=None).to_numpy()

'''dtrain = pd.read_csv(path+'dtrain.csv', header=None).to_numpy()
dtest = pd.read_csv(path+'dtest.csv', header=None).to_numpy()
ddev = pd.read_csv(path+'ddev.csv', header=None).to_numpy()'''

# Model
features = (Utrain.shape[1],)   #.shape[1] for row dimensions ([0] for number of columns)
optim = K.optimizers.Adam()
losscy = K.losses.MeanSquaredError()  # mean_squared_error
metric = [K.metrics.RootMeanSquaredError(), K.metrics.MeanAbsoluteError()]

nm = 0
nl = 20  # number of layers
if nm:
    TestModel = K.Sequential([Dense(20, activation='relu', input_shape=[len(Utrain[1]), ], name='hidden1'),
                              Dense(20, activation='relu', name='Hidden2'), #Dense: each neuron in the layer is connected to every neuron in the previous layer
                              Dense(20, activation='relu', name='Hidden3'),
                              #Dense(20, activation='relu', name='Hidden4'),
                              #Dense(20, activation='relu', name='Hidden5'),
                              #Dense(20, activation='relu', name='Hidden6'),
                              #Dense(20, activation='relu', name='Hidden7'),
                              #Dense(20, activation='relu', name='Hidden8'),
                              #Dense(20, activation='relu', name='Hidden9'),
                              #Dense(20, activation='relu', name='Hidden10'),
                              #Dense(20, activation='relu', name='Hidden11'),
                              #Dense(20, activation='relu', name='Hidden12'),
                              #Dense(20, activation='relu', name='Hidden13'),
                              #Dense(20, activation='relu', name='Hidden14'),
                              #Dense(20, activation='relu', name='Hidden15'),
                              #Dense(20, activation='relu', name='Hidden16'),
                              #Dense(20, activation='relu', name='Hidden17'),
                              #Dense(20, activation='relu', name='Hidden18'),
                              #Dense(20, activation='relu', name='Hidden19'),
                              #Dense(20, activation='relu', name='Hidden20'),
                              Dense(1, activation='linear', name='output')], name=str(nl)+'hl_v'+str(mv))

    TestModel.compile(loss=losscy, optimizer=optim, metrics=metric)

    earlystop = K.callbacks.EarlyStopping(monitor='val_loss', patience=20, verbose=1)

    ###
    Testrun = TestModel.fit(Utrain, cytrain,
                            epochs=200,
                            batch_size=56,
                            callbacks=earlystop,
                            validation_steps=5,
                            validation_data=(Udev, cydev))         # validationsteps: amount of batches for the validation data (to save RAM)

    hist = pd.DataFrame(Testrun.history)    # print progress
    hist.to_csv(str(nl) + 'hl_v'+str(mv)+'_History.csv')      # save under a different number each time so you don't overwrite your previous models, they might have been better (plot this to evaluate training process)

    #K.utils.plot_model(TestModel, to_file='./Groot2.png', show_shapes=True, show_layer_names=True)   # save model structure

    TestModel.save(str(nl) + 'hl_v'+str(mv), save_format='h5')
else:
    TestModel = K.models.load_model(str(nl) + '_nr1')
    #TestModel.summary()

# Evaluate the data on the unseen testing data
TestModel_eval = TestModel.evaluate(Utest, cytest)
cypredtst = TestModel.predict(Utest)
assert (cypredtst.shape == cytest.shape)

Udev_sim = TestModel.predict(Udev)
rel_RMSEdev = rel_rms_np(Udev_sim, cydev)

# metrics
num_trainsamples = len(cytrain)
num_testsamples = len(cytest)
num_params = TestModel.count_params()
err = (cypredtst - cytest)

abs_err = np.abs(err)           # abs = absolute waarde (alles >=0 en element |R)
avg_err = np.mean(err)
std_err = np.std(err)
loss = np.sum(err**2)/num_testsamples       # MSE
rel_err = loss/(np.sum(np.abs(cytest) ** 2) / num_testsamples)   # MSE/Mean Squared juiste waarde
rel_RMSE = np.sqrt(rel_err)

#############################################################
'''data = [nl, rel_RMSE]
from csv import writer
with open('RMSE.csv', 'a') as f_object:
    writer_object = writer(f_object)
    writer_object.writerow(data)
    f_object.close()


data2 = {
  "layers": [nl],
  "rel_RMSE": [rel_RMSE]
}
RMSE_print = pd.DataFrame(data2)
RMSE_print.to_csv(str(nl) + 'hl_v'+str(mv) + '_RMSE.csv')'''
#############################################################

TestModel_perfo = {'Model activations': 'ReLU-Linear',
                    'Number of training samples': num_trainsamples,
                    'Number of model parameters': num_params,
                    'Ratio training samples-parameters':
                    num_trainsamples/num_params,
                    'Model loss on dev data': TestModel_eval[0],
                    'Model RMSE on dev data': TestModel_eval[1],
                    'Model Absolute error on dev data': TestModel_eval[2],
                    'Number of test samples': num_testsamples,
                    'Average Error': avg_err,
                    'Standard Deviation of Error': std_err,
                    'Average Absolute Error': np.mean(abs_err),
                    'Maximum Error': np.max(abs_err),
                    'Loss': loss,
                    'Relative Error': rel_err,
                    'Relative RMSE on validation set': rel_RMSEdev,
                    'Relative RMSE on test set': rel_RMSE}

Model1Perfo = pd.DataFrame.from_dict(TestModel_perfo, orient='index')
print(Model1Perfo)
Model1Perfo.index.name = 'FFNN'
Model1Perfo.to_csv(str(nl) + '_nr1' + '_perfo.csv', header=['Value'])
'''
#plotting
hist = pd.read_csv(str(nl) + 'hl_v'+str(mv) + '_History.csv')

plt.figure()
plt.plot(hist["root_mean_squared_error"][:], 'r-', label='rmse')
plt.plot(hist["val_root_mean_squared_error"][:], 'b-', label='val-rmse')
plt.title('Training progress of the TDNN cy model')
plt.xlabel('epochs')
plt.ylabel('RMSE')
plt.legend()

time = np.arange(0.00, len(cytest)*0.01, step=0.01)     #timestep is gekend als 0.01, tijdsrange namaken | hier ineens wel 0 als beginwaarde gebruiken?
time.shape = (len(time), 1)
print(str(time.shape) +"|"+ str(cytest.shape))
assert(time.shape == cytest.shape)

plt.figure()
plt.plot(time, cytest, 'b--', linewidth=1, label='cy true values')
plt.plot(time, cypredtst, 'r:', linewidth=1.2, label='cy predicted values')
plt.plot(time, cytest-cypredtst, 'k-', linewidth=0.2, label='error')
plt.xlabel('Time (s)')
plt.ylabel('cy | error')
plt.suptitle('Test predictions and error of the trained model', fontsize=14)
plt.legend(loc='upper right')

plt.show()
'''
