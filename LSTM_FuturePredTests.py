import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras as K
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, Dropout
from keras.utils import plot_model
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf  # toegevoegd voor het laden
physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

plt.close('all')

nm = 0  #for ifs cycles, if data already simulated use nm=0.
FutPred = 0    #want to predict 10 time steps  into the future

# Custom metric
def rel_rms(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred-y_true)))/K.sqrt(K.mean(K.square(y_true-K.mean(y_true, axis=0))))

#same idea but with different libraries (numpy and keras)
def rel_rms_np(y_true, y_sim):
    return np.sqrt(np.mean(np.square(y_true-y_sim)))/np.sqrt(np.mean(np.square(y_true-np.mean(y_true))))


# Load data
path = 'C:\\Users\\Ignacio\\Desktop\\TFM\\FFNN_Test\\'
Utrain = pd.read_csv(path+'Utrain.csv', header=None).to_numpy()
Utest = pd.read_csv(path+'Utest.csv', header=None).to_numpy()
Udev = pd.read_csv(path+'Udev.csv', header=None).to_numpy()

cytrain = pd.read_csv(path + 'ftrain.csv', header=None).to_numpy()
cytest = pd.read_csv(path + 'ftest.csv', header=None).to_numpy()
cydev = pd.read_csv(path + 'fdev.csv', header=None).to_numpy()

# Option for selecting data
UTrain = Utrain[:, 0:21]    # number of inputs #All data from rows, first 21 columns(all, as there are 21 probes)
yTrain = cytrain

# PLOT Utrain

UTest = Utest[:, 0:21]
yTest = cytest

UDev = Udev[:, 0:21]
yDev = cydev

time_steps = 160  # let state evolve over 50 ts
batch_sizef = UTrain.shape[0]-time_steps+1-FutPred  # -10 for future prediction     #.shape returns shape of array; .shape[0] returns dimensions for rows
#LSTM batch size: from u(1) to u(N-m+1)  (see page 37)
features = 21
LSTM_neurons = 20  # use 30 outputs which are recombined using a dense net afterwords     #LSTM neurons

# Initialise UTrain for LSTM: Each row contains x time_steps of the features on the 2nd and 3rd dimensions
UTrain_LSTM = np.zeros((batch_sizef, time_steps, features))
#print(UTrain_LSTM)

for i in range(batch_sizef):
    UTrain_LSTM[i, :, :] = UTrain[i:i+time_steps, :]       # [0-9],[1-10],[2-11],...
# Input data structured into 3D matrix using the 2D input structure
# of the m previous input time steps for every output time step (page 37)

yTrain_LSTM = yTrain[time_steps-1+FutPred:]     # +10 for future prediction  # and same for development and testing data
                                        # Cut off first time steps of output so that output matches last time step of input
# Idem for validation data
time_steps_dev = time_steps  # must be equal!
batch_size_dev = UDev[:].shape[0]-time_steps_dev+1-FutPred
UDev_LSTM = np.zeros((batch_size_dev, time_steps_dev, features))

for i in range(batch_size_dev):
    UDev_LSTM[i, :, :] = UDev[i:i+time_steps_dev, :]  # [50,21]

yDev_LSTM = yDev[time_steps_dev-1+FutPred:] #+10 for future prdiction

# Testing data
time_steps_test = time_steps  # must be equal!
batch_size_test = UTest[:].shape[0]-time_steps_test+1-FutPred
UTest_LSTM = np.zeros((batch_size_test, time_steps_test, features))

for i in range(batch_size_test):
    UTest_LSTM[i, :, :] = UTest[i:i+time_steps_test, :]

yTest_LSTM = yTest[time_steps_test-1+FutPred:]  #1st output: row 49 (0+50-1) is 1st output (hm)

# Initialise model
if nm:
    n_epoch = 3000
    optim = K.optimizers.Adam()
    lossf = K.losses.MeanSquaredError()  # mean_squared_error
    metric = [K.metrics.RootMeanSquaredError(), K.metrics.MeanAbsoluteError()]
    regul = K.regularizers.l2(0.001)
    batch_size = 128

    model = K.Sequential()
    model.add(LSTM(LSTM_neurons, input_shape=(time_steps, features),
                   return_sequences=False, stateful=False, name='LSTM1'))
    model.add(Dense(30, activation='relu', name='layer6'))
    model.add(Dense(1, activation='linear', name='outputLayer'))

    model.compile(optimizer='Adam', loss='mse', metrics=metric)
    earlystop = K.callbacks.EarlyStopping(monitor='val_loss', patience=20, verbose=1)

    # Create model
    import time
    start_time = time.time()
    runfit = model.fit(UTrain_LSTM, yTrain_LSTM,
                       batch_size=batch_size,
                       epochs=n_epoch,
                       shuffle=True,
                       callbacks=earlystop,
                       validation_data=(UDev_LSTM, yDev_LSTM))  # ,validation_data=(UVal,yVal),validation_steps=5)
    training_time = time.time() - start_time

    #AUTOMATE IT FOR NAMES
    model.save('LSTM_' + str(time_steps) +'ts_' + str(FutPred) + 'FutPred', save_format='h5')
    hist = pd.DataFrame(runfit.history)    # progress printen
    hist.to_csv('LSTM_' + str(time_steps) +'ts_' + str(FutPred) + 'FutPred' '_hist.csv')

# Load model
else:
    model = K.models.load_model('LSTM_' + str(time_steps) +'ts_' + str(FutPred) + 'FutPred')

# Use and test model
yTrain_sim = model.predict(UTrain_LSTM)

yTest_sim = model.predict(UTest_LSTM)
model_eval = model.evaluate(UTest_LSTM, yTest_LSTM)

# Metrics
rel_RMSEtr = rel_rms_np(yTrain_sim, yTrain_LSTM)

yDev_sim = model.predict(UDev_LSTM)

rel_RMSEdev = rel_rms_np(yDev_sim, yDev_LSTM)

print('Tr Rel rms: ' + str(rel_rms_np(yTrain_LSTM, yTrain_sim)))
print('Tst Rel rms: ' + str(rel_rms_np(yTest_LSTM, yTest_sim)))

# Saved Metrics
num_trainsamples = len(yTrain_LSTM)
num_testsamples = len(yTest_LSTM)
num_params = model.count_params()
err = (yTest_sim-yTest_LSTM)

abs_err = np.abs(err)           # abs = absolute waarde (alles >=0 en element |R)
avg_err = np.mean(err)
std_err = np.std(err)
loss = np.sum(err**2)/num_testsamples       # MSE
rel_err = loss/(np.sum(np.abs(yTest_LSTM)**2)/num_testsamples)   # MSE/Mean Squared juiste waarde
print(str(rel_err))
print(str(np.sum(err**2)/np.sum(yTest_LSTM**2)))
print(str(np.sqrt(np.mean(np.square(yTest_LSTM-np.mean(yTest_LSTM))))))
print(str(np.sqrt(np.mean(np.square(yTest_LSTM)))))
rel_RMSE = np.sqrt(rel_err)
print(rel_RMSE)

##########################################
if nm:
    #Setting initial value of the counter to zero
    rowcount = -1
    #iterating through the whole file
    for row in open('LSTM_' + str(time_steps) +'ts_' + str(FutPred) + 'FutPred' + '_hist.csv'):
      rowcount+= 1
     #printing the result
    print(rowcount)
    ##########################################
    data = [FutPred,rel_RMSE,rowcount,training_time]
    print(data)
    from csv import writer
    with open('LSTM_FutureResults.csv', 'a') as f_object:
        writer_object = writer(f_object)
        writer_object.writerow(data)
        f_object.close()
############################################

if nm:  #TABLE
    LSTMModel_perfo = {'Model activations': 'ReLU-Linear',
                       'Number of training samples': num_trainsamples,
                       'Number of model parameters': num_params,
                       'Ratio training samples-parameters': num_trainsamples/num_params,
                       'Model loss on test data': model_eval[0],
                       'Model RMSE on test data': model_eval[1],
                       # 'Model Absolute error on test data': model_eval[2],
                       'Number of test samples': num_testsamples,
                       'Average Error': avg_err,
                       'Standard Deviation of Error': std_err,
                       'Average Absolute Error': np.mean(abs_err),
                       'Maximum Error': np.max(abs_err),
                       'Loss': loss,
                       'Relative Error': rel_err,
                       'Relative RMSE on train set': rel_RMSEtr,
                       'Relative RMSE on validation set': rel_RMSEdev,
                       'Relative RMSE on test set': rel_RMSE}

    ModelPerfo = pd.DataFrame.from_dict(LSTMModel_perfo, orient='index')
    ModelPerfo.index.name = 'LSTM_' + str(time_steps) +'ts_' + str(FutPred) + 'FutPred' + '_perfo'
    ModelPerfo.to_csv('LSTM_' + str(time_steps) +'ts_' + str(FutPred) + 'FutPred' + '_perfo.csv', header=['Value'])
    #ModelPerfo.to_latex('LSTMM_f_50ts' + str(name) + '_perfo.tex', header=['Value'])

# Plots
hist = pd.read_csv('LSTM_' + str(time_steps) +'ts_' + str(FutPred) + 'FutPred' + '_hist.csv')
plt.plot(hist["root_mean_squared_error"][:], 'r-', label='rmse')
plt.plot(hist["val_root_mean_squared_error"][:], 'b-', label='val-rmse')
plt.legend()

time = np.arange(0.00, len(yTest_LSTM)*0.01, step=0.01) #x0.01 bc 1TS=0.01s
timel = np.arange(0.00, len(yTest)*0.01, step=0.01)
timec = timel[time_steps-1+FutPred:]  #+FutPred?
timecutoff = timel[-20:]        #????????
time.shape = (len(time), 1)
print(str(time.shape) + "|" + str(yTest_LSTM.shape))
#assert(time.shape == yTest_LSTM.shape)

fig = plt.figure()
plt.plot(timel, yTest, 'b--', label='y true values')
plt.plot(timec, yTest_sim, 'r:', label='y predicted values')
plt.grid(which='both')
plt.plot(timec, err, 'k', label='Error')
plt.xlabel('Time (s)')
plt.ylabel('$C_y$ | error')
plt.grid(which='both')
plt.legend()

# different plots
if 0:
    fig, axs = plt.subplots(2, 1, sharex=True)                    # nu wordt displacement geplot
    axs[0].plot(timel, yTest, 'b-', label='$c_y$ true values')
    axs[0].plot(timec, yTest_simc, 'r:', label='$c_y$ predicted values')
    axs[0].plot(timecutoff, yTest_simcutoff, 'g:', label='cut off samples')
    axs[0].grid(b=1, which='both')
    axs[1].plot(timec, err, 'k', label='Error')
    plt.setp(axs[-1], xlabel='Time (s)', ylabel='Error')
    axs[1].grid(b=1, which='both')
    fig.suptitle('Predictions and error of the trained model', fontsize=14)
    fig.legend(loc='upper right')

    plt.rcParams.update({'font.size': 15})
    fig, axs = plt.subplots(2, 1, sharex=True)                  # nu wordt displacement geplot
    axs[0].plot(timel[:-20], yTest[:-20], 'b-', label='Target values')
    axs[0].plot(timec, yTest_simc, 'r:', label='Predicted values')
    axs[1].plot(timec, err, 'k', label='Error')
    plt.setp(axs[0], ylabel='$c_f$')
    plt.setp(axs[1], xlabel='Time (s)', ylabel='Error')
    axs[0].grid(b=1, which='major')
    axs[1].grid(b=1, which='major')
    axs[0].legend(loc='upper right')
    axs[0].title.set_text('$c_y$')
    axs[1].title.set_text('Error')
    fig.suptitle('Force model test results', fontsize=20)

plt.show()
