import tensorflow as tf
import numpy as np

np.set_printoptions(suppress=True)
X_test = np.load("../../dataset/X.dat_smol.npy")
Y_test = np.load("../../dataset/Y.dat_smol.npy")

X_train = np.load("../../dataset/X.dat.npy")
Y_train = np.load("../../dataset/Y.dat.npy")

flattenl_ = tf.keras.layers.Flatten()

X_train = flattenl_(X_train)
Y_train = flattenl_(Y_train)
print(X_train.shape)
print(Y_train.shape)

X_test = flattenl_(X_test)
Y_test = flattenl_(Y_test)
print(X_test.shape)
print(Y_test.shape)

# This slows down training as it performs inference on each epoch, to determine the CV set's MSE to later plot and determine
# How to further optimize the model
class MeasureTestMSE(tf.keras.callbacks.Callback):
    def __init__(self, x_test, x_train, y_test, y_train):
        self.x_test = x_test
        self.y_test = y_test
        self.x_train = x_train
        self.y_train = y_train
        self.mse_test_log = []
        self.mse_train_log = []

    def on_epoch_end(self, epoch, logs=None):
        y_pred = self.model.predict(self.x_test, verbose=0)
        y_pred_train = self.model.predict(self.x_train, verbose=0)
        mse = tf.keras.losses.MeanSquaredError()
        self.mse_test_log.append(mse(self.y_test, y_pred).numpy())
        self.mse_train_log.append(mse(self.y_train, y_pred_train).numpy())

    def on_train_end(self, logs=None):
        print("TRAIN END")
        print("TRAIN MSE")
        print(self.mse_train_log)
        print("TEST MSE")
        print(self.mse_test_log)


attempt_descriptions = [
    "Reg 0.5, 0.45",
    "Reg 0.5, 0.45, 0.005",
    "Reg 0.5, 0.45, 0.015",
    "Reg 0.5, 0.45, 0.30",
    "Reg 0.5, 0.45, 0.45"
]

attempt_data_vals = [
    {
        "reg_L1": 0.0000050,
        "reg_L2": 0.0000045,
        "reg_L3": 0.0,
        "epochs" : 50
    },
    {
        "reg_L1": 0.0000050,
        "reg_L2": 0.0000045,
        "reg_L3": 0.0000005,
        "epochs" : 50
    },
    {
        "reg_L1": 0.0000050,
        "reg_L2": 0.0000045,
        "reg_L3": 0.0000015,
        "epochs" : 50
    },
    {
        "reg_L1": 0.0000050,
        "reg_L2": 0.0000045,
        "reg_L3": 0.0000030,
        "epochs" : 50
    },
    {
        "reg_L1": 0.0000050,
        "reg_L2": 0.0000045,
        "reg_L3": 0.0000045,
        "epochs" : 50
    },
]

# I pay for the whole Colab im gonna use the whole Colab
for i in range(len(attempt_descriptions)):
    print("\n\nBEGIN TRAIN-----------------------------")
    print(attempt_descriptions[i])
    model = tf.keras.models.Sequential([
        tf.keras.Input(shape=(49,)),
        tf.keras.layers.Dense(units=1235, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(attempt_data_vals[i]['reg_L1'])),
        tf.keras.layers.Dense(units=768, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(attempt_data_vals[i]['reg_L2'])),
        tf.keras.layers.Dense(units=532, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(attempt_data_vals[i]['reg_L3'])),
        tf.keras.layers.Dense(units=149, activation='relu'),
        tf.keras.layers.Dense(units=98, activation='relu'),
        tf.keras.layers.Dense(units=49, activation='linear'),
    ])
    #
    model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=tf.keras.optimizers.Adam(amsgrad=True))

    history = model.fit(X_train, Y_train, epochs=attempt_data_vals[i]['epochs'], batch_size=64,
                        callbacks=[MeasureTestMSE(X_test, X_train, Y_test, Y_train)], verbose=0)
    model.save('m-',i,'.h5')
    print("HISTORY FOR MODEL ", i)
    print(history.history['loss'])
    print("\n")
