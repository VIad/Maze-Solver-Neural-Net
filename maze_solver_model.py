import tensorflow as tf
import numpy as np

np.set_printoptions(suppress=True)

X_train = np.load("dataset/X.dat.npy")
Y_train = np.load("dataset/Y.dat.npy")

# CV_X
X_test = np.load("dataset/X.dat_smol.npy")
# CV_Y
Y_test = np.load("dataset/Y.dat_smol.npy")

flattenl_ = tf.keras.layers.Flatten()

X_train = flattenl_(X_train)
Y_train = flattenl_(Y_train)
print(X_train.shape)
print(Y_train.shape)

X_test = flattenl_(X_test)
Y_test = flattenl_(Y_test)
print(X_test.shape)
print(Y_test.shape)

model = tf.keras.models.Sequential([
        tf.keras.Input(shape=(49,)),
        tf.keras.layers.Dense(units=1235, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.0000050)),
        tf.keras.layers.Dense(units=768, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.0000045)),
        tf.keras.layers.Dense(units=532, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.0)),
        tf.keras.layers.Dense(units=149, activation='relu'),
        tf.keras.layers.Dense(units=98, activation='relu'),
        tf.keras.layers.Dense(units=49, activation='linear'),
    ])

model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=tf.keras.optimizers.Adam(amsgrad=True))

model.fit(X_train, Y_train, epochs=300, batch_size=64, verbose=1)

