import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from maze_logic.maze_solver_determine_correct import evaluate_total_predictions_set

np.set_printoptions(suppress=True)
X_test = np.load("dataset/X.dat_test.npy")
Y_test = np.load("dataset/Y.dat_test.npy")

X_train = np.load("dataset/X.dat.npy")
Y_train = np.load("dataset/Y.dat.npy")

flattenl_ = tf.keras.layers.Flatten()

X_train = flattenl_(X_train)
Y_train = flattenl_(Y_train)
print(X_train.shape)
print(Y_train.shape)

X_test = flattenl_(X_test)
Y_test = flattenl_(Y_test)
print(X_test.shape)
print(Y_test.shape)

plt.title('Loss over epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

models = [
    tf.keras.models.load_model('model/trained/ms.h5'),
    tf.keras.models.load_model('model/trained/ms-2.h5'),
    tf.keras.models.load_model('model/trained/ms-3.h5'),
    tf.keras.models.load_model('model/trained/ms-4.h5'),
    tf.keras.models.load_model('model/trained/ms-5.h5'),
    tf.keras.models.load_model('model/trained/ms-6.h5'),
    tf.keras.models.load_model('model/trained/ms-9.h5'),
    tf.keras.models.load_model('model/trained/ms-10.h5'),
    tf.keras.models.load_model('model/collaboratory/ms-13-collab.h5'),
    tf.keras.models.load_model('model/collaboratory/m_300.h5'),
]

predictions = []
for model in models:
    predictions.append(model.predict(X_test))
    model.summary()


mse = tf.keras.losses.MeanSquaredError()
for prediction in predictions:
    print("MSE for Pn: ", mse(Y_test, prediction))

def plot_accuracy():
    pc_1 = evaluate_total_predictions_set(predictions[0], X_test)
    pc_2 = evaluate_total_predictions_set(predictions[1], X_test)
    pc_3 = evaluate_total_predictions_set(predictions[2], X_test)
    pc_4 = evaluate_total_predictions_set(predictions[3], X_test)
    pc_5 = evaluate_total_predictions_set(predictions[4], X_test)
    pc_6 = evaluate_total_predictions_set(predictions[5], X_test)
    pc_7 = evaluate_total_predictions_set(predictions[6], X_test)
    pc_8 = evaluate_total_predictions_set(predictions[7], X_test)
    pc_9 = evaluate_total_predictions_set(predictions[8], X_test)
    pc_10 = evaluate_total_predictions_set(predictions[9], X_test)

    # PLOT ACCURACY
    data_accuracy = {
        'Small/RELU': pc_1,
        '+2L/RELU': pc_2,
        'Sigmoid pre-out': pc_3,
        'More data': pc_4,
        'Sigmoid': pc_5,
        'More Layers/Linear': pc_6,
        'v2': pc_7,
        'v3': pc_8,
        'Large NN': pc_9,
        'Large NN(300ep) + L2 reg': pc_10,
    }

    models = list(data_accuracy.keys())
    values = list(data_accuracy.values())

    fig = plt.figure(figsize=(15, 5))
    plt.bar(models, values, color='maroon',
            width=0.4)

    plt.xlabel("Models")
    plt.ylabel("% Accuracy")
    plt.title("Measuring model accuracy (AMSGRAD, 500 epochs, 64 batch)")
    plt.show()


plot_accuracy()

start_ind = 2142
for i in range(10):
    fig, axes = plt.subplots(3, 4, figsize=(15, 9))
    fig.suptitle('Model comparison (AMSGRAD, 500 epochs, 64 batch)', fontsize=16)

    axes[0, 0].imshow(predictions[0][i + start_ind].reshape((7, 7)))
    axes[0, 0].axis('off')
    axes[0, 0].set_title('Small NN / RELU')

    axes[0, 1].imshow(predictions[1][i + start_ind].reshape((7, 7)))
    axes[0, 1].axis('off')
    axes[0, 1].set_title('+2 Layers / RELU')

    axes[0, 2].imshow(predictions[2][i + start_ind].reshape((7, 7)))
    axes[0, 2].axis('off')
    axes[0, 2].set_title('Sigmoid pre-out norm / RELU')

    axes[0, 3].imshow(predictions[3][i + start_ind].reshape((7, 7)))
    axes[0, 3].axis('off')
    axes[0, 3].set_title('Trained thicker & sparse mazes / RELU')

    axes[1, 0].imshow(predictions[4][i + start_ind].reshape((7, 7)))
    axes[1, 0].axis('off')
    axes[1, 0].set_title('Sigmoid')

    axes[1, 1].imshow(predictions[5][i + start_ind].reshape((7, 7)))
    axes[1, 1].axis('off')
    axes[1, 1].set_title('More Layers / Linear')

    axes[1, 2].imshow(predictions[6][i + start_ind].reshape((7, 7)))
    axes[1, 2].axis('off')
    axes[1, 2].set_title('V2 + Sigmoid pre-out norm/Linear out')

    axes[1, 3].imshow(predictions[7][i + start_ind].reshape((7, 7)))
    axes[1, 3].axis('off')
    axes[1, 3].set_title('More Layers V3 / Linear out')

    axes[2, 0].imshow(predictions[8][i + start_ind].reshape((7, 7)))
    axes[2, 0].axis('off')
    axes[2, 0].set_title('Large NN(30ep)')

    axes[2, 1].imshow(predictions[9][i + start_ind].reshape((7, 7)))
    axes[2, 1].axis('off')
    axes[2, 1].set_title('Large NN(300ep) + L2 Reg ')

    axes[2, 2].imshow(X_test[i + start_ind].numpy().reshape((7, 7)))
    axes[2, 2].axis('off')
    axes[2, 2].set_title('TASK')

    axes[2, 3].imshow(Y_test[i + start_ind].numpy().reshape((7, 7)))
    axes[2, 3].axis('off')
    axes[2, 3].set_title('TASK ANSWER')

    plt.show()
