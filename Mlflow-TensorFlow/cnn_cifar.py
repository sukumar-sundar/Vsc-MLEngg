import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt 
from tensorflow.keras.datasets import cifar10
import mlflow

print("Starting the CNN App")
# Collect the data and preprocess images.
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train/255.0
x_test = x_test/255.0

#Batvh Size 
batch_size = 128
learning_rate = 0.001
epochs = 5


print("Collected the data.")
def create_type1_model():
    try:
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation='relu', input_shape=[32,32,3]))
        model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
        model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))

        model.add(tf.keras.layers.Conv2D(filters=64, kernel_size =3, padding= 'same', activation='relu'))
        model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
        model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides =2, padding="valid"))

        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(units=128, activation='relu'))
        model.add(tf.keras.layers.Dense(units=10, activation='softmax'))
        model.summary()
    except Exception as exc:
        print("Caught in the exception at {0}".format(exc))
    return model


def create_type2_model():
    try:
        model2 = tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=[32, 32, 3]),
        tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'),
    
        tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'),
        tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(units=10, activation='softmax')
        ])
        model2.summary()
    except Exception as exc:
        print("Caught in the exception at {0}".format(exc))
    return model2


current_model = create_type2_model()

current_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

history = current_model.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test))

train_loss=history.history['loss'][-1]
train_acc=history.history['sparse_categorical_accuracy'][-1]
val_loss=history.history['val_loss'][-1]
val_acc=history.history['val_sparse_categorical_accuracy'][-1]

print("train_loss: ", train_loss)
print("train_accuracy: ", train_acc)
print("val_loss: ", val_loss)
print("val_accuracy: ", val_acc)

tf.keras.models.save_model(current_model, "./model")

with mlflow.start_run(run_name="CNN_Perf_Tuning"):
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("epochs", epochs)
    mlflow.log_metric("train_loss", train_loss)
    mlflow.log_metric("train_accuracy", train_acc)
    mlflow.log_metric("val_loss", val_loss)
    mlflow.log_metric("val_accuracy", val_acc)
    mlflow.log_artifacts("./model")


