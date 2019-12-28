from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
from csv_writer_reader import CSV_Writer_Reader
# load dataset
print('tf version : ', tf.__version__)
pathin = '/home/zheng/ws_xiao/gym_test/gym_pre_arms/inputX.csv'
pathout = '/home/zheng/ws_xiao/gym_test/gym_pre_arms/outputY.csv'
# split into input (X) and output (Y) variables
csvrw = CSV_Writer_Reader()
X = csvrw.readcsv(filepath=pathin)
Y = csvrw.readcsv(filepath=pathout)
print(X, Y)
dimin = len(X)
dimou = len(Y)
x = np.array(X)
y = np.array(Y)
print(dimin, dimou, x.shape)
x_train = tf.convert_to_tensor(x[1:900,:], dtype=tf.float32)
y_train = tf.convert_to_tensor(y[1:900,:], dtype=tf.float32)
x_test = tf.convert_to_tensor(x[901:999,:], dtype=tf.float32)
y_test = tf.convert_to_tensor(y[901:999,:], dtype=tf.float32)

#print(len(x_train), len(y_train), len(x_test), y_test)

model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(15, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss=tf.keras.losses.MeanSquaredError(),
              metrics=[tf.keras.metrics.Accuracy()])

model.fit(x_train, y_train,
          batch_size=32,
          epochs=int(1e6))

model.evaluate(x_test, y_test, verbose=1)


'''
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

mnist = tf.keras.datasets.mnist


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Add a channels dimension
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(10000).batch(32)

test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

class MyModel(Model):
  def __init__(self):
    super(MyModel, self).__init__()
    self.conv1 = Conv2D(32, 3, activation='relu')
    self.flatten = Flatten()
    self.d1 = Dense(128, activation='relu')
    self.d2 = Dense(10, activation='softmax')

  def call(self, x):
    x = self.conv1(x)
    x = self.flatten(x)
    x = self.d1(x)
    return self.d2(x)

# Create an instance of the model
model = MyModel()

loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

optimizer = tf.keras.optimizers.Adam()
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

@tf.function
def train_step(images, labels):
  with tf.GradientTape() as tape:
    predictions = model(images)
    loss = loss_object(labels, predictions)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  train_loss(loss)
  train_accuracy(labels, predictions)

@tf.function
def test_step(images, labels):
  predictions = model(images)
  t_loss = loss_object(labels, predictions)

  test_loss(t_loss)
  test_accuracy(labels, predictions)

EPOCHS = 5

for epoch in range(EPOCHS):
  # Reset the metrics at the start of the next epoch
  train_loss.reset_states()
  train_accuracy.reset_states()
  test_loss.reset_states()
  test_accuracy.reset_states()

  for images, labels in train_ds:
    train_step(images, labels)

  for test_images, test_labels in test_ds:
    test_step(test_images, test_labels)

  template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
  print(template.format(epoch+1,
                        train_loss.result(),
                        train_accuracy.result()*100,
                        test_loss.result(),
                        test_accuracy.result()*100))





# Regression Example With Boston Dataset: Baseline
from pandas import read_csv
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

from csv_writer_reader import CSV_Writer_Reader
# load dataset
pathin = '/home/zheng/ws_xiao/gym_test/gym_pre_arms/inputX.csv'
pathout = '/home/zheng/ws_xiao/gym_test/gym_pre_arms/outputY.csv'
# split into input (X) and output (Y) variables
csvrw = CSV_Writer_Reader()
X = csvrw.readcsv(filepath=pathin)
Y = csvrw.readcsv(filepath=pathout)
print(X, Y)
dimin = len(X)
dimou = len(Y)
x = np.array(X)
y = np.array(Y)
print(dimin, dimou, x.shape)
# define base model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(128, kernel_initializer='normal', activation='relu'))
	model.add(Dense(15, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model
# evaluate model
estimator = KerasRegressor(build_fn=baseline_model, epochs=1000, batch_size=5, verbose=1)
kfold = KFold(n_splits=10)
results = cross_val_score(estimator, x, y, cv=kfold)
print("Baseline: %.2f (%.2f) MSE" % (results.mean(), results.std()))
'''