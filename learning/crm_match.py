import pandas as pd
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

matplotlib.use('TkAgg')

df = pd.read_csv('~/Downloads/crm_matched.csv')
df = df[df.distance < 1000]

data = df[['name', 'address', 'distance']]
label = df['matched']

training_data, test_data = data[0:200000].values, data[200000:].values
training_label, test_label = label[0:200000].values, label[200000:].values

print(training_data.shape)
print(training_data, training_label)


model = keras.Sequential([
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(2, activation=tf.nn.softmax)
])

model.compile(
    optimizer=tf.train.AdamOptimizer(),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.fit(training_data, training_label, epochs=5)
test_loss, test_acc = model.evaluate(test_data, test_label)

predictions = model.predict(test_data)

print(predictions)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

correct = []
wrong = []
for i in range(0, len(test_data), 1):
    prediction_array = predictions[i]
    result = np.argmax(prediction_array)

    if result == test_label[i]:
        correct.append(test_data[i])
    else:
        wrong.append(test_data[i])

print( 1.0 * len(wrong) / (len(wrong) + len(correct)))
#ax.scatter(*zip(*correct), color='blue', marker='^', s=1)
ax.scatter(*zip(*wrong), color='red', marker='o', s=2)
print('done')

ax.set_xlabel('name')
ax.set_ylabel('address')
ax.set_zlabel('distance')
plt.show()
