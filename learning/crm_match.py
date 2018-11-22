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

training_data_size = 150000

training_data, test_data = data[0:training_data_size].values, data[training_data_size:].values
training_label, test_label = label[0:training_data_size].values, label[training_data_size:].values

print(training_data.shape)
print(training_data, training_label)


model = keras.Sequential([
    keras.layers.Dense(3, activation=tf.nn.relu),
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

positive = []
negative = []
true_positive = []
true_negative = []
false_positive = []
false_negative = []
for i in range(0, len(test_data), 1):
    prediction_array = predictions[i]
    result = np.argmax(prediction_array)
    d = test_data[i]
    if test_label[i]:
        positive.append(d)
        if result:
            true_positive.append(d)
        else:
            false_negative.append(d)
    else:
        negative.append(d)
        if result:
            false_positive.append(d)
        else:
            true_negative.append(d)

print( 1.0 * len(negative) / (len(negative) + len(positive)))
# ax.scatter(*zip(*positive), color='blue', marker='^', s=2)
# ax.scatter(*zip(*negative), color='red', marker='o', s=2)
# ax.scatter(*zip(*true_positive), color='red', marker='o', s=2)
# ax.scatter(*zip(*true_negative), color='red', marker='o', s=2)
# ax.scatter(*zip(*false_positive), color='red', marker='o', s=2)
ax.scatter(*zip(*false_negative), color='red', marker='o', s=2)
print('done')

ax.set_xlabel('name')
ax.set_ylabel('address')
ax.set_zlabel('distance')
plt.show()
