# Import libraries. You may or may not use all of these.
!pip install -q git+https://github.com/tensorflow/docs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
  # %tensorflow_version only exists in Colab.
  %tensorflow_version 2.x
except Exception:
  pass
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling

# Import data
!wget https://cdn.freecodecamp.org/project-data/health-costs/insurance.csv
dataset = pd.read_csv('insurance.csv')
dataset.tail()

df = dataset.copy()
val, _ = df['sex'].factorize()
df['sex'] = val
val, _ = df['smoker'].factorize()
df['smoker'] = val
val, _ = df['region'].factorize()
df['region'] = val

train_dataset = df.sample(frac=0.8, random_state=1)
test_dataset = df.drop(train_dataset.index)

train_labels = train_dataset.pop('expenses')
test_labels = test_dataset.pop('expenses')

# create model
normalizer = layers.Normalization()
normalizer.adapt(np.array(train_dataset))
model = keras.Sequential([
              normalizer,
              layers.Dense(16, activation='relu'),
              layers.Dense(units=1)                                     
])
optimizer = keras.optimizers.Adam(learning_rate=0.1)
model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
model.summary()

history = model.fit(train_dataset, train_labels,
                    epochs=1000,
                    validation_split=0.2,
                    verbose=0)

# RUN THIS CELL TO TEST YOUR MODEL. DO NOT MODIFY CONTENTS.
# Test model by checking how well the model generalizes using the test set.
loss, mae, mse = model.evaluate(test_dataset, test_labels, verbose=2)

print("Testing set Mean Abs Error: {:5.2f} expenses".format(mae))

if mae < 3500:
  print("You passed the challenge. Great job!")
else:
  print("The Mean Abs Error must be less than 3500. Keep trying.")

# Plot predictions.
test_predictions = model.predict(test_dataset).flatten()

a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True values (expenses)')
plt.ylabel('Predictions (expenses)')
lims = [0, 50000]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims,lims)
