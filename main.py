from __future__ import print_function

import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


def fbeta(y_true, y_pred, threshold_shift=0):
    beta = 1

    # just in case of hipster activation at the final layer
    y_pred = K.clip(y_pred, 0, 1)

    # shifting the prediction threshold from .5 if needed
    y_pred_bin = K.round(y_pred + threshold_shift)

    tp = K.sum(K.round(y_true * y_pred_bin), axis=1) + K.epsilon()
    fp = K.sum(K.round(K.clip(y_pred_bin - y_true, 0, 1)), axis=1)
    fn = K.sum(K.round(K.clip(y_true - y_pred, 0, 1)), axis=1)

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    beta_squared = beta ** 2
    return K.mean((beta_squared + 1) * (precision * recall) / (beta_squared * precision + recall + K.epsilon()))


num_classes = 8
img_rows, img_cols = 64, 64
BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 100

dir_path = os.getcwd()
data_path = dir_path + '/data/'

train_images = np.load(data_path + 'fer_train_processed_images.npy')
train_labels = np.load(data_path + 'fer_train_processed_labels.npy')
val_images = np.load(data_path + 'fer_val_processed_images.npy')
val_labels = np.load(data_path + 'fer_val_processed_labels.npy')
test_images = np.load(data_path + 'fer_test_processed_images.npy')
test_labels = np.load(data_path + 'fer_test_processed_labels.npy')

train_images = train_images.reshape(train_images.shape[0], img_rows, img_cols, 1).astype('float32') / 255
val_images = val_images.reshape(val_images.shape[0], img_rows, img_cols, 1).astype('float32') / 255
test_images = test_images.reshape(test_images.shape[0], img_rows, img_cols, 1).astype('float32') / 255

print(train_images.shape, train_labels.shape)
print(val_images.shape, val_labels.shape)
print(test_images.shape, test_labels.shape)

train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).batch(BATCH_SIZE).shuffle(SHUFFLE_BUFFER_SIZE).repeat()
val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_labels)).batch(BATCH_SIZE).repeat()
#test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(BATCH_SIZE)

model = Sequential()
# Feature Learning Layer 0
model.add(Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', activation='elu',
                 input_shape=(img_rows, img_cols, 1)))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', activation='elu',
                 input_shape=(img_rows, img_cols, 1)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(Dropout(0.25))

# Feature Learning Layer 1
model.add(Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal', activation='elu'))
model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal', activation='elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(Dropout(0.25))

# Feature Learning Layer 2
model.add(Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal', activation='elu'))
model.add(BatchNormalization())
model.add(Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal', activation='elu'))
model.add(BatchNormalization())
model.add(Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal', activation='elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(Dropout(0.25))

# Feature Learning Layer 3
model.add(Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal', activation='elu'))
model.add(BatchNormalization())
model.add(Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal', activation='elu'))
model.add(BatchNormalization())
model.add(Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal', activation='elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(Dropout(0.25))

# Feature Learning Layer 4
model.add(Flatten())
model.add(Dense(1024, kernel_initializer='he_normal', activation='elu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# Feature Learning Layer 5
model.add(Dense(1024, kernel_initializer='he_normal', activation='elu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# Feature Learning Layer 6
model.add(Dense(num_classes, kernel_initializer='he_normal', activation='softmax'))

print(model.summary())

#
# Creating Models
#

checkpoint = ModelCheckpoint('emotion_classification_vgg_7_emotions.h5', monitor='val_loss', mode='min', save_best_only=True, verbose=1)
early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=1, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, min_delta=0.0001, min_lr=0, cooldown=0)
tensor_board = TensorBoard(log_dir='./graph')
callbacks = [early_stop, checkpoint, reduce_lr, tensor_board]

model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy', fbeta])

epochs = 25
nb_train_samples = 25045
nb_validation_samples = 3191

history = model.fit(train_dataset, epochs=epochs, callbacks=callbacks, validation_data=val_dataset,
                    steps_per_epoch=nb_train_samples // BATCH_SIZE,
                    validation_steps=nb_validation_samples // BATCH_SIZE, verbose=1)

# EVALUATION
score = model.evaluate(test_images, test_labels, steps=len(test_images) // BATCH_SIZE)
print('Evaluation loss: ', score[0])
print('Evaluation accuracy: ', score[1])

# summarize history for accuracy
plt.plot(history.history['accuracy'], color='b', label='Training')
plt.plot(history.history['val_accuracy'], color='g', label='Validation')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.show()
plt.savefig("Accuracy.png")

# summarize history for loss
plt.plot(history.history['loss'], color='b', label='Training')
plt.plot(history.history['val_loss'], color='g', label='Validation')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='lower left')
plt.show()
plt.savefig('Loss.png')

y_pred = np.argmax(model.predict(test_images), axis=-1)
y_true = np.asarray([np.argmax(i) for i in test_labels])
emotion_labels = {0: 'neutral', 1: 'happiness', 2: 'surprise', 3: 'sadness', 4: 'anger', 5: 'disgust', 6: 'fear', 7: 'contempt'}

cm = confusion_matrix(y_true, y_pred)
cm_normalised = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.set(font_scale=1.5)
fig, ax = plt.subplots(figsize=(10, 10))
ax = sns.heatmap(cm_normalised, annot=True, linewidths=0, square=False, cmap="Blues", yticklabels=emotion_labels,
                 xticklabels=emotion_labels, vmin=0, vmax=np.max(cm_normalised), fmt=".2f", annot_kws={"size": 20})
ax.set(xlabel='Predicted label', ylabel='True label')
plt.show()
plt.savefig('conf-mat.png')

# model.save("/SavedModel", overwrite=True)
model_json = model.to_json()
with open("emotion_classification_vgg_7_emotions.json", "w") as json_file:
    json_file.write(model_json)
