from pathlib import Path
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from tensorflow.keras import layers

from keras.layers import Dense
from keras.models import Sequential

RETRAIN = False

# Regression Network
images_dir_path = Path('images/')
pretrain_bit = "https://tfhub.dev/google/bit/s-r50x1/1"
pretrain_resnet50 = "https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/4"

IMAGE_SHAPE = (256, 256)

# Load feature vector extractor module
module = hub.KerasLayer(pretrain_bit, input_shape=(IMAGE_SHAPE[1], IMAGE_SHAPE[0], 3))

# Load dataframe
df_merged = pd.read_csv('train_data_square_image.csv')

# Generate Image Batches
bs = 32
coords = ['r_ankle_x', 'r_knee_x', 'r_hip_x', 'l_hip_x', 'l_knee_x', 'l_ankle_x', 'pelvis_x', 'thorax_x',
          'upper_neck_x','head_top_x', 'r_wrist_x', 'r_elbow_x', 'r_shoulder_x', 'l_shoulder_x', 'l_elbow_x',
          'l_wrist_x', 'r_ankle_y', 'r_knee_y', 'r_hip_y', 'l_hip_y', 'l_knee_y', 'l_ankle_y', 'pelvis_y',
          'thorax_y', 'upper_neck_y', 'head_top_y', 'r_wrist_y', 'r_elbow_y', 'r_shoulder_y', 'l_shoulder_y',
          'l_elbow_y', 'l_wrist_y']
image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)
train_generator = image_generator.flow_from_dataframe(dataframe=df_merged, directory=None, x_col='image_file_path',
                                                      y_col=coords, has_ext=True, class_mode="other",
                                                      target_size=(IMAGE_SHAPE[1], IMAGE_SHAPE[0]), batch_size=bs)
x, y = next(train_generator)
feature_batch = module(x)

# Print batch filenames
for i in train_generator:
    idx = (train_generator.batch_index - 1) * train_generator.batch_size
    print(train_generator.filenames[idx: idx + train_generator.batch_size])

# Freeze feature extractor weight training
module.trainable = False

# Add regression output layer
model = tf.keras.Sequential([module, layers.Dense(32)])
model.summary()

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['acc'])
steps_per_epoch = np.ceil(train_generator.samples/train_generator.batch_size)
history = model.fit(train_generator, epochs=2, steps_per_epoch=steps_per_epoch)

keypts_pred = model.predict(train_generator)

# Save predicted keypoints to use in the activity classifier network
keypts_pred_df = pd.DataFrame(keypts_pred)
keypts_pred_df.to_csv('regnet_output.csv')

# Save trained model
model.save('regnet', save_format='tf')

# Load saved model
if not RETRAIN:
    model = tf.keras.models.load_model('regnet')

# Classification Network
trainX = keypts_pred
trainY = pd.get_dummies(df_merged['category_name'], drop_first=True).values

# Network Architecture
class_model = Sequential()
class_model.add(Dense(100, activation='relu', input_shape=(32,)))
class_model.add(Dense(100, activation='relu'))
class_model.add(Dense(100, activation='relu'))
class_model.add(Dense(20, activation='softmax'))

class_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
class_model.fit(trainX, trainY)
