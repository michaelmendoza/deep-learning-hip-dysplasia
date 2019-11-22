  
model0 = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation=tf.nn.relu, input_shape=(HEIGHT, WIDTH, CHANNELS)),
  tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation=tf.nn.relu),
  tf.keras.layers.MaxPooling2D(padding="same", strides=(2, 2), pool_size=(2, 2)),
  
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(128, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(NUM_OUTPUTS, activation=tf.nn.softmax)
])

model1 = tf.keras.models.Sequential([
tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation=tf.nn.relu, input_shape=(HEIGHT, WIDTH, CHANNELS)),
tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation=tf.nn.relu),
tf.keras.layers.MaxPooling2D(padding="same", strides=(2, 2), pool_size=(2, 2)),

tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation=tf.nn.relu),
tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation=tf.nn.relu),
tf.keras.layers.MaxPooling2D(padding="same", strides=(2, 2), pool_size=(2, 2)),

tf.keras.layers.Conv2D(256, (3, 3), padding="same", activation=tf.nn.relu),
tf.keras.layers.Conv2D(256, (3, 3), padding="same", activation=tf.nn.relu),
tf.keras.layers.MaxPooling2D(padding="same", strides=(2, 2), pool_size=(2, 2)),

tf.keras.layers.Conv2D(512, (3, 3), padding="same", activation=tf.nn.relu),
tf.keras.layers.Conv2D(512, (3, 3), padding="same", activation=tf.nn.relu),
tf.keras.layers.MaxPooling2D(padding="same", strides=(2, 2), pool_size=(2, 2)),

tf.keras.layers.Flatten(),
tf.keras.layers.Dense(4096, activation=tf.nn.relu),
tf.keras.layers.Dropout(0.2),
tf.keras.layers.Dense(4096, activation=tf.nn.relu),
tf.keras.layers.Dropout(0.2),
tf.keras.layers.Dense(NUM_OUTPUTS, activation=tf.nn.softmax)
])

model2 = tf.keras.models.Sequential([
tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation=tf.nn.relu, input_shape=(HEIGHT, WIDTH, CHANNELS)),
tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation=tf.nn.relu),
tf.keras.layers.BatchNormalization(),
tf.keras.layers.Dropout(0.2),
tf.keras.layers.MaxPooling2D(padding="same", strides=(2, 2), pool_size=(2, 2)),

tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation=tf.nn.relu),
tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation=tf.nn.relu),
tf.keras.layers.BatchNormalization(),
tf.keras.layers.Dropout(0.2),
tf.keras.layers.MaxPooling2D(padding="same", strides=(2, 2), pool_size=(2, 2)),

tf.keras.layers.Conv2D(256, (3, 3), padding="same", activation=tf.nn.relu),
tf.keras.layers.Conv2D(256, (3, 3), padding="same", activation=tf.nn.relu),
tf.keras.layers.BatchNormalization(),
tf.keras.layers.Dropout(0.2),
tf.keras.layers.MaxPooling2D(padding="same", strides=(2, 2), pool_size=(2, 2)),

tf.keras.layers.Conv2D(512, (3, 3), padding="same", activation=tf.nn.relu),
tf.keras.layers.Conv2D(512, (3, 3), padding="same", activation=tf.nn.relu),
tf.keras.layers.BatchNormalization(),
tf.keras.layers.Dropout(0.2),
tf.keras.layers.MaxPooling2D(padding="same", strides=(2, 2), pool_size=(2, 2)),

tf.keras.layers.Flatten(),
tf.keras.layers.Dense(4096, activation=tf.nn.relu),
tf.keras.layers.BatchNormalization(),
tf.keras.layers.Dropout(0.2),
tf.keras.layers.Dense(4096, activation=tf.nn.relu),
tf.keras.layers.BatchNormalization(),
tf.keras.layers.Dropout(0.2), 
tf.keras.layers.Dense(NUM_OUTPUTS, activation=tf.nn.softmax)
])



