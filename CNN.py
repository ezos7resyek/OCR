import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Activation, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import RMSprop, SGD
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array



train_datagen = ImageDataGenerator(rescale = 1./255)
validation_datagen = ImageDataGenerator(rescale = 1./255)
train_generator = train_datagen.flow_from_directory(r'C:\Users\shari\Downloads\Digits\Train', class_mode="categorical", batch_size=64, target_size=(32, 32), color_mode="grayscale")
validation_generator = validation_datagen.flow_from_directory(r'C:\Users\shari\Downloads\Digits\Validation', class_mode="categorical", batch_size=64, target_size=(32, 32), color_mode="grayscale")


model = Sequential()

#Layer1
model.add(Conv2D(32, (5, 5), strides=1, activation='relu', kernel_regularizer=l2(0.0005), input_shape=(32, 32, 1)))

#Layer2
model.add(Conv2D(32, (5, 5), strides=1, activation='relu', use_bias=False))

#Layer3
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.1))

#Layer4
model.add(Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.0005)))

#Layer5
model.add(Conv2D(64, (3, 3), activation='relu', use_bias=False))

#Layer6
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())

model.add(Dense(512, activation='relu', use_bias=False))
model.add(BatchNormalization())
model.add(Activation("relu"))

#Layer11
model.add(Dense(256, activation='relu', use_bias=False))

#Layer12
model.add(BatchNormalization())
model.add(Activation("relu"))


#Layer13
model.add(Dense(128, activation='relu', use_bias=False))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Dropout(0.2))


model.add(Dense(62, activation='softmax'))
opt = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()



#my_callback = ReduceLROnPlateau(monitor='val_accuracy', patience=3, factor=0.5, min_lr=0.00001)

history = model.fit(train_generator, epochs=15, steps_per_epoch=200, validation_data=validation_generator, validation_steps=20, verbose=1)

model.save("CharRecog.h5")

score = model.evaluate(train_generator, verbose=0)
print(score)



img = load_img('unnamed.jpg', color_mode="grayscale", target_size=(32, 32))
x = img_to_array(img)
x=x.reshape(1, 32, 32, 1)

classes = np.argmax(model.predict(x), axis = 1)
print(classes)


