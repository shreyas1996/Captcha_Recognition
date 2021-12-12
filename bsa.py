
# Building a CNN

# Import the libraries
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout

# Initialize the CNN model
cnn = Sequential()

# Step 1: Convolution (add the convolutional layer)
cnn.add(Convolution2D(filters = 32, kernel_size = 5,
                      activation = 'relu',
                      input_shape = (20,20, 1)))

# Step 2: Max pooling (add the pooling layer)
cnn.add(MaxPooling2D(pool_size = (2,2))) #stride 2 not tested

# Adding another convolutional layer
cnn.add(Convolution2D(filters = 64, kernel_size = 5,
                      activation = 'relu'))

# Adding another Pooling layer
cnn.add(MaxPooling2D(pool_size = (2,2)))

# Adding another convolutional layer
#cnn.add(Convolution2D(filters = 64, kernel_size = 3,
#                      activation = 'relu'))

# Adding another Pooling layer
#cnn.add(MaxPooling2D(pool_size = (2,2)))

# Step 3: Flattening (adding the flattening layer)
cnn.add(Flatten())

# Step 4: add an ANN (Fully connected)

# add hidden layer
cnn.add(Dense(units = 512, activation = 'relu'))
# add Dropout layer
#cnn.add(Dropout(0.2))
# add hidden layer
#cnn.add(Dense(units = 128, activation = 'relu'))
# add Dropout layer
#cnn.add(Dropout(0.3))
# add output layer
cnn.add(Dense(units = 32, activation = 'softmax'))

# compile the CNN model
cnn.compile(optimizer = 'adam', loss = 'categorical_crossentropy',
            metrics = ['accuracy'])

# Pre process and augment the images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(horizontal_flip = True,
                                   rescale = 1./255,
                                   zoom_range = 0.2,
                                   shear_range = 0.2)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory(directory = 'extracted_letter_images',
                                                 target_size = (20, 20),
                                                 class_mode = 'categorical',
                                                 batch_size = 32)

test_set = test_datagen.flow_from_directory(directory = 'extracted_letters_test',
                                                target_size = (20, 20),
                                                class_mode = 'categorical',
                                                batch_size = 32)

import time
t0 = time.time()

cnn.fit_generator(training_set,
                  epochs = 10,
                  validation_data = test_set,
                  steps_per_epoch = 1,
                  validation_steps = 1)

t1 = time.time()
print("Training completed in " + str(t1-t0) + "seconds" )

cnn.save("captcha_breaker.hdf5")

'''import graphviz
from keras.utils import plot_model
plot_model(cnn, to_file='model.png')'''
