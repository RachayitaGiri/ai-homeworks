# This fine-tuning code was taken from Keras documentation and modified slightly to have onlyi
# a single dense neuron that is fine-tuned.

from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, Input
import pprint

# dimensions of our images.
img_width, img_height = 150, 150

train_data_dir = 'cats_and_dogs_medium/train'      # Path to training images
validation_data_dir = 'cats_and_dogs_medium/test'  # Validation and test set are the same here
nb_train_samples = 6000                           # originally 30000
nb_validation_samples = 180                        # originally 900
epochs = 5
batch_size = 16

# Build the VGG16 network
input_tensor = Input(shape=(150,150,3))
base_model = VGG16(weights='imagenet',include_top=False,input_tensor=input_tensor)

# Uncomment this block to print the names and number of layers in the base_model
'''
i = 1
for layer in base_model.layers:
    print(i, layer.name)
    i = i+1
'''

# Add an additional MLP model at the "top" (end) of the network
top_model = Sequential()
top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
top_model.add(Dense(1, activation='sigmoid'))
model = Model(input= base_model.input, output= top_model(base_model.output))

#pprint.pprint(model.layers)

# Uncomment this block to print the names and number of layers in the modified model
'''
i = 1
for layer in model.layers:
    print(i, layer.name)
    i = i + 1
'''
NUM_LAYERS = 19

# Freeze all the layers in the original model (fine-tune only the added Dense layers)
for layer in model.layers[:NUM_LAYERS]:       # You need to figure out how many layers were in the base model to freeze
    layer.trainable = False

# Compile the model with a SGD/momentum optimizer and a slow learning rate.
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=1e-3, momentum=0.9),
              metrics=['accuracy'])

# Prepare data augmentation configuration
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')
model.summary()
# Fine-tune the model
model.fit_generator(
    train_generator,
    samples_per_epoch=nb_train_samples//batch_size,
    epochs=epochs,				            # For Keras 2.0 API change to epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples//batch_size)       # For Keras 2.0 API change to validation_steps=nb_validation_samples

model.save("fine_tuned_model.h5")
