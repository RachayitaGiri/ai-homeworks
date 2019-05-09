from keras.models import load_model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np
import sys
import os

model = load_model("vggTransfer_7e-4_epochs5.h5")

# read image path from command line arguments
img_path = sys.argv[1]

# load and preprocess image
img = image.load_img(img_path, target_size=(150, 150))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# predict class = cat if probability < 0.5, else class = dog
preds = model.predict(x)
print(preds)
#threshold = 0.5
#if preds[0][0] > 0.5:
#    print("\nPredicted class: Dog\n")
#else:
#    print("\nPredicted class: Cat\n")
