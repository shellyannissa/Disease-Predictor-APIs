import pandas as pd
import numpy as np
import tensorflow as tf
import cv2
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
train=pd.read_csv('/content/drive/MyDrive/1k_data_train.csv')
#a file containing the labels along with images
for i in range(len(train.values)):
	train['path_gambar'][i]='/content/drive/MyDrive/dest_folder/'+train['path_gambar'][i][32:]
test=pd.read_csv('/content/drive/MyDrive/1k_data_test.csv')
for i in range(len(test.values)):
	test['path_gambar'][i]='/content/drive/MyDrive/dest_folder/'+test['path_gambar'][i][32:]
train=train.sample(frac=1)
test=test.sample(frac=1)
train.index=[x for x in range(train.shape[0])]
test.index=[x for x in range(test.shape[0])]
labels = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Effusion', 'Hernia', 'Infiltration', 'Mass', 'No Finding', 'Nodule', 'Pneumonia', 'Pneumothorax']
def sharpening(img):
  kernel=np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
  image_sharp=cv2.filter2D(src=img,ddepth=-1,kernel=kernel)
  return image_sharp
train_datagen_aug=ImageDataGenerator(rescale=1/255.,preprocessing_function=sharpening)
test_datagen_aug=ImageDataGenerator(rescale=1/255.,preprocessing_function=sharpening)
IMG_SHAPE=(224,224)
BATCH_SIZE=32
train_data_aug=train_datagen_aug.flow_from_dataframe(
    train,
    x_col="path_gambar",
    y_col=labels,
    target_size=IMG_SHAPE,
    classes=labels,
    color_mode="rgb",
    class_mode="raw",
    seed=42,
    batch_size=BATCH_SIZE
)
test_data_aug=test_datagen_aug.flow_from_dataframe(
    test,
    x_col="path_gambar",
    y_col=labels,
    target_size=IMG_SHAPE,
    classes=labels,
    color_mode="rgb",
    class_mode="raw",
    shuffle=False,
    seed=42,
    batch_size=BATCH_SIZE
)


from tensorflow.keras.layers import Input
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
# 1. Create model
input_shape=(224, 224, 3)
img_input = Input(shape=input_shape)


#using transfer learning from DenseNet121

base_model = DenseNet121(include_top=False, input_tensor=img_input, input_shape=input_shape, 
                         pooling="avg", weights='imagenet')
x = base_model.output

predictions = Dense(len(labels), activation="linear", name="output")(x)
model = Model(inputs=img_input, outputs=predictions)

# for layer in model.layers:
#   assert layer.trainable=False
#   layer.trainable=False

# 2. Compile the model
optimizer = Adam(learning_rate=0.0005)
model.compile(optimizer=optimizer, 
              loss=keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics="accuracy")
# '''
# #ignored this fitting because of low accuracy
# earlystopper = tf.keras.callbacks.EarlyStopping(
#     monitor='val_loss', patience=2, verbose=0, mode='min',
#     restore_best_weights=True
# )



#  history = model.fit(train_data_aug,
#                     epochs=10,
#                     steps_per_epoch=len(train_data_aug),
#                     validation_data=test_data_aug,
#                     validation_steps=len(test_data_aug),
#                    callbacks=[earlystopper])


# '''
model.fit(train_data_aug,epochs=15,verbose=2,batch_size=32)
model.evaluate(test_data_aug,verbose=2,batch_size=32)

#fitted with loss: 0.0430 - accuracy: 0.9879 


# '''
# #trying to predict with the model
# import cv2
# import numpy as np

# image_path = "/content/drive/MyDrive/dest_folder/00002796_000.png"
# desired_size = (224, 224)
# image = cv2.imread(image_path)
# image_array = np.array(image)

# #applying preprocessing to the image so as to match the model input criteria
# def sharpening(img):
#     kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
#     image_sharp = cv2.filter2D(src=img, ddepth=-1, kernel=kernel)
#     return image_sharp

# image_processed = sharpening(image_array)
# image_processed = image_processed.astype('float32') / 255.0
# image_processed = np.expand_dims(image_processed, axis=0)
# pred = model.predict(image_processed)

# ind = np.argmax(pred, axis=1)#index of most likely class
# labels[ind[0]]

# '''

model.save('x_ray.h5')
