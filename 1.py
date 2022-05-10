import tensorflow as tf
#from tensorflow import keras
from tensorflow.keras import layers
#from keras.utils.vis_utils import plot_model
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)
print(tf.test.is_gpu_available())

#tf.enable_eager_execution()

dataset,info=tfds.load(name='oxford_iiit_pet:3.*.*',download=False,with_info=True)

def resize(input_image, input_mask):
   input_image = tf.image.resize(input_image, (128, 128), method="nearest")
   input_mask = tf.image.resize(input_mask, (128, 128), method="nearest")
   return input_image, input_mask

def augment(input_image, input_mask):
   if tf.random.uniform(()) > 0.5:
       # Random flipping of the image and mask
       input_image = tf.image.flip_left_right(input_image)
       input_mask = tf.image.flip_left_right(input_mask)

   return input_image, input_mask

def normalize(input_image, input_mask):
   input_image = tf.cast(input_image, tf.float32) / 255.0
   input_mask -= 1
   return input_image, input_mask

def load_image_train(datapoint):
   input_image = datapoint["image"]
   input_mask = datapoint["segmentation_mask"]
   input_image, input_mask = resize(input_image, input_mask)
   input_image, input_mask = augment(input_image, input_mask)
   input_image, input_mask = normalize(input_image, input_mask)

   return input_image, input_mask

def load_image_test(datapoint):
   input_image = datapoint["image"]
   input_mask = datapoint["segmentation_mask"]
   input_image, input_mask = resize(input_image, input_mask)
   input_image, input_mask = normalize(input_image, input_mask)

   return input_image, input_mask

train_dataset = dataset["train"].map(load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
print(train_dataset)
test_dataset = dataset["test"].map(load_image_test, num_parallel_calls=tf.data.experimental.AUTOTUNE)
print(test_dataset)

BATCH_SIZE = 64
BUFFER_SIZE = 1000

train_batches = train_dataset.take(3000).cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
train_batches = train_batches.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
validation_batches = test_dataset.take(3000).batch(BATCH_SIZE)
test_batches = test_dataset.skip(3000).take(669).batch(BATCH_SIZE)


def display(display_list):
 plt.figure(figsize=(15, 15))
 title = ["Input Image", "True Mask", "Predicted Mask"]

 for i in range(len(display_list)):
   plt.subplot(1, len(display_list), i+1)
   plt.title(title[i])
   plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
   plt.axis("off")
 plt.show()


sample_batch = next(iter(train_batches))
random_index = np.random.choice(sample_batch[0].shape[0])
sample_image, sample_mask = sample_batch[0][random_index], sample_batch[1][random_index]
display([sample_image,sample_mask])

def downsample(x,n_filters):
    c1 = layers.Conv2D(n_filters, 3, padding="same", activation="relu", kernel_initializer="he_normal")(x)
    c2 = layers.Conv2D(n_filters, 3, padding="same", activation="relu", kernel_initializer="he_normal")(c1)
    mp=layers.MaxPool2D(2)(c2)
    d=layers.Dropout(0.3)(mp)
    return c2,d

def upsample(x,conv_features,n_filters):
    ct=layers.Conv2DTranspose(n_filters,3,2,padding="same")(x)
    ct=layers.concatenate([ct,conv_features])
    d=layers.Dropout(0.3)(ct)
    c1 = layers.Conv2D(n_filters, 3, padding="same", activation="relu", kernel_initializer="he_normal")(d)
    c2 = layers.Conv2D(n_filters, 3, padding="same", activation="relu", kernel_initializer="he_normal")(c1)
    return c2
#encoding
inputs=layers.Input(shape=(128,128,3))
c1,d1=downsample(inputs,64)
c2,d2=downsample(d1,128)
c3,d3=downsample(d2,256)
c4,d4=downsample(d3,512)

connect = layers.Conv2D(1024, 3, padding="same", activation="relu", kernel_initializer="he_normal")(d4)
connect = layers.Conv2D(1024, 3, padding="same", activation="relu", kernel_initializer="he_normal")(connect)
#decoding
u1=upsample(connect,c4,512)
u2=upsample(u1,c3,256)
u3=upsample(u2,c2,128)
u4=upsample(u3,c1,64)

outputs = layers.Conv2D(3, 1, padding="same", activation = "softmax")(u4)

# unet model with Keras Functional API
unet_model = tf.keras.Model(inputs, outputs, name="U-Net")
#
unet_model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
#
NUM_EPOCHS = 20

TRAIN_LENGTH = info.splits["train"].num_examples
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE
#
VAL_SUBSPLITS = 5
TEST_LENTH = info.splits["test"].num_examples
VALIDATION_STEPS = TEST_LENTH // BATCH_SIZE // VAL_SUBSPLITS
#
model_history = unet_model.fit(train_batches,
                              epochs=NUM_EPOCHS,
                              steps_per_epoch=STEPS_PER_EPOCH,
                              validation_steps=VALIDATION_STEPS,
                              validation_data=test_batches)

loss=model_history.history['loss']
val_loss=model_history.history['val_loss']

unet_model.save('UNet4Pets.h5')
#
# def create_mask(pred_mask):
#  pred_mask = tf.argmax(pred_mask, axis=-1)
#  pred_mask = pred_mask[..., tf.newaxis]
#  return pred_mask[0]
#
# def show_predictions(dataset=None, num=1):
#  if dataset:
#    for image, mask in dataset.take(num):
#      pred_mask = unet_model.predict(image)
#      display([image[0], mask[0], create_mask(pred_mask)])
#  else:
#    display([sample_image, sample_mask,
#             create_mask(unet_model.predict(sample_image[tf.newaxis, ...]))])
#
# count = 0
# for i in test_batches:
#    count +=1
# print("number of batches:", count)


