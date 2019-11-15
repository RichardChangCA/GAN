from __future__ import absolute_import, division, print_function, unicode_literals

# try:
#   # %tensorflow_version only exists in Colab.
#   %tensorflow_version 2.x
# except Exception:
#   pass


# error: Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).


import tensorflow as tf

import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time
from IPython import display
import datetime
from numpy.random import randint

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("physical_devices-------------", len(physical_devices))
tf.config.experimental.set_memory_growth(physical_devices[0], True)


(train_images, train_labels), (_, _) = tf.keras.datasets.cifar10.load_data()


train_images = train_images.reshape(train_images.shape[0], 32, 32, 3).astype('float32')

print(train_images.max())
print(train_images.min())

train_images = (train_images - 127.5) / 127.5 # Normalize the images to [-1, 1],0-255
# train_images = train_images/255.

print("train_datasize",train_images.size)

BUFFER_SIZE = 60000
BATCH_SIZE = 256

# Batch and shuffle the data
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
# train_dataset = train_images

latent_size = 100
hidden_layer_num = 0
# set it for free, random set


EPOCHS = 1000
num_examples_to_generate = 16    
# generator_loss_metrics.reset_states()
# discriminator_loss_metrics.reset_states()

noise_dim = latent_size

def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(4*4*256, input_shape=(latent_size,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((4, 4, 256)))
    assert model.output_shape == (None, 4, 4, 256) # Note: None is the batch size

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same'))
    assert model.output_shape == (None, 8, 8, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    
    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same'))
    assert model.output_shape == (None, 16, 16, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', activation = 'tanh'))
    assert model.output_shape == (None, 32, 32, 3)

    return model

generator = make_generator_model()

noise = tf.random.normal([1, latent_size])
generated_image = generator(noise, training=False)

plt.imshow(generated_image[0, :, :, :]*0.5 + 0.5)
plt.savefig('initial_noise')

def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same',
                                     input_shape=[32, 32, 3]))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    # model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    # model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(256, (3, 3), strides=(2, 2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    # model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(512, (3, 3), strides=(2, 2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    # model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

discriminator = make_discriminator_model()
decision = discriminator(generated_image)

print ("decision",decision.numpy())

# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    print(real_output)
    print(fake_output)
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

# 1e-4
generator_optimizer = tf.keras.optimizers.Adam(lr=0.0001)
discriminator_optimizer = tf.keras.optimizers.Adam(lr=0.0001)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)



# We will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
seed = tf.random.normal([num_examples_to_generate, noise_dim])

generator_loss_metrics = tf.keras.metrics.Mean('generator_loss_metrics', dtype=tf.float32)
discriminator_loss_metrics = tf.keras.metrics.Mean('generator_loss_metrics', dtype=tf.float32)

# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)

      real_output = discriminator(images, training=True)
      fake_output = discriminator(generated_images, training=True)

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    generator_loss_metrics(gen_loss)
    discriminator_loss_metrics(disc_loss)
    
def train(dataset, epochs):
  iteration = 0
  for epoch in range(epochs):
    start = time.time()
    # batch_per_epoch = int(dataset.shape[0] / BATCH_SIZE)
    for image_batch in dataset:
      train_step(image_batch)
    # for _ in range(batch_per_epoch):
    #   x_index = randint(0,dataset.shape[0],BATCH_SIZE)
    #   image_batch = dataset[x_index]
    #   train_step(image_batch)
    # with train_summary_writer.as_default():
    #     tf.summary.scalar('generator_loss_metrics', generator_loss_metrics.result(), step=epoch)
    #     tf.summary.scalar('discriminator_loss_metrics', discriminator_loss_metrics.result(), step=epoch)
      with train_summary_writer.as_default():
        tf.summary.scalar('generator_loss_metrics', generator_loss_metrics.result(), step=iteration)
        tf.summary.scalar('discriminator_loss_metrics', discriminator_loss_metrics.result(), step=iteration)
        iteration += 1
      
      # generator_lossreset_states()_metrics.reset_states()
      # discriminator_loss_metrics.reset_states()


    # Produce images for the GIF as we go
    # display.clear_output(wait=True)
    generate_and_save_images(generator,
                             epoch + 1,
                             seed)

    # Save the model every 15 epochs
    # if (epoch + 1) % 15 == 0:
    #   checkpoint.save(file_prefix = checkpoint_prefix)

    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

    template = 'Epoch {}, generator_loss: {}, discriminator_loss: {}'
    print (template.format(epoch+1,
                         generator_loss_metrics.result(), 
                         discriminator_loss_metrics.result()))

    generator_loss_metrics.reset_states()
    discriminator_loss_metrics.reset_states()

  # Generate after the final epoch
  # display.clear_output(wait=True)
  generate_and_save_images(generator,
                           epochs,
                           seed)

def generate_and_save_images(model, epoch, test_input):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
  predictions = model(test_input, training=False)

  fig = plt.figure(figsize=(num_examples_to_generate/2,num_examples_to_generate/2))

  for i in range(predictions.shape[0]):
      plt.subplot(num_examples_to_generate/2, num_examples_to_generate/2, i+1)
      plt.imshow(predictions[i, :, :, :]*0.5 + 0.5)
      # print("predictions[i, :, :, :]",predictions[i, :, :, :])
      plt.axis('off')

  plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
#   plt.show()

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/' + current_time + '/GAN_CIFAR10_l' + str(latent_size) + '_h' + str(hidden_layer_num)
train_summary_writer = tf.summary.create_file_writer(train_log_dir)

# %%time
train(train_dataset, EPOCHS)

# checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

# Display a single image using the epoch number
def display_image(epoch_no):
  return PIL.Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))

display_image(EPOCHS)

anim_file = 'gan.gif'

with imageio.get_writer(anim_file, mode='I') as writer:
  filenames = glob.glob('image*.png')
  filenames = sorted(filenames)
  last = -1
  for i,filename in enumerate(filenames):
    frame = 2*(i**0.5)
    if round(frame) > round(last):
      last = frame
    else:
      continue
    image = imageio.imread(filename)
    writer.append_data(image)
  image = imageio.imread(filename)
  writer.append_data(image)

import IPython
if IPython.version_info > (6,2,0,''):
  display.Image(filename=anim_file)




