# Author: Jonathan Siegel
#
# Plots the first MNIST digit and the corresponding point cloud to help visualize the task.

import jax.numpy as jnp
from matplotlib import pyplot
import tensorflow as tf
# Ensure TF does not see GPU and grab all GPU memory.
tf.config.set_visible_devices([], device_type='GPU')

import tensorflow_datasets as tfds
import os
import pickle

def main():
  # Load the MNIST training and test data.
  data_dir = './mnist_data'
  # Fetch full datasets for evaluation
  # tfds.load returns tf.Tensors (or tf.data.Datasets if batch_size != -1)
  # You can convert them to NumPy arrays (or iterables of NumPy arrays) with tfds.dataset_as_numpy
  mnist_data, info = tfds.load(name="mnist", batch_size=-1, data_dir=data_dir, with_info=True)
  mnist_data = tfds.as_numpy(mnist_data)
  train_data, test_data = mnist_data['train'], mnist_data['test']
  train_images, train_labels = train_data['image'], train_data['label']
  train_images = jnp.squeeze(jnp.moveaxis(train_images * (1.0/256), -1, 1))

  # Plot first MNIST image
  pyplot.imshow(train_images[0])
  pyplot.show()

  # Load the point cloud dataset.
  mnist_data_name = 'point_cloud_mnist_data'
  f = open(mnist_data_name, 'rb')
  mnist_data = pickle.load(f)
  f.close()

  train_data, test_data = mnist_data['train'], mnist_data['test']
  train_clouds, train_labels = train_data['cloud'], train_data['label']

  # Plot the first point cloud image.
  ax = pyplot.gca()
  ax.set_xlim(0,1)
  ax.set_ylim(1,0)
  pyplot.scatter(train_clouds[0,:,1], train_clouds[0,:,0])
  pyplot.show()

if __name__ == '__main__':
  main()
