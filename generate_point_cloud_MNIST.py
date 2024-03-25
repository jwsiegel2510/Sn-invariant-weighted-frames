# Author: Jonathan Siegel
#
# Processes the MNIST dataset by converting each image of a digit into a point cloud. 
# The goal is to test permutation invariant methods on the task of classifying the resulting point clouds.

import jax.numpy as jnp
from matplotlib import pyplot
import tensorflow as tf
# Ensure TF does not see GPU and grab all GPU memory.
tf.config.set_visible_devices([], device_type='GPU')
import os
import pickle

import tensorflow_datasets as tfds
from jax._src.api import jit
from jax import random

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
  test_images, test_labels = test_data['image'], test_data['label']
  test_images = jnp.squeeze(jnp.moveaxis(test_images * (1.0/256), -1, 1))


  # Process the MNIST images into point clouds
  key = random.PRNGKey(0)
  train_key, test_key = random.split(key)
  point_count = 100
  mnist_point_cloud_train = jnp.zeros([train_images.shape[0], point_count, 2])
  train_keys = random.split(train_key, train_images.shape[0])
  for i in range(train_images.shape[0]):
    mnist_point_cloud_train = mnist_point_cloud_train.at[i,:,:].set(process_image_to_point_cloud(train_images[i,:,:], point_count, train_keys[i])) 
  mnist_point_cloud_test = jnp.zeros([test_images.shape[0], point_count, 2])
  test_keys = random.split(test_key, test_images.shape[0])
  for i in range(test_images.shape[0]):
    mnist_point_cloud_test = mnist_point_cloud_test.at[i,:,:].set(process_image_to_point_cloud(test_images[i,:,:], point_count, test_keys[i]))
  point_cloud_train_data = {'cloud': mnist_point_cloud_train, 'label': train_labels}
  point_cloud_test_data = {'cloud': mnist_point_cloud_test, 'label': test_labels}
  point_cloud_data = {'train': point_cloud_train_data, 'test': point_cloud_test_data}

  # Save the dataset
  save_name = './point_cloud_mnist_data'
  f = open(save_name, 'wb')
  pickle.dump(point_cloud_data, f)
  f.close()

def process_image_to_point_cloud(image, point_count, key):
  """A function which converts a grayscale image (values in [0,1]) into a point cloud in [0,1]^2.

      Args: 
        image: The input grayscale image
        point_count: The number of points in the point cloud.
        key: The random seed 

      Returns:
        The corresponding point cloud.
  """
  sample_key, choice_key, gaussian_key = random.split(key, 3)
  coins = random.uniform(sample_key, image.shape)
  trunc = (coins < image)
  cloud = jnp.asarray(jnp.nonzero(trunc))
  cloud = jnp.transpose(cloud) / jnp.array([image.shape[0], image.shape[1]])
  index_sample = random.choice(choice_key, cloud.shape[0], [point_count])
  # Jiggle the points a bit
  return cloud[index_sample,:] + 0.2 * random.normal(gaussian_key, (point_count, 2)) / jnp.array([image.shape[0], image.shape[1]])
 
if __name__ == '__main__':
  main()
