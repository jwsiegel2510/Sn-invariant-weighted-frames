# Author: Jonathan Siegel
#
# Tests different approaches to invariant deep learning on the point cloud MNIST dataset.

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
from jax import vmap
from jax import lax
from jax import grad

def one_hot(x, k, dtype=jnp.float32):
  """Create a one-hot encoding of x of size k."""
  return jnp.array(x[:, None] == jnp.arange(k), dtype)

def random_layer_params(m, n, key):
  """A function which randomly initializes weights and biases.

      Args:
        m: input dimension
        n: output dimension
        key: The random seed

      Returns:
        Randomly initialized weights and biases of the appropriate dimension.
  """
  w_key, b_key = random.split(key)
  scale = 2.0 / jnp.sqrt(m)
  return scale * random.normal(w_key, (m, n)), scale * random.normal(b_key, (n,))

def random_conv_layer_params(m, n, k, key):
  """A function which randomly initializes weights and biases for a 1d convolutional layer.

      Args:
        m: number of input channels
        n: number of output channels
        k: kernel size
        key: The random seed

      Returns:
        Randomly initialized weights and biases for the convolutional layer
  """
  w_key, b_key = random.split(key)
  scale = 2.0 / (k * jnp.sqrt(m))
  return scale * random.normal(w_key, (n, m, k, k)), scale * random.normal(b_key, (n,))

def main():
  # Load point cloud mnist data.
  mnist_data_name = 'point_cloud_mnist_data'
  f = open(mnist_data_name, 'rb')
  mnist_data = pickle.load(f)
  f.close()

  train_data, test_data = mnist_data['train'], mnist_data['test']
  num_labels = 10

  # Full train set
  train_clouds, train_labels = train_data['cloud'], train_data['label']
  train_labels = one_hot(train_labels, num_labels)

  # Full test set
  test_clouds, test_labels = test_data['cloud'], test_data['label']
  test_labels = one_hot(test_labels, num_labels)


if __name__ == '__main__':
  main()
