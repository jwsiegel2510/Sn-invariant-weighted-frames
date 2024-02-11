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
  scale = 2.0 / jnp.sqrt(m * k)
  return scale * random.normal(w_key, (n, m, k)), scale * random.normal(b_key, (n,))

def relu(x):
  """The ReLU activation function. """
  return jnp.maximum(0,x)

def init_network_params(widths, key):
  """Randomly initializes all of the parameters for a fully connected network with the given widths.
  
  Args:
    full_widths: a list of the widths of the following fully connected layers.
    key: The random seed

  Returns:
    A list of randomly initialized weights for the network
  """
  keys = random.split(key, len(widths) - 1)
  return [random_layer_params(m,n,key) for m,n,key in zip(widths[:-1], widths[1:], keys)]

def init_CNN_network_params(conv_blocks, full_widths, key):
  """Randomly initializes all of the parameters for a 1d CNN with the given block structure.
  
  Args:
    conv_blocks: a nested list of the channel blocks in each convolutional layer between pooling layers.
    full_widths: a list of the widths of the following fully connected layers (first dimension must match dimension from the convolutional blocks).
    key: The random seed

  Returns:
    A list of randomly initialized weights for the CNN.
  """
  keys = random.split(key, 2)
  conv_keys = random.split(keys[0], len(conv_blocks))
  conv_layers = []
  for i in range(len(conv_blocks)):
    block_keys = random.split(conv_keys[i], len(conv_blocks[i]) - 1)
    channels = conv_blocks[i]
    # Uses convolutional kernels of size 5.
    kernel_size = 5
    conv_layers.append([random_conv_layer_params(m,n,kernel_size,key) for m,n,key in zip(channels[:-1], channels[1:], block_keys)])
  full_keys = random.split(keys[1], len(full_widths) - 1)
  full_layer = [random_layer_params(m,n,key) for m,n,key in zip(full_widths[:-1], full_widths[1:], full_keys)]
  return [conv_layers, full_layer]

def apply_conv_layer(conv_layer, inputs):
  """Applies a convolutional layer to a batch of inputs.

  Args:
    conv_layer: list of parameters for each convolution in the layer
    inputs: tensor containing a batch of inputs

  Returns:
    convolutional layer applied to the inputs
  """
  for conv_weights, biases in conv_layer[:-1]:
      inputs = lax.conv(inputs, conv_weights, [1], 'SAME')
      inputs = relu(inputs +
                    # Interesting hack to get bias to right shape.
                    jnp.moveaxis(jnp.broadcast_to(biases, [inputs.shape[0], inputs.shape[2], inputs.shape[1]]), -1,1))
  conv_weights, biases = conv_layer[-1]
  # Convolution with stride to downsample the image
  inputs = lax.conv(inputs, conv_weights, [2], 'SAME')
  return relu(inputs + jnp.moveaxis(jnp.broadcast_to(biases, [inputs.shape[0], inputs.shape[2], inputs.shape[1]]), -1,1))

def apply_CNN_network(parameters, inputs):
  """Applies the network defined by the input parameters to a batch of inputs. Includes the application of softmax to get probabilities.

  Args:
    parameters: A nested list containing the network parameters
    inputs: tensor containing a batch of inputs

  Returns:
    Output probabilities over the classes as calculated by the network
  """
  [conv_layers, full_layer] = parameters
  for blocks in conv_layers:
    inputs = apply_conv_layer(blocks, inputs)
  inputs = jnp.reshape(inputs, [inputs.shape[0],-1])
  for w,b in full_layer[:-1]:
    inputs = jnp.dot(inputs, w)
    inputs = inputs + b
    inputs = relu(inputs)
  final_w, final_b = full_layer[-1]
  outputs = jnp.dot(inputs, final_w) + final_b
  normalization = jnp.sum(jnp.exp(outputs), -1)
  return jnp.exp(outputs) / normalization

def apply_network(parameters, inputs):
  """Applies a fully connected network to the batch of inputs. Includes the application of softmax to get probabilities.

  Args:
    parameters: A list containing the network parameters.
    inputs: tensor containing the batch of inputs

  Returns:
    Output probabilities of the classes as calculated by the network.
  """
  inputs = jnp.reshape(inputs, (inputs.shape[0], -1))
  for w,b in parameters[:-1]:
    inputs = jnp.dot(inputs, w)
    inputs = inputs + b
    inputs = relu(inputs)
  final_w, final_b = parameters[-1]
  outputs = jnp.dot(inputs, final_w) + final_b
  normalization = jnp.sum(jnp.exp(outputs), -1)
  return jnp.transpose(jnp.transpose(jnp.exp(outputs)) / normalization)

def update(model_params, velocity, grads, lr, mom):
  """Updates the model parameters and velocities based upon the gradients, step size, and momentum

  Args:
    model_params: The model parameters to update
    velocity: The current velocities (required for momentum)
    grads: the gradients of the model parameters
    lr: the learning rate
    mom: the momentum

  Returns:
    updated model parameters and velocities
  """
  if velocity == None:
    velocity = grads
  else:
    velocity = [(mom * w + dw, mom * b + db) for (w,b), (dw,db) in zip(velocity, grads)]
  model_params = [(w - lr * vw, b - lr * vb) for (w,b), (vw,vb) in zip(model_params, velocity)]
  return model_params, velocity

def cross_entropy(model_params, clouds, labels):
  """Calculates the cross entropy loss 
  Args:
    model_params: model parameters
    clouds: point cloud inputs
    labels: training labels

  Output:
    Cross entropy loss.
  """
  probs = apply_network(model_params, clouds)
  return -1.0*jnp.sum(jnp.log(jnp.sum(jnp.multiply(probs, labels), -1))) / clouds.shape[0]

def sort_cloud_along_direction(point_cloud, direction):
  """Sorts a given point cloud in the specified direction.

  Args:
    point_cloud: the clouds of points to sort
    direction: the direction along which to sort

  Returns:
    The sorted point cloud.
  """
  inner_prods = jnp.sum(point_cloud * direction, -1)
  indices = jnp.argsort(inner_prods)
  for i in range(point_cloud.shape[0]):
    point_cloud = point_cloud.at[i,:,:].set(point_cloud[i,indices[i,:],:])
  return point_cloud

def sort_cloud_along_random_direction(point_cloud, key):
  """Sorts a given collection of point clouds along random 2d directions.

  Args:
    point_cloud: the collection of point clouds
    key: Random seed

  Returns:
    The sorted collection of point clouds
  """
  directions = random.normal(key, (point_cloud.shape[0], 1, 2))
  inner_prods = jnp.sum(point_cloud * directions, -1)
  indices = jnp.argsort(inner_prods)
  for i in range(point_cloud.shape[0]):
    point_cloud = point_cloud.at[i,:,:].set(point_cloud[i,indices[i,:],:])
  return point_cloud

def train(model_params, train_clouds, train_labels, invariance, key, batch_size = 60, lr = 0.01, mom = 0.9, steps = 5000):
  """Trains the network with invariance imposed in a variety of ways

  Args:
    model_params: Initial model parameters
    train_clouds: Training point cloud data
    train_labels: Training labels
    invariance: Parameter controlling how invariance is imposed: 
      'None': no invariance 
      'canonical': canonicalize by sorting along the x-axis
      'randomized': create invariance while preserving continuity by sorting along a random direction.
    key: Random seed
    batch size: Batch size
    lr: learning rate
    mom: momentum parameter
    steps: number of training steps

  Output:
    Trained model parameters
  """
  keys = random.split(key, steps)
  velocity = None
  for i in range(steps):
    choice_key, direction_key = random.split(keys[i])
    inds = random.choice(choice_key, train_clouds.shape[0], [batch_size])
    clouds = train_clouds[inds,:,:]
    labels = train_labels[inds,:]
    if invariance == 'randomized':
      clouds = sort_cloud_along_random_direction(clouds, key)
    if invariance == 'canonical':
      clouds = sort_cloud_along_direction(clouds, jnp.array([1,0]))
    if i%5 == 0:
      print('Step : ', i, cross_entropy(model_params, clouds, labels))
    grads = grad(cross_entropy)(model_params, clouds, labels) 
    model_params, velocity = update(model_params, velocity, grads, lr, mom)
  return model_params

def test(model_params, test_clouds, test_labels, invariance):
  """Tests the trained model on the test dataset with invariance imposed in different ways.

  Args:
    model_params: Trained model parameters
    test_clouds: point clouds to test on
    test_labels: test labels
    invariance: Parameter controlling how invariance is imposed:
      'None': no invariance
      'canonical': canonicalize by sorting along the x-axis

  Returns:
    The accuracy of the model on the test dataset.
  """
  if invariance == 'canonical':
    test_clouds = sort_cloud_along_direction(test_clouds, jnp.array([1,0]))
  predictions = jnp.argmax(apply_network(model_params, test_clouds), -1)
  correct = 0.0
  for i in range(predictions.shape[0]):
    correct = correct + test_labels[i][predictions[i]]
  return correct / predictions.shape[0]

def test_with_randomized_invariance(model_params, test_clouds, test_labels, num_average, key):
  """Test the trained model with invariance imposed through averaging over random directions.

  Args:
    model_params: Trained model parameters
    test_clouds: point clouds to test on
    test_labels: test labels
    num_average: the number of directions to average over when making the prediction
    key: Random seed

  Returns:
    The accuracy of the model on the test dataset.
  """
  predictions = jnp.zeros(test_labels.shape)
  keys = random.split(key, num_average)
  for i in range(num_average):
    test_clouds = sort_cloud_along_random_direction(test_clouds, keys[i])
    predictions += apply_network(model_params, test_clouds) 
  predictions = jnp.argmax(predictions, -1)
  correct = 0.0
  for i in range(predictions.shape[0]):
    correct = correct + test_labels[i][predictions[i]]
  return correct / predictions.shape[0]



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

  initialize_key, train_key, test_key = random.split(random.PRNGKey(0), 3)

  print('No Invariance:')

  # Initialize the CNN for classifying point clouds.
  model_params = init_network_params([200, 150, 100, 50, 10], initialize_key)

  # Train the fully connected network without any invariance.
  model_params = train(model_params, train_clouds, train_labels, 'None', train_key)

  # Test the trained network without any invariance on the test dataset.
  print('Test Accuracy: ', test(model_params, test_clouds, test_labels, 'None'))

  print('Invariance via canonicalization:')

  # Initialize the CNN for classifying point clouds.
  model_params = init_network_params([200, 150, 100, 50, 10], initialize_key)

  # Train the fully connected network by canonicalizing via sorting along the x-axis.
  model_params = train(model_params, train_clouds, train_labels, 'canonical', train_key)

  # Test the trained network with canonicalization via sorting along the x-axis..
  print('Test Accuracy: ', test(model_params, test_clouds, test_labels, 'canonical'))

  print('Invariance via random_averaging:')

  # Initialize the CNN for classifying point clouds.
  model_params = init_network_params([200, 150, 100, 50, 10], initialize_key)

  # Train the fully connected network by canonicalizing via sorting along the x-axis.
  model_params = train(model_params, train_clouds, train_labels, 'randomized', train_key)

  # Test the trained network with canonicalization via sorting along the x-axis..
  print('Test Accuracy: ', test_with_randomized_invariance(model_params, test_clouds, test_labels, 5, test_key))
if __name__ == '__main__':
  main()
