# Following: https://docs.jax.dev/en/latest/notebooks/Neural_Network_and_Data_Loading.html


# -------- Dataset --------- #

import numpy as np
import jax.numpy as jnp
from jax.tree_util import tree_map
from torch.utils.data import DataLoader, default_collate
from torchvision.datasets import MNIST

# We decrease the image size for reducing dimension.
batch_size = 128
n_targets = 10

def numpy_collate(batch):
  """
  Collate function specifies how to combine a list of data samples into a batch.
  default_collate creates pytorch tensors, then tree_map converts them into numpy arrays.
  """
  return tree_map(np.asarray, default_collate(batch))

def resize_flatten_cast(pic):
  """Convert PIL image to flat (1-dimensional) numpy array."""
  return np.ravel(np.array(pic, dtype=jnp.float32))

def one_hot(x, k, dtype=jnp.float32):
  """Create a one-hot encoding of x of size k."""
  return jnp.array(x[:, None] == jnp.arange(k), dtype)

# Define our dataset, using torch datasets
mnist_dataset = MNIST('/tmp/mnist/', download=True, transform=resize_flatten_cast)
# Create pytorch data loader with custom collate function
training_generator = DataLoader(mnist_dataset, batch_size=batch_size, collate_fn=numpy_collate)

# Get the full train dataset (for checking accuracy while training)
train_images = np.array(mnist_dataset.train_data).reshape(len(mnist_dataset.train_data), -1)
train_labels = one_hot(np.array(mnist_dataset.train_labels), n_targets)

# Get full test dataset
mnist_dataset_test = MNIST('/tmp/mnist/', download=True, train=False)
test_images = jnp.array(mnist_dataset_test.test_data.numpy().reshape(len(mnist_dataset_test.test_data), -1), dtype=jnp.float32)
test_labels = one_hot(np.array(mnist_dataset_test.test_labels), n_targets)


# -------- Multi-Layer Perceptron --------- #

from jax import grad, jit, vmap
from jax import random

# A helper function to randomly initialize weights and biases
# for a dense neural network layer
def random_layer_params(m, n, key, scale=1e-2):
  w_key, b_key = random.split(key)
  return scale * random.normal(w_key, (n, m)), 0. * random.normal(b_key, (n,))

# Initialize all layers for a fully-connected neural network with sizes "sizes"
def init_network_params(sizes, key):
  keys = random.split(key, len(sizes))
  params = []
  for m, n, k in zip(sizes[:-1], sizes[1:], keys):
    params += random_layer_params(m, n, k)
  return params

layer_sizes = [28*28, 512, 512, 10]
num_epochs = 5
params = init_network_params(layer_sizes, random.key(0))

from jax.scipy.special import logsumexp

def relu(x):
  return jnp.maximum(0, x)

def predict(params, image):
  grouped_params = [(params[2*i], params[2*i+1]) for i in range(len(params)//2)]

  # per-example predictions
  activations = image
  for w, b in grouped_params[:-1]:
    outputs = jnp.dot(w, activations) + b
    activations = relu(outputs)

  final_w, final_b = grouped_params[-1]
  logits = jnp.dot(final_w, activations) + final_b
  return logits - logsumexp(logits)

# Make a batched version of the `predict` function
batched_predict = vmap(predict, in_axes=(None, 0))

def accuracy(params, images, targets):
  target_class = jnp.argmax(targets, axis=1)
  predicted_class = jnp.argmax(batched_predict(params, images), axis=1)
  return jnp.mean(predicted_class == target_class)

def loss(params, images, targets):
  preds = batched_predict(params, images)
  return -jnp.mean(preds * targets)


# -------- Higher-Order Correction --------- #

import jax

def loss_diffs(params, inputs, targets, step_size):
  grads = jax.grad(loss)(params, inputs, targets)
  new_params = [p - step_size * g for p, g in zip(params, grads)]
  first_order = -sum([step_size * jnp.sum(jnp.square(g)) for g in grads])
  total = loss(new_params, inputs, targets) - loss(params, inputs, targets)
  return first_order, total


# -------- Visualization --------- #

import optax
import matplotlib.pyplot as plt
from soma_jax import soma

solver = optax.adam(learning_rate=1e-3)
# solver = optax.sgd(learning_rate=1e-3)
# solver = optax.sgd(learning_rate=1e-3, momentum=0.95)
# solver = soma(learning_rate=2e-4)  # test ours.

@jit
def update(params, x, y, opt_state):
  grads = grad(loss)(params, x, y)
  updates, opt_state = solver.update(grads, opt_state, params)
  return optax.apply_updates(params, updates), opt_state

step = 0
steps, L_lst, dL1_lst, dL_lst = [], [], [], []
opt_state = solver.init(params)
for epoch in range(num_epochs):
  for x, y in training_generator:
    y = one_hot(y, n_targets)
    params, opt_state = update(params, x, y, opt_state)
    step += 1
    if step % 50 == 0:
      L = loss(params, x, y)
      dL1, dL = loss_diffs(params, x, y, 1e-3)
      steps.append(step)
      L_lst.append(L)
      dL1_lst.append(dL1)
      dL_lst.append(dL)

  train_acc = accuracy(params, train_images, train_labels)
  test_acc = accuracy(params, test_images, test_labels)
  print("Training set accuracy {}".format(train_acc))
  print("Test set accuracy {}".format(test_acc))

plt.clf()
plt.plot(steps, L_lst, 'o--')
plt.xlabel('iterate steps')
plt.ylabel('loss')
plt.legend()
plt.show()

plt.clf()
plt.plot(steps, dL1_lst, 'o--', label='first-order')
plt.plot(steps, dL_lst, 'o--', label='total')
plt.xlabel('iterate steps')
plt.ylabel('loss differences')
plt.legend()
plt.show()

plt.clf()
plt.plot(steps, [a/b for a, b in zip(dL1_lst, dL_lst)],
         'o--', label='first-order / total')
plt.xlabel('iterate steps')
plt.ylabel('loss differences')
plt.legend()
plt.show()
