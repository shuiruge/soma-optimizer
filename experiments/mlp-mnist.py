# Following: https://docs.jax.dev/en/latest/notebooks/Neural_Network_and_Data_Loading.html


# -------- Dataset --------- #

import numpy as np
import jax.numpy as jnp
from jax.tree_util import tree_map
from torch.utils.data import DataLoader, default_collate
from torchvision.datasets import MNIST
from torchvision.transforms import Resize

# We decrease the image size for reducing dimension.
image_size = 8
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
  pic = Resize(size=image_size)(pic)
  return np.ravel(np.array(pic, dtype=jnp.float32))

def one_hot(x, k, dtype=jnp.float32):
  """Create a one-hot encoding of x of size k."""
  return jnp.array(x[:, None] == jnp.arange(k), dtype)

# Define our dataset, using torch datasets
mnist_dataset = MNIST('/tmp/mnist/', download=True, transform=resize_flatten_cast)
# Create pytorch data loader with custom collate function
training_generator = DataLoader(mnist_dataset, batch_size=batch_size, collate_fn=numpy_collate)

# Get the full train dataset (for checking accuracy while training)
train_images = np.array(Resize(size=image_size)(mnist_dataset.train_data)).reshape(len(mnist_dataset.train_data), -1)
train_labels = one_hot(np.array(mnist_dataset.train_labels), n_targets)

# Get full test dataset
mnist_dataset_test = MNIST('/tmp/mnist/', download=True, train=False)
test_images = jnp.array(Resize(size=image_size)(mnist_dataset_test.test_data).numpy().reshape(len(mnist_dataset_test.test_data), -1), dtype=jnp.float32)
test_labels = one_hot(np.array(mnist_dataset_test.test_labels), n_targets)


# -------- Multi-Layer Perceptron --------- #

from jax import grad, jit, vmap
from jax import random

# A helper function to randomly initialize weights and biases
# for a dense neural network layer
def random_layer_params(m, n, key, scale=1e-2):
  w_key, b_key = random.split(key)
  return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))

# Initialize all layers for a fully-connected neural network with sizes "sizes"
def init_network_params(sizes, key):
  keys = random.split(key, len(sizes))
  params = []
  for m, n, k in zip(sizes[:-1], sizes[1:], keys):
    params += random_layer_params(m, n, k)
  return params

layer_sizes = [image_size*image_size, 20, 10]
step_size = 0.01
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

@jit
def update(params, x, y):
  grads = grad(loss)(params, x, y)
  return [p - step_size * g for (p, g) in zip(params, grads)]


# -------- Second-Order Correction --------- #

import jax

def get_param_shapes_and_boundaries(params):
  shapes = []
  boundaries = []
  offset = 0
  for p in params:
    offset += jnp.reshape(p, [-1]).shape[0]
    boundaries.append(offset)
    shapes.append(p.shape)
  return shapes, boundaries

def concat_params(params):
  return jnp.concat([jnp.reshape(p, [-1]) for p in params])

def split_params(concated_params, shapes, boundaries):
  params = []
  i = 0
  for shape, offset in zip(shapes, boundaries):
    #print(i, offset, shape)
    #print(concated_params[i:offset].shape)
    p = jnp.reshape(concated_params[i:offset], shape)
    i = offset
    params.append(p)
  return params

def get_mlp_grad(params, inputs, targets):
  shapes, boundaries = get_param_shapes_and_boundaries(params)

  def f(concated_params):
    params = split_params(concated_params, shapes, boundaries)
    return loss(params, inputs, targets)

  g = jax.grad(f)
  return g(concat_params(params))

def hessian(f):
  return jax.jacrev(jax.grad(f))

def get_mlp_hessian(params, inputs, targets):
  shapes, boundaries = get_param_shapes_and_boundaries(params)

  def f(concated_params):
    params = split_params(concated_params, shapes, boundaries)
    return loss(params, inputs, targets)

  h = hessian(f)
  return h(concat_params(params))


# -------- Visualization --------- #

import matplotlib.pyplot as plt

def normalize(vector, eps=0.):
    return vector / (np.sqrt(np.sum(np.square(vector))) + eps)

step = 0
L1_lst, L2_lst = [], []
for epoch in range(num_epochs):
  for x, y in training_generator:
    y = one_hot(y, n_targets)
    params = update(params, x, y)

    step += 1
    if step % 100 == 0:
      print(f'step = {step}')
      #eval_inputs = x
      #eval_targets = one_hot(y, n_targets)
      eval_inputs = train_images[:5000, :]
      eval_targets = one_hot(train_labels[:5000], n_targets)
      g = get_mlp_grad(params, eval_inputs, eval_targets)
      normalized_g = normalize(g)
      gsign = jnp.sign(g)
      normalized_gsign = normalize(gsign)
      # print(f'(H_inv @ g) @ sign(g) = {jnp.dot(normalized_g2, normalized_gsign)}')
      print(f'g @ sign(g) = {jnp.dot(normalized_g, normalized_gsign)}')

      H = get_mlp_hessian(params, eval_inputs, eval_targets)

      dx = step_size * g
      L1 = jnp.dot(g, dx)
      L2 = 0.5 * jnp.dot(dx, H @ dx)
      print(f'L1 = {L1}, L2 = {L2}')
      L1_lst.append(L1)
      L2_lst.append(L2)

  train_acc = accuracy(params, train_images, train_labels)
  test_acc = accuracy(params, test_images, test_labels)
  print("Training set accuracy {}".format(train_acc))
  print("Test set accuracy {}".format(test_acc))

plt.clf()
plt.plot(L1_lst, label='L1')
plt.plot(L2_lst, label='L2')
plt.legend()
plt.show()

plt.clf()
plt.plot([a/b for a, b in zip(L1_lst, L2_lst)], label='L1/L2')
plt.legend()
plt.show()


# eps = 1e-12
# H_inv = jnp.linalg.inv(H + eps * jnp.eye(H.shape[0]))
# jnp.isnan(H_inv).sum()  # => 0, indicating that H_inv is well-defined.
# g2 = jnp.dot(H_inv, g)
# normalized_g2 = normalize(g2)
# print(f'(H_inv @ g) @ g = {jnp.dot(normalized_g2, normalized_g)}')


# jnp.linalg.eigvals(H)

# jnp.linalg.det(H)
# jnp.linalg.det(H + 1e-5 * jnp.eye(H.shape[0]))
# jnp.linalg.det(1e-1 * jnp.eye(H.shape[0]))
