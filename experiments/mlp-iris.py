# Following: https://docs.jax.dev/en/latest/notebooks/Neural_Network_and_Data_Loading.html


# -------- Dataset --------- #

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

iris = load_iris()
X = iris['data']
y = iris['target']
names = iris['target_names']
feature_names = iris['feature_names']
n_targets = 3

# Scale data to have mean 0 and variance 1 
# which is importance for convergence of the neural network
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data set into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=2)


# -------- Multi-Layer Perceptron --------- #

from jax import grad, jit, vmap
from jax import random
import jax.numpy as jnp

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

layer_sizes = [X_train.shape[1], 256, n_targets]
step_size = 0.01
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

def one_hot(x, k, dtype=jnp.float32):
  """Create a one-hot encoding of x of size k."""
  return jnp.array(x[:, None] == jnp.arange(k), dtype)


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
    return vector / (jnp.sqrt(jnp.sum(jnp.square(vector))) + eps)

num_epochs = 5000
step = 0
L1_lst, L2_lst = [], []
for epoch in range(num_epochs):
  x = X_train
  y = one_hot(y_train, n_targets)
  params = update(params, x, y)

  step += 1
  if step % 200 == 0:
    print(f'step = {step}')
    g = get_mlp_grad(params, x, y)
    normalized_g = normalize(g)
    gsign = jnp.sign(g)
    normalized_gsign = normalize(gsign)
    # print(f'(H_inv @ g) @ sign(g) = {jnp.dot(normalized_g2, normalized_gsign)}')
    print(f'g @ sign(g) = {jnp.dot(normalized_g, normalized_gsign)}')

    H = get_mlp_hessian(params, x, y)

    dx = step_size * g
    L1 = jnp.dot(g, dx)
    L2 = 0.5 * jnp.dot(dx, H @ dx)
    print(f'L1 = {L1}, L2 = {L2}')
    L1_lst.append(L1)
    L2_lst.append(L2)

    train_acc = accuracy(params, X_train, one_hot(y_train, n_targets))
    test_acc = accuracy(params, X_test, one_hot(y_test, n_targets))
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
