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


# -------- Higher-Order Correction --------- #

import jax

def loss_diffs(params, inputs, targets, step_size):
  grads = jax.grad(loss)(params, inputs, targets)
  new_params = [p - step_size * g for p, g in zip(params, grads)]
  first_order = -sum([step_size * jnp.sum(jnp.square(g)) for g in grads])
  total = loss(new_params, inputs, targets) - loss(params, inputs, targets)
  return first_order, total


# -------- Visualization --------- #

import matplotlib.pyplot as plt

step = 0
steps, dL1_lst, dL_lst = [], [], []
for epoch in range(10000):
  params = update(params, X_train, one_hot(y_train, n_targets))
  step += 1
  if step % 100 == 0:
    eval_inputs = X_train
    eval_targets = one_hot(y_train, n_targets)
    dL1, dL = loss_diffs(params, eval_inputs, eval_targets, step_size)
    steps.append(step)
    dL1_lst.append(dL1)
    dL_lst.append(dL)

    train_acc = accuracy(params, X_train, one_hot(y_train, n_targets))
    test_acc = accuracy(params, X_test, one_hot(y_test, n_targets))
    print("Training set accuracy {}".format(train_acc))
    print("Test set accuracy {}".format(test_acc))

plt.clf()
plt.plot(steps, dL1_lst, label='dL1')
plt.plot(steps, dL_lst, label='dL')
plt.legend()
plt.show()

plt.clf()
plt.plot(steps, [a/b for a, b in zip(dL1_lst, dL_lst)], label='dL1/dL')
plt.legend()
plt.show()
