from jax import random
import jax.numpy as jnp
import matplotlib.pyplot as plt

def normalize(x, eps=0.):
    return x / (jnp.sqrt(jnp.sum(jnp.square(x))) + eps)

dim = 1000000
#dim = 100
random_key = 42
experiments = 15

keys = random.split(random.key(random_key), experiments)
vectors = list(jnp.ones(dim)] + [random.uniform(k, dim) for k in keys)
directions = list(normalize(v) for v in vectors)

inner_product_mat = jnp.ones([len(directions), len(directions)])
for i in range(len(directions)):
    for j in range(i+1, len(directions)):
        inner_product = jnp.dot(directions[i], directions[j])
        inner_product_mat = inner_product_mat.at[i, j].set(inner_product)
        inner_product_mat = inner_product_mat.at[j, i].set(inner_product)

plt.matshow(inner_product_mat)
plt.colorbar()
plt.show()

