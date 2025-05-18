import jax
import numpy as np
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
from soma_jax import soma


def beale(x):
    """
    The Beale function. The input `x` is 2-dimensional. The minimum value sits
    in (3, 0.5).

    References
    ----------
    https://optimlib.readthedocs.io/en/latest/test_functions.html#beale-function
    """
    return (1.5 - x[0] + x[0]*x[1])**2 + (2.25 - x[0] + x[0]*x[1]**2)**2 + (2.625 - x[0] + x[0]*x[1]**3)**2

def hessian(fn):
    return jax.jacrev(jax.grad(fn))

def normalize(x, eps=0.):
    return x / (jnp.sqrt(jnp.sum(jnp.square(x))) + eps)

class Trajectory:

    def __init__(self, fn):
        self.fn = fn
        self.x_history = []
        self.f_history = []
        self.grad_history = []
        self.hess_history = []
        self.newton_history = []
        self.steps = 0

    def append(self, x):
        f = self.fn(x)
        g = jax.grad(self.fn)(x)
        H = hessian(self.fn)(x)
        n = jnp.dot(jnp.linalg.inv(H), g)

        self.x_history.append(x)
        self.f_history.append(f)
        self.grad_history.append(g)
        self.hess_history.append(H)
        self.newton_history.append(n)
        self.steps += 1


class Optimizer:

    def __init__(self, fn, solver):
        self.fn = fn
        self.solver = solver
        self.opt_state = self.solver.init(x)

    @property
    def optimize(self):
        @jax.jit
        def update_fn(x):
            g = jax.grad(self.fn)(x)
            updates, self.opt_state = self.solver.update(g, self.opt_state, x)
            return optax.apply_updates(x, updates)
        return update_fn


traj = Trajectory(beale)
optimize = Optimizer(beale, optax.sgd(learning_rate=1e-3)).optimize
#optimize = Optimizer(beale, optax.adagrad(learning_rate=1e-3)).optimize
#optimize = Optimizer(beale, optax.rmsprop(learning_rate=1e-3)).optimize
#optimize = Optimizer(beale, optax.adam(learning_rate=1e-3)).optimize
#optimize = Optimizer(beale, soma(learning_rate=1e-3)).optimize
x = jnp.array([1.5, 2])
for step in range(2000):
    if step % 200 == 0:
        traj.append(x)
    x = optimize(x)

traj.x_history

for g, n in zip(traj.grad_history, traj.newton_history):
    print(jnp.dot(normalize(g), normalize(n)))

x0 = jnp.linspace(-4.5, 4.5, 200)
x1 = jnp.linspace(-4.5, 4.5, 200)
y = beale(jnp.stack(jnp.meshgrid(x0, x1), axis=0))

plt.clf()
plt.rcParams['figure.figsize'] = [6, 6]
plt.contour(x0, x1, y, levels=jnp.logspace(0, 6, 30), cmap=plt.cm.jet)
plt.axis('equal')
plt.plot(3, 0.5, 'k*', markersize=10)
plt.plot([x[0] for x in traj.x_history], [x[1] for x in traj.x_history],
         '.--')
plt.show()
