import jax
import numpy as np
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
from soma_jax import soma


class Beale:
    """
    The Beale function. The input `x` is 2-dimensional, in range [-4.5, 4.5]^2.
    The minimum value sits in (3, 0.5).

    References
    ----------
    https://optimlib.readthedocs.io/en/latest/test_functions.html#beale-function
    """

    def __call__(self, x):
        return (1.5 - x[0] + x[0]*x[1])**2 + (2.25 - x[0] + x[0]*x[1]**2)**2 + (2.625 - x[0] + x[0]*x[1]**3)**2

    def plot(self, trajectory):
        x0 = jnp.linspace(-4.5, 4.5, 200)
        x1 = jnp.linspace(-4.5, 4.5, 200)
        y = self(jnp.stack(jnp.meshgrid(x0, x1), axis=0))

        plt.clf()
        plt.rcParams['figure.figsize'] = [6, 6]
        plt.contour(x0, x1, y, levels=jnp.logspace(0, 6, 30), cmap=plt.cm.jet)
        plt.axis('equal')
        plt.plot(3, 0.5, 'k*', markersize=10)
        plt.plot([x[0] for x in trajectory.x_history],
                 [x[1] for x in trajectory.x_history],
                '.--')
        plt.show()


class Booth:
    """
    The Booth function. The input `x` is 2-dimensional, in range [-10, 10]^2.
    The minimum value sits in (1, 3).

    References
    ----------
    https://optimlib.readthedocs.io/en/latest/test_functions.html#booth-function
    """

    def __call__(self, x):
        return (x[0] + 2*x[1] - 7)**2 + (2*x[0] + x[1] - 5)**2

    def plot(self, trajectory):
        x0 = jnp.linspace(-10, 10, 200)
        x1 = jnp.linspace(-10, 10, 200)
        y = self(jnp.stack(jnp.meshgrid(x0, x1), axis=0))

        plt.clf()
        plt.rcParams['figure.figsize'] = [6, 6]
        plt.contour(x0, x1, y, levels=jnp.logspace(0, 6, 30), cmap=plt.cm.jet)
        plt.axis('equal')
        plt.plot(1., 3., 'k*', markersize=10)
        plt.plot([x[0] for x in trajectory.x_history],
                 [x[1] for x in trajectory.x_history],
                '.--')
        plt.show()


class Rosenbrock:
    """
    The Rosenbrock function. The input `x` is 2-dimensional. The minimum value
    sits in (1, 1).

    References
    ----------
    https://optimlib.readthedocs.io/en/latest/test_functions.html#rosenbrock-function
    """

    def __call__(self, x):
        return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2

    def plot(self, trajectory):
        x0 = jnp.linspace(-3, 3, 200)
        x1 = jnp.linspace(-3, 3, 200)
        y = self(jnp.stack(jnp.meshgrid(x0, x1), axis=0))

        plt.clf()
        plt.rcParams['figure.figsize'] = [6, 6]
        plt.contour(x0, x1, y, levels=jnp.logspace(0, 6, 30), cmap=plt.cm.jet)
        plt.axis('equal')
        plt.plot(1., 1., 'k*', markersize=10)
        plt.plot([x[0] for x in trajectory.x_history],
                 [x[1] for x in trajectory.x_history],
                '.--')
        plt.show()


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

    def __init__(self, solver):
        self.solver = solver
        self.opt_state = None

    def optimize(self, fn):
        @jax.jit
        def update_fn(x):
            if self.opt_state is None:
                self.opt_state = self.solver.init(x)
            g = jax.grad(fn)(x)
            updates, self.opt_state = self.solver.update(g, self.opt_state, x)
            return optax.apply_updates(x, updates)
        return update_fn

#optimizer = Optimizer(optax.sgd(1e-3))
optimizer = Optimizer(optax.adam(1e-3))
#optimizer = Optimizer(soma(1e-3))

test_fn = Beale()
#test_fn = Booth()
#test_fn = Rosenbrock()
traj = Trajectory(test_fn)
optimize = optimizer.optimize(test_fn)
x = jnp.array([-2, -0.5])
for step in range(10):
    if step % 1 == 0:
        traj.append(x)
    x = optimize(x)

for g, n in zip(traj.grad_history, traj.newton_history):
    print()

test_fn.plot(traj)

# Examine the relation between linear approximation and the departure between
# gradient and Newton's directions.
for x, next_x in zip(traj.x_history[:-1], traj.x_history[1:]):
    dx = next_x - x
    df = test_fn(next_x) - test_fn(x)
    linear_df = jnp.dot(jax.grad(test_fn)(x), dx)
    print(f'df/linear_df = {df/linear_df:.3f}, '
          f'theta(g, n) = {jnp.dot(normalize(g), normalize(n)):.3f}')
# => Conclusion: no direct relationship.
