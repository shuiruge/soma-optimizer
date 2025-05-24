"""Implement soma optimizer using optax."""

import optax


def soma(
    learning_rate: optax.ScalarOrSchedule,
    decay_rate: float,
) -> optax.GradientTransformation:
  # transform.trace implements the moving average.
  return optax.chain(
      optax.trace(decay_rate),
      optax.scale_by_sign(),
      optax.scale_by_learning_rate(learning_rate),
  )

