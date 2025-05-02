import equinox as eqx
import jax
from jax.tree_util import Partial


class M(eqx.Module):
    x: jax.Array

    def f(self, y):
        return self.x + y


@jax.jit
def g(x):
    fx = Partial(M(x).f)  # <--- this line is changed
    return jax.lax.switch(0, [fx, fx], x)


x = jax.numpy.array([1, 2])
out = g(x)

print(out)
