from jax import vmap
import jax.numpy as jnp
import jax.random as jr
from typing import Tuple
import chex

from bsm.utils.normalization import Data
from diff_smoothers.Base_Differentiator import BaseDifferentiator, DifferentiatorState

def fitSinglePolynomial(t: chex.Array,
                        x: chex.Array,
                        degree: int,
                        lambda_: float) -> chex.Array:
    """Fit a polynomial to the data with regularisation
    t: (m, 1) - m different samples of data
    x: (m, 1)
    degree: Degree of the polynomial
    lambda_: Regularization parameter
    """
    m = degree + 1
    A = jnp.vander(t.flatten(), m)
    D = jnp.eye(m)
    pol_coeff = jnp.linalg.solve(A.T @ A + lambda_ * D.T @ D, jnp.dot(A.T, x))
    return pol_coeff.flatten()

@chex.dataclass
class PolFitState:
    pol_coeff: chex.Array

class PolFit_Differentiator(BaseDifferentiator):
    def __init__(self,
                 state_dim: int,
                 degree: int,
                 lambda_: float):
        super().__init__(state_dim)
        self.degree = degree
        self.lambda_ = lambda_

    def train(self,
              key: jr.PRNGKey,
              data: Data) -> DifferentiatorState[PolFitState]:
        assert data.inputs.shape[1] == 1
        assert data.outputs.shape[1] == self.state_dim
        v_apply = vmap(fitSinglePolynomial, in_axes=(None, 1, None, None),
                       out_axes=(1))
        pol_coeff = v_apply(data.inputs, data.outputs,
                            self.degree, self.lambda_)
        if jnp.isnan(pol_coeff).any():
            raise ValueError('NaNs in the polynomial coefficients.\
                              This might be because of a too high degree or too low lambda.')
        return DifferentiatorState(input_data=data,
                                   key=key,
                                   algo_state=PolFitState(pol_coeff=pol_coeff))

    def differentiate(self,
                      state: DifferentiatorState[PolFitState],
                      t: chex.Array) -> Tuple[DifferentiatorState[PolFitState], chex.Array]:
        assert t.shape[1] == 1
        def diff_single(t, pol_coeff):
            der_pol_coeff = jnp.polyder(pol_coeff)
            return jnp.polyval(der_pol_coeff, t)
        x_dot_fit = vmap(diff_single, in_axes=(None, 1), out_axes=(1))\
                        (t.reshape(-1,), state.algo_state.pol_coeff)
        return state, x_dot_fit

    def predict(self,
                state: DifferentiatorState[PolFitState],
                t: chex.Array) -> Tuple[DifferentiatorState[PolFitState], chex.Array]:
        assert t.shape[1] == 1
        x_fit = vmap(jnp.polyval, in_axes=(1, None), out_axes=(1))\
                    (state.algo_state.pol_coeff, t.reshape(-1,))
        return state, x_fit

if __name__=="__main__":
    key = jr.PRNGKey(0)

    def f(x):
        return (jnp.sin(2 * jnp.pi * x / 2) + 0.5 * jnp.sin(6 * jnp.pi * x / 2) +
                0.25 * jnp.cos(4 * jnp.pi * x) + 0.1 * x) + 2.5
    def f_dot(x):
        return (jnp.pi * jnp.cos(2 * jnp.pi * x / 2) + 1.5 * jnp.pi * jnp.cos(6 * jnp.pi * x / 2) -
                1 * jnp.pi * jnp.sin(4 * jnp.pi * x) + 0.1)
    
    noise_level = 0.1
    d_l, d_u = 0, 10
    num_samples = 200
    t = jnp.linspace(d_l, d_u, num_samples).reshape(-1, 1)
    x = f(t)
    x_dot = f_dot(t)
    x = x + noise_level*jr.normal(key=key, shape=x.shape)
    data = Data(inputs=t, outputs=x)

    # Create target timestamps
    test_t = jnp.linspace(d_l, d_u, num_samples).reshape(-1, 1)
    diff = PolFit_Differentiator(state_dim=1,
                                 degree=15,
                                 lambda_=0.001)
    state = diff.train(key, data)
    state, x_dot_fit = diff.differentiate(state, test_t)
    state, x_fit = diff.predict(state, test_t)
    fig, _ = diff.plot_fit(true_t=t,
                  pred_x=x_fit,
                  true_x=x,
                  pred_x_dot=x_dot_fit,
                  true_x_dot=x_dot,
                  pred_t=test_t,
                  state_labels=['x'])
    fig.savefig('PolFit_Differentiator.png')