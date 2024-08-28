from jax import vmap
import jax.numpy as jnp
import jax.random as jr
from typing import Tuple
import chex

from bsm.utils.normalization import Data
from diff_smoothers.Base_Differentiator import BaseDifferentiator, DifferentiatorState

def createIntegrationMatrix(t_source, t_target):
    m = t_target.shape[0]
    n = t_source.shape[0]
    t_target_next = jnp.roll(t_target, -1)
    t_target_next = t_target_next.at[-1].set(t_target[-1])
    def a_ij(t_i, t_j, t_j_next):
        return jnp.where(t_i <= t_j, 0.0,
                         jnp.where(t_i <= t_j_next, t_i - t_j, t_j_next - t_j))
    
    A = vmap(lambda t_i: vmap(lambda j: jnp.squeeze(a_ij(t_i, t_target[j], t_target_next[j])))(jnp.arange(m)))(t_source)
    return A

def fitSingleStateTikhonov(t_source: chex.Array,
                           x_source: chex.Array,
                           t_target: chex.Array,
                           regtype: str,
                           lambda_: float,
                           init_offset: chex.Array) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
    """Fit to data with Tikhonov regularization
    t: (m, 1) - m different samples of data
    x: (m, 1)
    regtype: 'zero', 'first', 'second'
    lambda_: Regularization parameter
    init_offset: true function value at the lower bound
    """
    assert regtype in ['zero', 'first', 'second'], 'Unknown regtype - should be "zero", "first" or "second"'
    x = x_source - init_offset

    # Calculate the derivative by solving the least squares problem
    A = createIntegrationMatrix(t_source, t_target)
    m = t_target.shape[0]

    if regtype == 'zero':
        D = jnp.eye(m)
    elif regtype == 'first':
        D1 = jnp.zeros((m - 1, m))# Set the right hand diagonal to 1
        D1 = D1.at[jnp.arange(m-1), jnp.arange(m-1)].set(-1)
        D1 = D1.at[jnp.arange(m-1), jnp.arange(1, m)].set(1)
        D = jnp.vstack((jnp.eye(m), D1))
    elif regtype == 'second':
        D1 = jnp.zeros((m - 1, m))
        D1 = D1.at[jnp.arange(m-1), jnp.arange(m-1)].set(-1)
        D1 = D1.at[jnp.arange(m-1), jnp.arange(1, m)].set(1)
        D2 = jnp.zeros((m - 2, m))
        D2 = D2.at[jnp.arange(m-2), jnp.arange(m-2)].set(1)
        D2 = D2.at[jnp.arange(m-2), jnp.arange(1, m-1)].set(-2)
        D2 = D2.at[jnp.arange(m-2), jnp.arange(2, m)].set(1)
        D = jnp.vstack((jnp.eye(m), D1, D2))
    else:
        raise ValueError('Unknown regtype')
    
    x_dot_fit = jnp.linalg.solve(A.T @ A + lambda_ * D.T @ D, jnp.dot(A.T, x))

    # Calculate the integral by summing up the derivatives
    dt = jnp.diff(t_target, axis=0).reshape(-1)
    x_fit = jnp.concatenate([jnp.array([0.0]), jnp.cumsum((x_dot_fit[:-1]*dt), axis=0)])
    x_fit += init_offset

    return x_fit, x_dot_fit, A, D

@chex.dataclass
class TikhonovState:
    A: chex.Array
    D: chex.Array


class TikhonovDifferentiator(BaseDifferentiator):
    def __init__(self,
                 state_dim: int,
                 regtype: str,
                 lambda_: float):
        super().__init__(state_dim)
        self.regtype = regtype
        self.lambda_ = lambda_

    def train(self, key: chex.PRNGKey, data: Data) -> DifferentiatorState[TikhonovState]:
        return DifferentiatorState(input_data=data, key=key, algo_state=TikhonovState(A=None, D=None))

    def differentiate(self,
                      state: DifferentiatorState[TikhonovState],
                      t: chex.Array) -> Tuple[DifferentiatorState[TikhonovState], chex.Array]:
        assert t.shape[1] == 1
        init_offset = state.input_data.outputs[0, :]
        v_apply = vmap(fitSingleStateTikhonov,
                       in_axes=(None, 1, None, None, None, 0),
                       out_axes=(1, 1, 2, 2))
        _, x_dot_fit, A, D = v_apply(state.input_data.inputs,
                                     state.input_data.outputs, t,
                                     self.regtype, self.lambda_, init_offset)
        state.algo_state = TikhonovState(A=A, D=D)
        return state, x_dot_fit

    def predict(self,
                state: DifferentiatorState[TikhonovState],
                t: chex.Array) -> Tuple[DifferentiatorState[TikhonovState], chex.Array]:
        assert t.shape[1] == 1
        init_offset = state.input_data.outputs[0, :]
        v_apply = vmap(fitSingleStateTikhonov,
                       in_axes=(None, 1, None, None, None, 0),
                       out_axes=(1, 1, 2, 2))
        x_fit, _, A, D = v_apply(state.input_data.inputs,
                                 state.input_data.outputs, t,
                                 self.regtype, self.lambda_, init_offset)
        state.algo_state = TikhonovState(A=A, D=D)
        return state, x_fit
    

if __name__ == '__main__':
    key = jr.PRNGKey(0)

    def f(x):
        return (jnp.sin(2 * jnp.pi * x / 2) + 0.5 * jnp.sin(6 * jnp.pi * x / 2) +
                0.25 * jnp.cos(4 * jnp.pi * x) + 0.1 * x) + 2.5
    def f_dot(x):
        return (jnp.pi * jnp.cos(2 * jnp.pi * x / 2) + 1.5 * jnp.pi * jnp.cos(6 * jnp.pi * x / 2) -
                1 * jnp.pi * jnp.sin(4 * jnp.pi * x) + 0.1)
    
    noise_level = 0.1
    d_l, d_u = 0, 10
    num_samples = 400
    t = jnp.linspace(d_l, d_u, num_samples).reshape(-1, 1)
    x = f(t)
    x_dot = f_dot(t)
    x = x + noise_level*jr.normal(key=key, shape=x.shape)
    data = Data(inputs=t, outputs=x)

    # Create target timestamps
    test_t = jnp.linspace(d_l, d_u, num_samples).reshape(-1, 1)
    diff = TikhonovDifferentiator(state_dim=1, regtype='second', lambda_=0.002)
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
    fig.savefig('tikhonov_differentiator.png')