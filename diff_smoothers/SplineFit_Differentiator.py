from jax import vmap, lax
import jax.numpy as jnp
import jax.random as jr
from typing import Tuple
import chex
import time

from bsm.utils.normalization import Data
from diff_smoothers.Base_Differentiator import BaseDifferentiator, DifferentiatorState

def scale_vector(x: chex.Array,
                 x_min: chex.Array,
                 x_max: chex.Array) -> chex.Array:
    """Scale the vector to [-1, 1]
    t: (m, 1) - m different samples of data
    t_min: The minimum time point
    t_max: The maximum time point
    """
    return 2 * (x - x_min) / (x_max - x_min) - 1

@chex.dataclass
class PolFitState:
    pol_coeff: chex.Array       # The polynomial coefficients
    t_min: chex.Array           # The minimum time point used for fitting
    t_max: chex.Array           # The maximum time point used for fitting
    x_min: chex.Array           # The minimum state value used for fitting
    x_max: chex.Array           # The maximum state value used for fitting

def fitSingleSpline(t: chex.Array,
                    x: chex.Array,
                    degree: int,
                    lambda_: float) -> PolFitState:
    """Fit a spline to the data with regularisation
    t: (m, 1) - m different samples of data
    x: (m, 1)
    degree: Degree of the polynomial
    lambda_: Regularization parameter
    """

    m = degree + 1
    t_scaled = scale_vector(t, t.min(), t.max())
    x_scaled = scale_vector(x, x.min(), x.max())
    
    # Fit the polynomial using the Vandermonde matrix
    A = jnp.vander(t_scaled.flatten(), m)
    D = jnp.eye(m)
    pol_coeff = jnp.linalg.inv(A.T @ A + lambda_ * D) @ (A.T @ x_scaled)

    fit_state = PolFitState(pol_coeff=pol_coeff.flatten(),
                            t_min=t.min(),
                            t_max=t.max(),
                            x_min=x.min(),
                            x_max=x.max())
    return fit_state

def MultipleSplines(t: chex.Array,
                    x: chex.Array,
                    degree: int,
                    lambda_: float,
                    num_splines: int,
                    ) -> PolFitState:
    slice_width = 2 * (t.shape[0] // (num_splines + 1))

    def fit_spline(indx, t, x, degree, lambda_):
        # Compute the dynamic slice start position
        start = indx * (t.shape[0] // (num_splines + 1))
        
        # Dynamically slice the arrays
        t_part = lax.dynamic_slice(t, (start, 0), (slice_width, t.shape[1]))
        x_part = lax.dynamic_slice(x, (start, 0), (slice_width, x.shape[1]))
        
        return fitSingleSpline(t_part, x_part, degree, lambda_)

    # Map over indices from 0 to num_splines
    spline_fits = vmap(fit_spline, in_axes=(0, None, None, None, None))(
        jnp.arange(num_splines), t, x, degree, lambda_
    )

    return spline_fits

def fitMultipleSplines(t: chex.Array,
                       x: chex.Array,
                       degree: int,
                       lambda_: float,
                       num_splines: int,
                       ) -> Tuple[chex.Array, chex.Array, chex.Array]:
    """Fit multiple splines to the data with regularisation.
    Each spline is fitted to a different part of the data.
    The splines overlap and have a width of 2*(m/num_splines).
    t: (m, 1) - m different samples of data
    x: (m, 1)
    degree: Degree of the polynomial
    lambda_: Regularization parameter
    """
    def fit_spline(indx, t, x, degree, lambda_):
        start = indx*(t.shape[0] // (num_splines + 1))
        end = start + 2*(t.shape[0] // (num_splines + 1))
        t_part = t[start:end]
        x_part = x[start:end]
        return fitSingleSpline(t_part, x_part, degree, lambda_)

    # Map over precomputed start and end indices
    spline_fits = []
    x_mins = []
    x_maxs = []

    for i in range(num_splines):
        spline_state = fit_spline(i, t, x, degree, lambda_)
        spline_fits.append(spline_state.pol_coeff)
        x_mins.append(spline_state.x_min)
        x_maxs.append(spline_state.x_max)
    spline_fits = jnp.array(spline_fits)
    x_mins = jnp.array(x_mins)
    x_maxs = jnp.array(x_maxs)
    return spline_fits, x_mins, x_maxs

if __name__ == '__main__':
    key = jr.PRNGKey(0)

    def f(x):
        return (jnp.sin(2 * jnp.pi * x / 2) + 0.5 * jnp.sin(6 * jnp.pi * x / 2) +
                0.25 * jnp.cos(4 * jnp.pi * x) + 0.1 * x) + 2.5
    def f_dot(x):
        return (jnp.pi * jnp.cos(2 * jnp.pi * x / 2) + 1.5 * jnp.pi * jnp.cos(6 * jnp.pi * x / 2) -
                1 * jnp.pi * jnp.sin(4 * jnp.pi * x) + 0.1)
    
    noise_level = 0.1
    d_l, d_u = 0, 4
    num_samples = 400
    t = jnp.linspace(d_l, d_u, num_samples).reshape(-1, 1)
    x = f(t)
    x_dot = f_dot(t)
    x = x + noise_level*jr.normal(key=key, shape=x.shape)
    data = Data(inputs=t, outputs=x)

    degree = 20
    lambda_ = 0.0001
    num_splines = 40
    # Time both implementations
    start = time.time()
    spline_state = MultipleSplines(t, x, degree, lambda_, num_splines)
    end = time.time()
    print(f'MultipleSplines took {end - start} seconds')
    start = time.time()
    spline_state, x_min, x_max = fitMultipleSplines(t, x, degree, lambda_, num_splines)
    end = time.time()
    print(f'fitMultipleSplines took {end - start} seconds')

    # Plot the fit
    t_min = data.inputs.min()
    t_max = data.inputs.max()
    if jnp.isnan(spline_state).any():
        raise ValueError('NaNs in the polynomial coefficients.\
                          This might be because of a too high degree or too low lambda.')
    
    import matplotlib.pyplot as plt
    def plot_single(i, t, x, pol_coeff, x_max, x_min):
        t_scaled = jnp.arange(0, 1, 0.1)
        x_fit_scaled = jnp.polyval(pol_coeff, t_scaled)
        # Unscale the t and x
        t_min = t[i * (t.shape[0] // (num_splines + 1))]
        t_max = t[i * (t.shape[0] // (num_splines + 1)) + 2*(t.shape[0] // (num_splines + 1))]
        t_pred = t_scaled * (t_max - t_min) + t_min
        x_fit = (x_fit_scaled + 1) / 2 * (x_max - x_min)\
                + x_min
        plt.plot(t_pred, x_fit, label=f'Spline {i}')

    for i in range(num_splines):
        plot_single(i, t, x, pol_coeff[i], x_min[i], x_max[i])
    plt.plot(t, x, 'o', label='Data')
    plt.legend()
    plt.show()