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
    weights: chex.Array = None  # The weights used for fitting

def fitSingleSpline(t: chex.Array,
                    x: chex.Array,
                    degree: int,
                    weights: chex.Array = None,
                    lambda_: float = 0.0) -> PolFitState:
    """Fit a spline to the data with regularisation
    t: (m, 1) - m different samples of data
    x: (m, 1)
    degree: Degree of the polynomial
    lambda_: Regularization parameter
    """

    m = degree + 1
    t_scaled = scale_vector(t, t.min(), t.max())
    x_scaled = scale_vector(x, x.min(), x.max())

    # Weight the samples
    if weights is None: weights = jnp.ones_like(t_scaled)
    W = jnp.diag(weights.flatten())
    
    # Fit the polynomial using the Vandermonde matrix
    A = jnp.vander(t_scaled.flatten(), m)
    D = jnp.eye(m)
    pol_coeff = jnp.linalg.inv(A.T @ W @ A + lambda_ * D) @ (A.T @ W @ x_scaled)

    fit_state = PolFitState(pol_coeff=pol_coeff.flatten(),
                            t_min=t.min(),
                            t_max=t.max(),
                            x_min=x.min(),
                            x_max=x.max(),
                            weights=weights)
    return fit_state

class SplineFit_Differentiator(BaseDifferentiator):
    def __init__(self,
                 state_dim: int,
                 degree: int,
                 lambda_: float,
                 num_splines: int,
                 overlapping: float,
                 weighting: float,):
        """Differentiator using multiple splines over subsets of the data.
        Args:
            - state_dim: The number of states
            - degree: The degree of the polynomial used as a spline
            - lambda_: Regularization parameter
            - num_splines: The number of splines to fit
            - overlapping: The amount of overlapping between the splines
            - weighting: Intensity of the weighting of the samples
        """
        super().__init__(state_dim)
        self.degree = degree
        self.lambda_ = lambda_
        self.num_splines = num_splines
        self.overlapping = overlapping
        self.weighting = weighting


    def train(self,
              key: jr.PRNGKey,
              data: Data) -> DifferentiatorState[PolFitState]:
        assert data.inputs.shape[1] == 1
        assert data.outputs.shape[1] == self.state_dim
        assert data.inputs.shape[0] > self.num_splines,\
            'Not enough data points for the splines. Either reduce the number of splines or increase the number of data points.'

        # Calculate slice dimensions, depending on the overlapping
        slice_width = int(((1 + self.overlapping) * data.inputs.shape[0]) // self.num_splines)
        slice_starts = jnp.arange(self.num_splines) * data.inputs.shape[0] / (self.num_splines+self.overlapping)
        slice_starts = jnp.round(slice_starts).astype(int)

        assert slice_width > self.degree + 1, 'Not enough data points for the spline degree.'
        assert slice_width < data.inputs.shape[0], 'The slice width is too large.'
        assert slice_starts[-1] < data.inputs.shape[0], 'The last slice starts after the last data point.'

        # Calculate the weights for the samples
        t_values = jnp.arange(-1, 1, 2 / (slice_width)).reshape(-1, 1)
        weights = jnp.exp(-self.weighting * t_values**2)

        def fit_spline(start, t, x):
            # Dynamically slice the arrays
            t_part = lax.dynamic_slice(t, (start, 0), (slice_width, t.shape[1]))
            x_part = lax.dynamic_slice(x, (start, 0), (slice_width, x.shape[1]))
        
            return fitSingleSpline(t=t_part, x=x_part,
                                   degree=self.degree, lambda_=self.lambda_,
                                   weights=weights)

        # Map over indices from 0 to num_splines
        spline_fits = vmap(fit_spline, in_axes=(0, None, None))(\
            slice_starts, data.inputs, data.outputs)

        return spline_fits
    
    def predict(self,
                state: DifferentiatorState[PolFitState],
                t: chex.Array,) -> Tuple[DifferentiatorState[PolFitState], chex.Array]:
        assert t.shape[1] == 1
        assert state.pol_coeff.shape[0] == self.num_splines, 'Number of splines does not match the number of fitted splines.'
        assert state.pol_coeff.shape[1] == self.degree + 1, 'Degree of the splines does not match the degree of the fitted splines.'

        state, x_stacked = self.predict_without_combining(state, t)
        # Mean the splines
        x_fit = jnp.nanmedian(x_stacked, axis=0)
        return state, x_fit
    
    def predict_without_combining(self,
                                  state: DifferentiatorState[PolFitState],
                                  t: chex.Array) -> Tuple[DifferentiatorState[PolFitState], chex.Array]:
        assert t.shape[1] == 1
        assert state.pol_coeff.shape[0] == self.num_splines, 'Number of splines does not match the number of fitted splines.'
        assert state.pol_coeff.shape[1] == self.degree + 1, 'Degree of the splines does not match the degree of the fitted splines.'

        def eval_single_spline(t, spline_state):
            # Only evaluate the spline for t values in the range [t_min, t_max]
            t_cut = jnp.where((t >= spline_state.t_min) & (t <= spline_state.t_max), t, jnp.nan)
            t_scaled = scale_vector(t_cut, spline_state.t_min, spline_state.t_max)
            x_scaled = jnp.polyval(spline_state.pol_coeff, t_scaled)
            return (x_scaled + 1) / 2 * (spline_state.x_max - spline_state.x_min)\
                    + spline_state.x_min
        
        x_stacked = vmap(eval_single_spline, in_axes=(None, 0))(t, state)
        return state, x_stacked

    def differentiate(self,
                      state: DifferentiatorState[PolFitState],
                      t: chex.Array) -> Tuple[DifferentiatorState[PolFitState], chex.Array]:
        assert t.shape[1] == 1
        assert state.pol_coeff.shape[0] == self.num_splines, 'Number of splines does not match the number of fitted splines.'
        assert state.pol_coeff.shape[1] == self.degree + 1, 'Degree of the splines does not match the degree of the fitted splines.'
        state, x_stacked = self.differentiate_without_combining(state, t)
        # Mean the splines
        if jnp.min(state.weights) == 1.0:
            # If no weighting is used, the mean is the same as the average
            x_dot_fit = jnp.nanmedian(x_stacked, axis=0)
        else:
            # Weight the values of each spline before averaging
            print('Weighting of the splines for combination is not yet implemented.')
            x_dot_fit = jnp.nanmedian(x_stacked, axis=0)
        return state, x_dot_fit
    
    def differentiate_without_combining(self,
                                        state: DifferentiatorState[PolFitState],
                                        t: chex.Array) -> Tuple[DifferentiatorState[PolFitState], chex.Array]:
        assert t.shape[1] == 1
        assert state.pol_coeff.shape[0] == self.num_splines, 'Number of splines does not match the number of fitted splines.'
        assert state.pol_coeff.shape[1] == self.degree + 1, 'Degree of the splines does not match the degree of the fitted splines.'
    
        def diff_single_spline(t, spline_state):
            # Only evaluate the spline for t values in the range [t_min, t_max]
            t_cut = jnp.where((t >= spline_state.t_min) & (t <= spline_state.t_max), t, jnp.nan)
            t_scaled = scale_vector(t_cut, spline_state.t_min, spline_state.t_max)
            der_pol_coeff = jnp.polyder(spline_state.pol_coeff)
            xdot_scaled = jnp.polyval(der_pol_coeff, t_scaled)
            # Unscaled derivative can be calculated with chain rule
            return (xdot_scaled) * (spline_state.x_max - spline_state.x_min)\
                    / (spline_state.t_max - spline_state.t_min)
    
        x_stacked = vmap(diff_single_spline, in_axes=(None, 0))(t, state)
        return state, x_stacked

        
    
    def plot_multiple_splines(self,
                              true_t: chex.Array,
                              pred_x: chex.Array,
                              true_x: chex.Array,
                              meas_x: chex.Array = None,
                              pred_x_dot: chex.Array=None,
                              true_x_dot: chex.Array=None,
                              pred_t: chex.Array=None,
                              state_labels: list[str]=None):
        """Plot the fit for multiple splines.
        Args:
            - true_t: The true time points, shape (n_timesteps, 1), type: chex.Array
            - pred_x: The predicted states, shape (n_spline, n_timesteps, n_states), type: chex.Array
            - true_x: The true states, shape (n_timesteps, n_states), type: chex.Array
            - meas_x: The measured states, shape (n_timesteps, n_states), type: chex.Array, OPTIONAL
            - pred_x_dot: The predicted state derivatives, shape (n_spline, n_timesteps, n_states), type: chex.Array, OPTIONAL
            - true_x_dot: The true state derivatives, shape (n_timesteps, n_states), type: chex.Array, OPTIONAL
            - pred_t: The predicted time points, shape (n_timesteps, 1), type: chex.Array, OPTIONAL
            - state_labels: The labels for the states, type: list[str], OPTIONAL"""
        if pred_t is None:
            pred_t = true_t
        chex.assert_shape(true_t, (None, 1))
        chex.assert_shape(pred_t, (None, 1))
        chex.assert_shape(pred_x, (self.num_splines, None, self.state_dim))
        chex.assert_shape(true_x, (None, self.state_dim))
        if meas_x is not None: chex.assert_shape(meas_x, (None, self.state_dim))
        if pred_x_dot is not None: chex.assert_shape(pred_x_dot, (self.num_splines, None, self.state_dim))
        if true_x_dot is not None: chex.assert_shape(true_x_dot, (None, self.state_dim))

        # Check if the true signals all have the same number of time steps
        assert true_x.shape[0] == true_t.shape[0]
        # Check if the predicted signals all have the same number of time steps
        assert pred_x.shape[1] == pred_t.shape[0]
        
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(self.state_dim, 1, figsize=(16, 9))
        colmap = plt.get_cmap('hsv', 5)
        if self.state_dim == 1:
            ax = [ax]
        for j in range(self.state_dim):
            # Plot the true and measured signals
            ax[j].plot(true_t, true_x[:,j], color=[0.6, 0.6, 0.6],
                       alpha=0.5, linewidth=2, label=r'$x_{TRUE}$')
            if meas_x is not None:
                ax[j].scatter(true_t, meas_x[:,j], s=10, color='black', alpha=0.5,
                              label=r'$x_{MEAS}$', edgecolor='none')
            if true_x_dot is not None:
                ax[j].plot(true_t, true_x_dot[:,j], color='black',
                           alpha=0.5, linewidth=2, label=r'$\dot{x}_{TRUE}$')
                
            # Plot the predicted splines
            for i in range(self.num_splines):
                ax[j].plot(pred_t, pred_x[i,:,j], color=colmap(i%5),
                           linewidth=1.0, alpha=0.9)
                if pred_x_dot is not None:
                    ax[j].plot(pred_t, pred_x_dot[i,:,j], color=colmap(i%5),
                               linestyle=':', linewidth=1.0, alpha=0.9)
            ax[j].grid(True, which='both')
        if state_labels is not None:
            for j in range(self.state_dim):
                ax[j].set_ylabel(state_labels[j])
        ax[-1].set_xlabel(r'Time [s]')
        ax[-1].legend()
        plt.tight_layout()

        return fig, ax
        

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
    x_true = f(t)
    x_dot = f_dot(t)
    x = x_true + noise_level*jr.normal(key=key, shape=x_true.shape)
    data = Data(inputs=t, outputs=x)

    test_t = jnp.linspace(d_l, d_u, num_samples).reshape(-1, 1)
    diff = SplineFit_Differentiator(state_dim=1,
                                    degree=9,
                                    lambda_=0.1,
                                    num_splines=50,
                                    overlapping=3.0,
                                    weighting=0.0)
    state = diff.train(key, data)
    state, x_dot_fit = diff.differentiate(state, test_t)
    state, x_fit = diff.predict(state, test_t)

    state, x_fit_stacked = diff.predict_without_combining(state, test_t)
    state, x_dot_fit_stacked = diff.differentiate_without_combining(state, test_t)
    fig, _ = diff.plot_multiple_splines(true_t=t,
                                        pred_x=x_fit_stacked,
                                        true_x=x_true,
                                        meas_x=x,
                                        pred_x_dot=None,
                                        true_x_dot=None,
                                        pred_t=test_t,
                                        state_labels=['x'])
    fig.savefig('SplineFit_Differentiator_MultipleSplines.pdf')

    fig, _ = diff.plot_fit(true_t=t,
                           pred_x=x_fit,
                           true_x=x,
                           pred_x_dot=x_dot_fit,
                           true_x_dot=x_dot,
                           pred_t=test_t,
                           state_labels=['x'])
    fig.savefig('SplineFit_Differentiator.pdf')