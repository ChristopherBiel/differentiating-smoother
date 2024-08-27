from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Tuple
import chex
import matplotlib.pyplot as plt

from bsm.utils.normalization import Data

AlgorithmState = TypeVar('AlgorithmState')

# Defines the state of a differentiator
@chex.dataclass
class DifferentiatorState:
    input_data: Data                    # The input data
    key: chex.Array                     # The key
    algo_state: AlgorithmState          # The state of the model (e.g. NN weights, parameters, etc.)

# Defines a base class for differentiators
class BaseDifferentiator(ABC, Generic[AlgorithmState]):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    @abstractmethod
    def train(self, key: chex.PRNGKey, data: Data) -> DifferentiatorState[AlgorithmState]:
        raise NotImplementedError
    
    @abstractmethod   
    def differentiate(self,
                      state: DifferentiatorState[AlgorithmState],
                      t: chex.Array) -> Tuple[DifferentiatorState[AlgorithmState], chex.Array]:
        """Estimate the state derivatives at the given time points."""
        raise NotImplementedError
    
    @abstractmethod
    def predict(self,
                state: DifferentiatorState[AlgorithmState],
                t: chex.Array) -> chex.Array:
        """Predict the states at the given time points."""
        raise NotImplementedError

    def plot_fit(self,
                 true_t: chex.Array,
                 pred_x: chex.Array,
                 true_x: chex.Array,
                 pred_x_dot: chex.Array,
                 true_x_dot: chex.Array,
                 pred_t: chex.Array = None,
                 state_labels: list[str] = None):
        """Plot the fit of the model.
        Args:
            - true_t: The true time points, shape (n_timesteps, 1), type: chex.Array
            - pred_x: The predicted states, shape (n_timesteps, n_states), type: chex.Array
            - true_x: The true states, shape (n_timesteps, n_states), type: chex.Array
            - pred_x_dot: The predicted state derivatives, shape (n_timesteps, n_states), type: chex.Array
            - true_x_dot: The true state derivatives, shape (n_timesteps, n_states), type: chex.Array
            - pred_t: The predicted time points, shape (n_timesteps, 1), type: chex.Array
            - state_labels: The labels for the states, type: list[str]
        The function returns a figure and axes object.
        pred_t is optional, in case the differentiator resamples the time points.    
        """
        if pred_t is None:
            pred_t = true_t
        chex.assert_shape(true_t, (None, self.input_dim))
        chex.assert_shape(pred_t, (None, self.input_dim))
        chex.assert_shape(pred_x, (None, self.output_dim))
        chex.assert_shape(true_x, (None, self.output_dim))
        chex.assert_shape(pred_x_dot, (None, self.output_dim))
        chex.assert_shape(true_x_dot, (None, self.output_dim))

        # Check if the true signals all have the same number of time steps
        assert true_x.shape[0] == true_x_dot.shape[0] == true_t.shape[0]
        # Check if the predicted signals all have the same number of time steps
        assert pred_x.shape[0] == pred_x_dot.shape[0] == pred_t.shape[0]

        fig, axes = plt.subplots(self.output_dim, 1, figsize=(16, 9))
        for j in range(self.output_dim):
            axes[j].plot(true_t, true_x[:,j], color=[0.2, 0.8, 0],label=r'$x_{TRUE}$')
            axes[j].plot(true_t, true_x_dot[:,j], color='green', label=r'$\dot{x}_{TRUE}$')
            axes[j].plot(pred_t, pred_x[:,j], color='orange', label=r'$x_{SMOOTHER}$')
            axes[j].plot(pred_t, pred_x_dot[:,j], color='red', label=r'$\dot{x}_{SMOOTHER}$')
            axes[j].grid(True, which='both')
        if state_labels is not None:
            for j in range(self.output_dim):
                axes[j].set_ylabel(state_labels[j])
        axes[2].set_xlabel(r'Time [s]')
        axes[2].legend()
        plt.tight_layout()

        return fig, axes
    
    def __call__(self, **kwargs):
        return self.differentiate(**kwargs)