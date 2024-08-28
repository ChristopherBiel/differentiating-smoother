import jax
import jax.numpy as jnp
import jax.random as jr
import chex
import optax
from typing import Union, Tuple

from jax import vmap, grad
from jaxtyping import PyTree
import matplotlib.pyplot as plt

from bsm.bayesian_regression.bayesian_neural_networks.bnn import BNNState, BayesianNeuralNet
from bsm.bayesian_regression.bayesian_neural_networks.deterministic_ensembles import DeterministicEnsemble
from bsm.bayesian_regression.bayesian_neural_networks.probabilistic_ensembles import ProbabilisticEnsemble
from bsm.utils.type_aliases import ModelState, StatisticalModelOutput, StatisticalModelState
from bsm.utils.normalization import Data
from bsm.utils.particle_distribution import ParticleDistribution
from bsm.utils.normalization import DataStats, Data

from diff_smoothers.base_differentiator import BaseDifferentiator, DifferentiatorState

class  ModelDerivativeDecorator:
    def __init__(self,
                 model: BayesianNeuralNet):
        self.model = model

    def _single_derivative(self,
                        params: PyTree,
                        x: chex.Array,
                        data_stats: DataStats) -> chex.Array:
        chex.assert_shape(x, (self.input_dim,))
        def sel_model_output(params, x, data_stats, ind) -> chex.Array:
            out, _ = self.apply_eval(params, x, data_stats)
            return out[ind]
        grad_single = grad(sel_model_output, argnums=1)
        v_apply = vmap(grad_single, in_axes=(None, None, None, 0))
        derivative = v_apply(params, x, data_stats, jnp.arange(self.output_dim)).reshape(-1)
        assert derivative.shape == (self.output_dim,)
        return derivative

    def derivative(self, input: chex.Array, bnn_state: BNNState) -> chex.Array:
        # Define derivative function for a single particle
        def _single_derivative(self,
                        params: PyTree,
                        x: chex.Array,
                        data_stats: DataStats) -> chex.Array:
            chex.assert_shape(x, (self.input_dim,))
            def sel_model_output(params, x, data_stats, ind) -> chex.Array:
                out, _ = self.apply_eval(params, x, data_stats)
                return out[ind]
            grad_single = grad(sel_model_output, argnums=1)
            v_apply = vmap(grad_single, in_axes=(None, None, None, 0))
            derivative = v_apply(params, x, data_stats, jnp.arange(self.output_dim)).reshape(-1)
            assert derivative.shape == (self.output_dim,)
            return derivative
        # Apply the derivative function to all particles
        chex.assert_shape(input, (self.input_dim,))
        v_apply = vmap(_single_derivative, in_axes=(None, 0, None, None), out_axes=0)
        derivative = v_apply(self, bnn_state.vmapped_params, input, bnn_state.data_stats)
        assert derivative.shape == (self.num_particles, self.output_dim)
        return ParticleDistribution(particle_means=derivative)
    
    def __getattr__(self, name):
        # Delegate all other attributes to the model
        return getattr(self.model, name)

class BNNSmootherDifferentiator(BaseDifferentiator):
    def __init__(self,
                 state_dim: int,
                 num_training_steps: Union[int, optax.Schedule] = 1000,
                 beta: chex.Array | optax.Schedule | None = None,
                 bnn_type: BayesianNeuralNet = DeterministicEnsemble,
                 *args, **kwargs):
        super().__init__(state_dim)
        self.bnn_type = bnn_type
        if isinstance(num_training_steps, int):
            self.num_training_steps = optax.constant_schedule(num_training_steps)
        else:
            self.num_training_steps = num_training_steps
        if beta is None:
            beta = jnp.ones(shape=(state_dim,))
        elif isinstance(beta, chex.Array):
            beta = optax.constant_schedule(beta)
        self._potential_beta = beta

        self.model = ModelDerivativeDecorator(self.bnn_type(input_dim=1,
                                                            output_dim=state_dim,
                                                            *args, **kwargs))

    def init(self,
             key = chex.PRNGKey) -> StatisticalModelState[BNNState]:
        # Override the function from StatisticalModel class
        model_state = self.model.init(key)
        beta = jnp.ones(self.state_dim)
        SmootherModelState = StatisticalModelState(beta=beta, model_state=model_state)
        return DifferentiatorState(input_data=None, key=key, algo_state=SmootherModelState)

    def train(self, key, data: Data) -> DifferentiatorState:
        assert data.inputs.ndim == 2
        assert data.outputs.ndim == 2
        assert data.inputs.shape[1] == 1
        assert data.outputs.shape[1] == self.state_dim
        assert data.outputs.shape[0] == data.inputs.shape[0]

        differentiator_state = self.init(key)
        data_size = data.inputs.shape[0]
        num_training_steps = self.num_training_steps(data_size)
        model_state = self.model.fit_model(data, num_training_steps, differentiator_state.algo_state.model_state)
        beta = self._potential_beta(data_size)
        assert beta.shape == (self.state_dim,)
        return DifferentiatorState(input_data=data, key=key, algo_state=StatisticalModelState(beta=beta, model_state=model_state))
    
    def differentiate(self,
                      state: DifferentiatorState[BNNState],
                      t: chex.Array) -> Tuple[DifferentiatorState[BNNState], chex.Array]:
        state, stats_model_output = self.differentiate_distribution(state, t)
        return state, stats_model_output.mean
    
    def differentiate_distribution(self,
                                   state: DifferentiatorState[BNNState],
                                   t: chex.Array) -> Tuple[DifferentiatorState[BNNState], StatisticalModelOutput[BNNState]]:
        chex.assert_shape(t, (None, 1))
        stats_model_output = vmap(self._derivative, in_axes=(0, None),
                                  out_axes=StatisticalModelOutput(mean=0, epistemic_std=0,
                                                                  aleatoric_std=0, statistical_model_state=None))\
                                  (t, state.algo_state)
        state.algo_state = stats_model_output.statistical_model_state
        state.algo_state.beta = self._potential_beta(state.input_data.inputs.shape[0])
        return state, stats_model_output
    
    def predict(self,
                state: DifferentiatorState[BNNState],
                t: chex.Array) -> chex.Array:
        prediction_distribution = self.predict_distribution(state, t)
        return prediction_distribution.mean

    def predict_distribution(self,
                             state: DifferentiatorState[BNNState],
                             t: chex.Array) -> StatisticalModelOutput[BNNState]:

        prediction_distribution = vmap(self._predict, in_axes=(0, None),
                                       out_axes=StatisticalModelOutput(mean=0, epistemic_std=0,
                                                                       aleatoric_std=0, statistical_model_state=None))\
                                       (t, state.algo_state)
        return prediction_distribution
    
    def _derivative(self,
                   input: chex.Array,
                   stats_model_state: StatisticalModelState[BNNState]) -> StatisticalModelOutput[BNNState]:
        part_dist = self.model.derivative(input=input, bnn_state=stats_model_state.model_state)
        statistical_model_output = StatisticalModelOutput(mean=part_dist.mean(), epistemic_std=part_dist.stddev(),
                                                   aleatoric_std=part_dist.aleatoric_std(),
                                                   statistical_model_state=stats_model_state)
        return statistical_model_output
    
    def _predict(self,
                 t: chex.Array, 
                 algo_state: StatisticalModelState[BNNState]) -> StatisticalModelOutput[BNNState]:
        chex.assert_shape(t, (1,))
        dist_f, dist_y = self.model.posterior(t, algo_state.model_state)
        out = StatisticalModelOutput(mean=dist_f.mean(), epistemic_std=dist_f.stddev(),
                                        aleatoric_std=dist_y.aleatoric_std(),
                                        statistical_model_state=algo_state)
        chex.assert_shape(out.mean, (self.state_dim,))
        return out
        
    
if __name__ == '__main__':

    # Create the data
    key = jr.PRNGKey(0)
    input_dim = 1
    state_dim = 1

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
    x = x + noise_level * jr.normal(key=jr.PRNGKey(0), shape=x.shape)
    data_std = noise_level * jnp.ones(shape=(state_dim,))
    print("Data shape: ", t.shape, x.shape, x_dot.shape)
    data = Data(inputs=t, outputs=x)

    diffr = BNNSmootherDifferentiator(state_dim=state_dim,
                                     output_stds=data_std,
                                     logging_wandb=False,
                                     beta=jnp.array([2.0]),
                                     num_particles=5,
                                     features=[64, 64],
                                     bnn_type=DeterministicEnsemble,
                                     train_share=1.0,
                                     num_training_steps=48_000,
                                     weight_decay=1e-4,
                                     return_best_model=True,
                                     eval_frequency=1_000,)

    differentiator_state = diffr.train(key=jr.PRNGKey(0), data=data)

    # Test on new data
    test_t = jnp.linspace(d_l-3, d_u+3, 300).reshape(-1, 1)
    test_x = f(test_t)

    pred_x = diffr.predict_distribution(differentiator_state, test_t)
    
    plt.scatter(test_t.reshape(-1), test_x, s=25, label='Data', color='red', alpha=0.5)
    plt.plot(test_t, pred_x.mean, label='Mean', color='blue')
    plt.fill_between(test_t.reshape(-1),
                     (pred_x.mean - differentiator_state.algo_state.beta * pred_x.epistemic_std).reshape(-1),
                     (pred_x.mean + differentiator_state.algo_state.beta * pred_x.epistemic_std).reshape(-1),
                     label=r'$2\sigma$', alpha=0.3, color='blue')
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.plot(test_t.reshape(-1), test_x, label='True', color='green')
    by_label = dict(zip(labels, handles))
    plt.axvline(x=d_l, color='black', linestyle='--')
    plt.axvline(x=d_u, color='black', linestyle='--')
    plt.legend(by_label.values(), by_label.keys())
    plt.grid(True, which='both')
    plt.show()
    plt.savefig('NNSmoother_extrapolate.pdf')
    plt.close()

    num_test_points = 200
    in_domain_test_t = jnp.linspace(d_l, d_u, num_test_points).reshape(-1, 1)
    in_domain_test_x = f(in_domain_test_t)
    in_domain_test_xdot = f_dot(in_domain_test_t)

    in_domain_preds = diffr.predict(differentiator_state, in_domain_test_t)
    differentiator_state, derivative = diffr.differentiate_distribution(differentiator_state, in_domain_test_t)
    plt.plot(in_domain_test_t, in_domain_preds, label=r'$x_{Est}$', color='blue')
    plt.plot(in_domain_test_t, in_domain_test_x, label=r'$x_{True}$', color='Green')
    plt.plot(in_domain_test_t, in_domain_test_xdot, label=r'$\dot{x}_{True}$', color='Red')
    plt.plot(in_domain_test_t, derivative.mean, label=r'$\dot{x}_{Est}$', color='Black')
    plt.fill_between(in_domain_test_t.reshape(-1),
                     (derivative.mean - differentiator_state.algo_state.beta * derivative.epistemic_std).reshape(-1),
                     (derivative.mean + differentiator_state.algo_state.beta * derivative.epistemic_std).reshape(-1),
                     label=r'$2\sigma$', alpha=0.3, color='Black')
    plt.grid(True, which='both')
    plt.legend()
    plt.show()
    plt.savefig('NNSmoother_interpolate.pdf')