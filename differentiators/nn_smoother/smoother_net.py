import jax
import jax.numpy as jnp
import jax.random as jr
import chex
import optax

from jax import vmap, grad
from jaxtyping import PyTree

from bsm.bayesian_regression.bayesian_neural_networks.bnn import BNNState, BayesianNeuralNet
from bsm.bayesian_regression.bayesian_neural_networks.deterministic_ensembles import DeterministicEnsemble
from bsm.bayesian_regression.bayesian_neural_networks.probabilistic_ensembles import ProbabilisticEnsemble
from bsm.statistical_model.bnn_statistical_model import BNNStatisticalModel
from bsm.utils.type_aliases import ModelState, StatisticalModelOutput, StatisticalModelState
from bsm.utils.normalization import Data
from bsm.utils.particle_distribution import ParticleDistribution
from bsm.utils.normalization import DataStats, Data

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
    print(f"Derivative shape: {derivative.shape}")
    assert derivative.shape == (self.num_particles, self.output_dim)
    return ParticleDistribution(particle_means=derivative)

class SmootherNet(BNNStatisticalModel):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 num_training_steps: int = 1000,
                 *args, **kwargs):
        super().__init__(input_dim=input_dim,
                       output_dim=output_dim,
                       num_training_steps=num_training_steps,
                       *args, **kwargs)
        self.model._single_derivative = _single_derivative
        self.model.derivative = derivative

    def _learnOneTrajectory(self, key, data: Data) -> StatisticalModelOutput[BNNState]:
        init_model_state = self.init(key)
        print("Data shape for learning: ", data.inputs.shape, data.outputs.shape)
        print("Data types for learning: ", type(data.inputs), type(data.outputs))
        def update_fn(model_state, data):
            input = jax.device_get(data.inputs)
            output = jax.device_get(data.outputs)
            data = Data(inputs=input, outputs=output)
            print("Data types for learning after get: ", type(data.inputs), type(data.outputs))
            return self.update(model_state, data)
        model_state = update_fn(model_state=init_model_state, data=data)
        return model_state
    
    def _derivative(self,
                   input: chex.Array,
                   stats_model_state: StatisticalModelState[ModelState]) -> StatisticalModelOutput[ModelState]:
        chex.assert_shape(input, (self.input_dim,))
        part_dist = self.model.derivative(self.model, input=input, bnn_state=stats_model_state.model_state)
        statistical_model = StatisticalModelOutput(mean=part_dist.mean(), epistemic_std=part_dist.stddev(),
                                                   aleatoric_std=part_dist.aleatoric_std(),
                                                   statistical_model_state=stats_model_state)
        return statistical_model
    
    def _derivative_batch(self,
                         input: chex.Array,
                         statistical_model_state: StatisticalModelState[ModelState]) -> StatisticalModelOutput[ModelState]:
        ders = vmap(self._derivative, in_axes=(0, self.vmap_input_axis(0)),
                     out_axes=self.vmap_output_axis(0))(input, statistical_model_state)
        return ders
    
    def learnSmoothers(self, key, data: Data) -> StatisticalModelState[BNNState]:
        learn = vmap(self._learnOneTrajectory, in_axes=(0,0), out_axes=0)
        keys = jr.split(key, data.inputs.shape[0])
        model_states = learn(keys, data)
        return model_states
    
    def calcDerivative(self, model_states: StatisticalModelState[BNNState],
                       data: Data) -> StatisticalModelOutput[BNNState]:
        # Split the different trajectories in the data into separate datasets
        v_apply = vmap(self._derivative_batch, in_axes=(0, 0), out_axes=0)
        derivatives = v_apply(data.inputs, model_states)
        return derivatives
    
    def smoother_predict(self,
                      input: chex.Array,
                      statistical_model_state: StatisticalModelState[BNNState]) -> StatisticalModelOutput[BNNState]:
        v_apply = vmap(self.predict_batch, in_axes=(0, 0), out_axes=0)
        preds = v_apply(input, statistical_model_state)
        return preds
    
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # Create the data
    key = jr.PRNGKey(0)
    input_dim = 1
    output_dim = 1

    noise_level = 0.1
    d_l, d_u = 0, 10
    num_samples = 64
    t = jnp.linspace(d_l, d_u, num_samples).reshape(-1, 1)
    x = jnp.sin(t) * jnp.cos(0.2*t)
    x_dot = jnp.sin(t) * (-0.2) * jnp.sin(0.2*t) + jnp.cos(t) * jnp.cos(0.2*t)
    x = x + noise_level * jr.normal(key=jr.PRNGKey(0), shape=x.shape)
    data_std = noise_level * jnp.ones(shape=(output_dim,))
    print("Data shape: ", t.shape, x.shape, x_dot.shape)
    data = Data(inputs=t, outputs=x)

    model = SmootherNet(input_dim=input_dim,
                        output_dim=output_dim,
                        output_stds=data_std,
                        logging_wandb=False,
                        beta=jnp.array([1.0]),
                        num_particles=10,
                        features=[64, 32],
                        bnn_type=DeterministicEnsemble,
                        train_share=0.6,
                        num_training_steps=4000,
                        weight_decay=1e-4,
                        return_best_model=True,
                        eval_frequency=1000,)

    model_states = model._learnOneTrajectory(key=jr.PRNGKey(0), data=data)

    # Test on new data
    test_t = jnp.linspace(d_l-5, d_u+5, 1000).reshape(-1, 1)
    test_x = jnp.sin(test_t) * jnp.cos(0.2*test_t)
    
    preds = model.predict_batch(test_t, model_states)
    
    plt.scatter(test_t.reshape(-1), test_x, label='Data', color='red')
    plt.plot(test_t, preds.mean, label='Mean', color='blue')
    plt.fill_between(test_t.reshape(-1),
                     (preds.mean - preds.statistical_model_state.beta * preds.epistemic_std).reshape(-1),
                     (preds.mean + preds.statistical_model_state.beta * preds.epistemic_std).reshape(-1),
                     label=r'$2\sigma$', alpha=0.3, color='blue')
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.plot(test_t.reshape(-1), test_x, label='True', color='green')
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.show()

    num_test_points = 1000
    in_domain_test_t = jnp.linspace(d_l, d_u, num_test_points).reshape(-1, 1)
    in_domain_test_x = jnp.sin(in_domain_test_t) * jnp.cos(0.2*in_domain_test_t)
    in_domain_test_xdot = jnp.sin(in_domain_test_t) * (-0.2) * jnp.sin(0.2*in_domain_test_t) + jnp.cos(in_domain_test_t) * jnp.cos(0.2*in_domain_test_t)

    in_domain_preds = model.predict_batch(in_domain_test_t, model_states)
    derivative = model._derivative_batch(in_domain_test_t, model_states).mean
    plt.plot(in_domain_test_t, in_domain_preds.mean, label='Mean', color='blue')
    plt.plot(in_domain_test_t, in_domain_test_x, label='Fun', color='Green')
    plt.plot(in_domain_test_t, in_domain_test_xdot, label='Derivative', color='Red')
    plt.plot(in_domain_test_t, derivative, label='Predicted Derivative', color='Black')
    plt.grid(True, which='both')
    plt.legend()
    plt.show()
    plt.savefig('bnn.pdf')