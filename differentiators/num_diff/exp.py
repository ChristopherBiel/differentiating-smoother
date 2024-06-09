import jax
import jax.numpy as jnp
import jax.random as jr
from jax import vmap
import matplotlib.pyplot as plt
import argparse

from differentiators.num_diff.numerical_differentiation import fitMultiStateTikhonov
from bsm.utils.normalization import Data
from bsm.bayesian_regression.bayesian_neural_networks.deterministic_ensembles import DeterministicEnsemble
from bsm.bayesian_regression.bayesian_neural_networks.probabilistic_ensembles import ProbabilisticEnsemble
from bsm.bayesian_regression.bayesian_neural_networks.fsvgd_ensemble import DeterministicFSVGDEnsemble, ProbabilisticFSVGDEnsemble
from bsm.statistical_model.bnn_statistical_model import BNNStatisticalModel
from data_functions.data_creation import create_example_data, example_function_derivative
from data_functions.data_creation import sample_pendulum_with_input, sample_random_pendulum_data
from data_functions.data_handling import split_dataset
from data_functions.data_output import plot_derivative_data, plot_data
from differentiators.eval import evaluate_dyn_model

def experiment(project_name: str = 'LearnDynamicsModel',
               seed: int=0,
               num_traj: int = 12,
               sample_points: int = 64,
               noise_level: float = None,
               diff_regtype: str = 'first',
               diff_lambda: float = 0.1,
               dyn_feature_size: int = 128,
               dyn_hidden_layers: int = 2,
               dyn_particles: int = 10,
               dyn_training_steps: int = 1000,
               dyn_weight_decay: float = 1e-4,
               dyn_train_share: float = 0.8,
               dyn_type: str = 'DeterministicEnsemble',
               logging_mode_wandb: int = 0,
               x_src: str = 'smoother'):
    
    # Input checks
    assert num_traj > 0
    assert sample_points > 0
    assert diff_regtype in ['zero', 'first', 'second'], f"Unknown differentiation regularisation type: {diff_regtype}"
    assert dyn_type in ['DeterministicEnsemble', 'ProbabilisticEnsemble', 'DeterministicFSVGDEnsemble',
                        'ProbabilisticFSVGDEnsemble'], f"Unknown dyanmics BNN type: {dyn_type}"
    
    config = dict(seed=seed,
                  num_traj=num_traj,
                  sample_points=sample_points,
                  noise_level=noise_level,
                  diff_regtype=diff_regtype,
                  diff_lambda=diff_lambda,
                  dyn_feature_size=dyn_feature_size,
                  dyn_hidden_layers=dyn_hidden_layers,
                  dyn_particles=dyn_particles,
                  dyn_training_steps=dyn_training_steps,
                  dyn_weight_decay=dyn_weight_decay,
                  dyn_train_share=dyn_train_share,
                  dyn_type=dyn_type,
                  logging_mode_wandb=logging_mode_wandb,
                  x_src=x_src)
    
    if logging_mode_wandb > 0:
        import wandb
        wandb.init(project=project_name,
                   config=config,)


    # Create the data
    key = jr.PRNGKey(seed=seed)
    t, x, u, x_dot = sample_random_pendulum_data(num_points=sample_points,
                                                 noise_level=noise_level,
                                                 key=key,
                                                 num_trajectories=num_traj,
                                                 initial_states=None,)
    state_dim = x.shape[-1]
    control_dim = u.shape[-1]

    if logging_mode_wandb > 0:
        fig = plot_data(t, x, u, x_dot, title='One trajectory of the sampled training data (pendulum)')
        wandb.log({'Training Data': wandb.Image(fig)})

    if logging_mode_wandb > 1:
        logging_dyn_wandb = True
    else:
        logging_smoother_wandb = False
        logging_dyn_wandb = False

    # Calculate the derivative
    v_apply = vmap(fitMultiStateTikhonov, in_axes=(0, 0, None, None, 0))
    pred_x, pred_x_dot = v_apply(t, x, diff_regtype, diff_lambda, x[:,0,:])

    # Plot the results for the first three trajectories
    if logging_mode_wandb > 0:
        fig, axes = plt.subplots(state_dim, min(2, num_traj), figsize=(16, 9))
        for i in range(min(2, num_traj)):
            for j in range(state_dim):
                axes[j][i].plot(t[i,:], x[i,:,j], color=[0.2, 0.8, 0], label=r'x')
                axes[j][i].plot(t[i,:], x_dot[i,:,j], color='green', label=r'$\dot{x}_{TRUE}$')
                axes[j][i].plot(t[i,:], pred_x[i,:,j], color='orange', label=r'$x_{SMOOTHER}$')
                axes[j][i].plot(t[i,:], pred_x_dot[i,:,j], color='red', label=r'$\dot{x}_{SMOOTHER}$')
                axes[j][i].set_title(f"Trajectory {i} - x{j}")
                axes[j][i].grid(True, which='both')
            axes[0][i].set_title(r'Trajectory %s'%(str(i)))
            axes[-1][i].set_xlabel(r'Time [s]')
            axes[-1][i].legend()
            axes[-1][i].legend()
        # Add labels, legends and titles
        axes[0][0].set_ylabel(r'$cos(\theta)$')
        axes[1][0].set_ylabel(r'$sin(\theta)$')
        axes[2][0].set_ylabel(r'$\omega$')
        plt.legend()
        plt.tight_layout()
        wandb.log({'smoother': wandb.Image(plt)})
    
    # -------------------- Dynamics Model --------------------
    # The split data is concatinated again and add the input
    if x_src == 'smoother':
        smoother_x = pred_x.reshape(-1, state_dim)
        inputs = jnp.concatenate([smoother_x, u.reshape(-1,control_dim)], axis=-1)
    elif x_src == 'data':
        inputs = jnp.concatenate([x.reshape(-1, state_dim), u.reshape(-1,control_dim)], axis=-1)
    else:
        raise ValueError(f"No x source {x_src}")
    outputs = pred_x_dot.reshape(-1, state_dim)

    if noise_level is None:
        data_std = jnp.ones(shape=(state_dim,))
    else:
        data_std = noise_level * jnp.ones(shape=(state_dim,))

    dyn_data = Data(inputs=inputs, outputs=outputs)
    dyn_features = [dyn_feature_size] * dyn_hidden_layers
    if dyn_type == 'DeterministicEnsemble':
        dyn_model = BNNStatisticalModel(input_dim=state_dim+control_dim,
                                        output_dim=state_dim,
                                        output_stds=data_std,
                                        logging_wandb=logging_dyn_wandb,
                                        beta=jnp.ones(shape=(state_dim,))*2,
                                        num_particles=dyn_particles,
                                        features=dyn_features,
                                        bnn_type=DeterministicEnsemble,
                                        train_share=dyn_train_share,
                                        num_training_steps=dyn_training_steps,
                                        weight_decay=dyn_weight_decay,
                                        return_best_model=True,
                                        eval_frequency=1000,
                                        )
    elif dyn_type == 'ProbabilisticEnsemble':
        dyn_model = BNNStatisticalModel(input_dim=state_dim+control_dim,
                                        output_dim=state_dim,
                                        output_stds=data_std,
                                        logging_wandb=logging_dyn_wandb,
                                        beta=jnp.ones(shape=(state_dim,))*2,
                                        num_particles=dyn_particles,
                                        features=dyn_features,
                                        bnn_type=ProbabilisticEnsemble,
                                        train_share=dyn_train_share,
                                        num_training_steps=dyn_training_steps,
                                        weight_decay=dyn_weight_decay,
                                        return_best_model=True,
                                        eval_frequency=1000,
                                        )
    elif dyn_type == 'DeterministicFSVGDEnsemble':
        dyn_model = BNNStatisticalModel(input_dim=state_dim+control_dim,
                                        output_dim=state_dim,
                                        output_stds=data_std,
                                        logging_wandb=logging_dyn_wandb,
                                        beta=jnp.ones(shape=(state_dim,))*2,
                                        num_particles=dyn_particles,
                                        features=dyn_features,
                                        bnn_type=DeterministicFSVGDEnsemble,
                                        train_share=dyn_train_share,
                                        num_training_steps=dyn_training_steps,
                                        weight_decay=dyn_weight_decay,
                                        return_best_model=True,
                                        eval_frequency=1000,
                                        )
    elif dyn_type == 'ProbabilisticFSVGDEnsemble':
        dyn_model = BNNStatisticalModel(input_dim=state_dim+control_dim,
                                        output_dim=state_dim,
                                        output_stds=data_std,
                                        logging_wandb=logging_dyn_wandb,
                                        beta=jnp.ones(shape=(state_dim,))*2,
                                        num_particles=dyn_particles,
                                        features=dyn_features,
                                        bnn_type=ProbabilisticFSVGDEnsemble,
                                        train_share=dyn_train_share,
                                        num_training_steps=dyn_training_steps,
                                        weight_decay=dyn_weight_decay,
                                        return_best_model=True,
                                        eval_frequency=1000,
                                        )
    
    dyn_model_state = dyn_model.init(key)
    dyn_model_state = dyn_model.update(dyn_model_state, dyn_data)
    dyn_preds = dyn_model.predict_batch(dyn_data.inputs, dyn_model_state)

    # Plot the results for the first trajectory only
    if logging_mode_wandb > 0:
        fig = plot_derivative_data(t=t.reshape(-1, 1),
                                   x=x.reshape(-1, state_dim),
                                   x_dot_true=x_dot.reshape(-1, state_dim),
                                   x_dot_est = dyn_preds.mean,
                                   x_dot_est_std=dyn_preds.epistemic_std,
                                   x_dot_smoother=pred_x_dot,
                                   x_dot_smoother_std=None,
                                   beta=dyn_preds.statistical_model_state.beta,
                                   source='DYN. MODEL',
                                   num_trajectories_to_plot=2,
                                   )
        wandb.log({'dynamics': wandb.Image(fig)})

    # Evaluate the dynamics model:
    state_pred_mse, derivative_pred_plot, state_pred_plot = evaluate_dyn_model(dyn_model=dyn_model,
                                                         dyn_model_state=dyn_model_state,
                                                         num_points=32,
                                                         plot_data=True,
                                                         return_performance=True,
                                                         plot_annotation_source="DYN,NUM-DIFF")

    if logging_mode_wandb > 0:
        wandb.log({'derivative_prediction': wandb.Image(derivative_pred_plot)})
        wandb.log({'state_prediction': wandb.Image(state_pred_plot)})
        wandb.log({'state_prediction_mse': state_pred_mse})


def main(args):
    experiment(project_name=args.project_name,
               seed=args.seed,
               num_traj=args.num_traj,
               sample_points=args.sample_points,
               noise_level=args.noise_level,
               diff_regtype=args.diff_regtype,
               diff_lambda=args.diff_lambda,
               dyn_feature_size=args.dyn_feature_size,
               dyn_hidden_layers=args.dyn_hidden_layers,
               dyn_particles=args.dyn_particles,
               dyn_training_steps=args.dyn_training_steps,
               dyn_weight_decay=args.dyn_weight_decay,
               dyn_train_share=args.dyn_train_share,
               dyn_type=args.dyn_type,
               logging_mode_wandb=args.logging_mode_wandb,
               x_src=args.x_src)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_name', type=str, default='LearnDynamicsModel')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_traj', type=int, default=12)
    parser.add_argument('--noise_level', type=float, default=0.01)
    parser.add_argument('--sample_points', type=int, default=64)
    parser.add_argument('--diff_regtype', type=str, default='second')
    parser.add_argument('--diff_lambda', type=float, default=0.002)
    parser.add_argument('--dyn_feature_size', type=int, default=128)
    parser.add_argument('--dyn_hidden_layers', type=int, default=2)
    parser.add_argument('--dyn_particles', type=int, default=6)
    parser.add_argument('--dyn_training_steps', type=int, default=32000)
    parser.add_argument('--dyn_weight_decay', type=float, default=3e-4)
    parser.add_argument('--dyn_train_share', type=float, default=0.8)
    parser.add_argument('--dyn_type', type=str, default='DeterministicFSVGDEnsemble')
    parser.add_argument('--logging_mode_wandb', type=int, default=2)
    parser.add_argument('--x_src', type=str, default='smoother')
    args = parser.parse_args()
    main(args)