import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import argparse

from bsm.utils.normalization import Data
from diff_smoothers.smoother_net import SmootherNet
from bsm.bayesian_regression.bayesian_neural_networks.deterministic_ensembles import DeterministicEnsemble
from bsm.bayesian_regression.bayesian_neural_networks.probabilistic_ensembles import ProbabilisticEnsemble
from bsm.bayesian_regression.bayesian_neural_networks.fsvgd_ensemble import DeterministicFSVGDEnsemble, ProbabilisticFSVGDEnsemble
from bsm.statistical_model.bnn_statistical_model import BNNStatisticalModel
from diff_smoothers.data_functions.data_creation import create_example_data, example_function_derivative
from diff_smoothers.data_functions.data_creation import sample_pendulum_with_input, sample_random_pendulum_data
from diff_smoothers.data_functions.data_handling import split_dataset
from diff_smoothers.data_functions.data_output import plot_derivative_data, plot_data
from diff_smoothers.eval import evaluate_dyn_model

def experiment(project_name: str = 'DiffSmoother',
               seed: int = 0,
               num_traj: int = 1,
               sample_points: int = 64,
               noise_level: float = 0.01,
               smoother_feature_size: int = 64,
               dyn_feature_size: int = 128,
               smoother_hidden_layers: int = 2,
               dyn_hidden_layers: int = 2,
               smoother_particles: int = 10,
               dyn_particles: int = 5,
               smoother_training_steps: int = 4000,
               dyn_training_steps: int = 8000,
               smoother_weight_decay: float = 1e-3,
               dyn_weight_decay: float = 1e-3,
               smoother_train_share: float = 0.8,
               dyn_train_share: float = 0.8,
               smoother_type: str = 'DeterministicEnsemble',
               dyn_type: str = 'DeterministicEnsemble',
               logging_mode_wandb: int = 0,
               x_src: str = 'smoother',
               return_model_state: bool = False,):
    
    # Input checks
    assert num_traj == 1
    assert sample_points > 0
    assert smoother_type in ['DeterministicEnsemble', 'ProbabilisticEnsemble', 'DeterministicFSVGDEnsemble',
                             'ProbabilisticFSVGDEnsemble'], f"Unknown smoother BNN type: {smoother_type}"
    assert dyn_type in ['DeterministicEnsemble', 'ProbabilisticEnsemble', 'DeterministicFSVGDEnsemble',
                        'ProbabilisticFSVGDEnsemble'], f"Unknown dyanmics BNN type: {dyn_type}"
    
    config = dict(seed=seed,
                  num_traj=num_traj,
                  sample_points=sample_points,
                  noise_level=noise_level,
                  smoother_feature_size=smoother_feature_size,
                  dyn_feature_size=dyn_feature_size,
                  smoother_hidden_layers=smoother_hidden_layers,
                  dyn_hidden_layers=dyn_hidden_layers,
                  smoother_particles=smoother_particles,
                  dyn_particles=dyn_particles,
                  smoother_training_steps=smoother_training_steps,
                  dyn_training_steps=dyn_training_steps,
                  smoother_weight_decay=smoother_weight_decay,
                  dyn_weight_decay=dyn_weight_decay,
                  smoother_train_share=smoother_train_share,
                  dyn_train_share=dyn_train_share,
                  smoother_type=smoother_type,
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
                                                 key=key,)

    if logging_mode_wandb > 0:
        fig = plot_data(t, x, u, x_dot, title='Training data (pendulum)')
        wandb.log({'Training Data': wandb.Image(fig)})

    smoother_data = Data(inputs=t, outputs=x)

    input_dim = smoother_data.inputs.shape[-1]
    output_dim = smoother_data.outputs.shape[-1]
    control_dim = u.shape[-1]
    if noise_level is None:
        data_std = jnp.ones(shape=(output_dim,)) * 0.001
    else:
        data_std = noise_level * jnp.ones(shape=(output_dim,))

    if logging_mode_wandb > 2:
        logging_smoother_wandb = True
        logging_dyn_wandb = True
    elif logging_mode_wandb == 2:
        logging_smoother_wandb = False
        logging_dyn_wandb = True
    else:
        logging_smoother_wandb = False
        logging_dyn_wandb = False
    

    # -------------------- Smoother --------------------
    smoother_features = [smoother_feature_size] * smoother_hidden_layers
    if smoother_type == 'DeterministicEnsemble':
        model = SmootherNet(input_dim=input_dim,
                            output_dim=output_dim,
                            output_stds=data_std,
                            logging_wandb=logging_smoother_wandb,
                            beta=jnp.ones(shape=(output_dim,))*3,
                            num_particles=dyn_particles,
                            features=smoother_features,
                            bnn_type=DeterministicEnsemble,
                            train_share=smoother_train_share,
                            num_training_steps=smoother_training_steps,
                            weight_decay=smoother_weight_decay,
                            return_best_model=True,
                            eval_frequency=1000,
                            )
    elif smoother_type == 'ProbabilisticEnsemble':
        model = SmootherNet(input_dim=input_dim,
                            output_dim=output_dim,
                            output_stds=data_std,
                            logging_wandb=logging_smoother_wandb,
                            beta=jnp.ones(shape=(output_dim,))*3,
                            num_particles=dyn_particles,
                            features=smoother_features,
                            bnn_type=ProbabilisticEnsemble,
                            train_share=smoother_train_share,
                            num_training_steps=smoother_training_steps,
                            weight_decay=smoother_weight_decay,
                            return_best_model=True,
                            eval_frequency=1000,
                            )
    elif smoother_type == 'DeterministicFSVGDEnsemble':
        model = SmootherNet(input_dim=input_dim,
                            output_dim=output_dim,
                            output_stds=data_std,
                            logging_wandb=logging_smoother_wandb,
                            beta=jnp.ones(shape=(output_dim,))*3,
                            num_particles=dyn_particles,
                            features=smoother_features,
                            bnn_type=DeterministicFSVGDEnsemble,
                            train_share=smoother_train_share,
                            num_training_steps=smoother_training_steps,
                            weight_decay=smoother_weight_decay,
                            return_best_model=True,
                            eval_frequency=1000,
                            )
    elif smoother_type == 'ProbabilisticFSVGDEnsemble':
        model = SmootherNet(input_dim=input_dim,
                            output_dim=output_dim,
                            output_stds=data_std,
                            logging_wandb=logging_smoother_wandb,
                            beta=jnp.ones(shape=(output_dim,))*3,
                            num_particles=dyn_particles,
                            features=smoother_features,
                            bnn_type=ProbabilisticFSVGDEnsemble,
                            train_share=smoother_train_share,
                            num_training_steps=smoother_training_steps,
                            weight_decay=smoother_weight_decay,
                            return_best_model=True,
                            eval_frequency=1000,
                            )
    else:
        raise NotImplementedError(f"Unknown BNN type: {smoother_type}")
    
    model_states = model.train_new_smoother(key, smoother_data)
    pred_x = model.predict_batch(t, model_states)
    ders = model.derivative_batch(t, model_states)

    # Plot the results for the first three trajectories
    if logging_mode_wandb > 0:
        fig, axes = plt.subplots(output_dim, 1, figsize=(16, 9))
        for j in range(output_dim):
            axes[j].plot(smoother_data.inputs[:], smoother_data.outputs[:,j], color=[0.2, 0.8, 0],label=r'x')
            axes[j].plot(smoother_data.inputs[:], x_dot[:,j], color='green', label=r'$\dot{x}_{TRUE}$')
            axes[j].plot(smoother_data.inputs[:], pred_x.mean[:,j], color='orange', label=r'$x_{SMOOTHER}$')
            axes[j].plot(smoother_data.inputs[:], ders.mean[:,j], color='red', label=r'$\dot{x}_{SMOOTHER}$')
            axes[j].fill_between(smoother_data.inputs[:].reshape(-1),
                                    (ders.mean[:,j] - ders.statistical_model_state.beta[j] * ders.epistemic_std[:,j]).reshape(-1),
                                    (ders.mean[:,j] + ders.statistical_model_state.beta[j] * ders.epistemic_std[:,j]).reshape(-1),
                                    label=r'$2\sigma$', alpha=0.3, color='red')
            axes[j].grid(True, which='both')
        axes[0].set_ylabel(r'$cos(\theta)$')
        axes[1].set_ylabel(r'$sin(\theta)$')
        axes[2].set_ylabel(r'$\omega$')
        axes[2].set_xlabel(r'Time [s]')
        axes[2].legend()
        axes[2].legend()
        plt.tight_layout()
        wandb.log({'smoother': wandb.Image(plt)})

    # -------------------- Dynamics Model --------------------
    # The split data is concatinated again and add the input
    if x_src == 'smoother':
        smoother_x = pred_x.mean.reshape(-1, output_dim)
        inputs = jnp.concatenate([smoother_x, u.reshape(-1,control_dim)], axis=-1)
    elif x_src == 'data':
        inputs = jnp.concatenate([x.reshape(-1, output_dim), u.reshape(-1,control_dim)], axis=-1)
    else:
        raise ValueError(f"No x source {x_src}")
    outputs = ders.mean.reshape(-1, output_dim)

    dyn_data = Data(inputs=inputs, outputs=outputs)
    dyn_features = [dyn_feature_size] * dyn_hidden_layers
    if dyn_type == 'DeterministicEnsemble':
        dyn_model = BNNStatisticalModel(input_dim=output_dim+control_dim,
                                        output_dim=output_dim,
                                        output_stds=data_std,
                                        logging_wandb=logging_dyn_wandb,
                                        beta=jnp.ones(shape=(output_dim,))*2,
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
        dyn_model = BNNStatisticalModel(input_dim=output_dim+control_dim,
                                        output_dim=output_dim,
                                        output_stds=data_std,
                                        logging_wandb=logging_dyn_wandb,
                                        beta=jnp.ones(shape=(output_dim,))*2,
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
        dyn_model = BNNStatisticalModel(input_dim=output_dim+control_dim,
                                        output_dim=output_dim,
                                        output_stds=data_std,
                                        logging_wandb=logging_dyn_wandb,
                                        beta=jnp.ones(shape=(output_dim,))*2,
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
        dyn_model = BNNStatisticalModel(input_dim=output_dim+control_dim,
                                        output_dim=output_dim,
                                        output_stds=data_std,
                                        logging_wandb=logging_dyn_wandb,
                                        beta=jnp.ones(shape=(output_dim,))*2,
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
                                   x=smoother_data.outputs.reshape(-1, output_dim),
                                   x_dot_true=x_dot.reshape(-1, output_dim),
                                   x_dot_est=dyn_preds.mean,
                                   x_dot_est_std=dyn_preds.epistemic_std,
                                   x_dot_smoother=ders.mean,
                                   x_dot_smoother_std=None,
                                   beta=dyn_preds.statistical_model_state.beta,
                                   source='DYN. MODEL',
                                   num_trajectories_to_plot=1,
                                   )
        wandb.log({'dynamics': wandb.Image(fig)})

    # Evaluate the dynamics model:
    state_pred_mse, derivative_pred_plot, state_pred_plot = evaluate_dyn_model(dyn_model=dyn_model,
                                                         dyn_model_state=dyn_model_state,
                                                         num_points=32,
                                                         seed = 1,
                                                         plot_data=True,
                                                         return_performance=True,
                                                         plot_annotation_source="DYN,SMOOTHER")

    if logging_mode_wandb > 0:
        wandb.log({'derivative_prediction': wandb.Image(derivative_pred_plot)})
        wandb.log({'state_prediction': wandb.Image(state_pred_plot)})
        wandb.log({'state_prediction_mse': state_pred_mse})

    if return_model_state:
        return dyn_model, dyn_model_state

def main(args):
    experiment(project_name=args.project_name,
               seed=args.seed,
               num_traj=args.num_traj,
               sample_points=args.sample_points,
               noise_level=args.noise_level,
               smoother_feature_size=args.smoother_feature_size,
               dyn_feature_size=args.dyn_feature_size,
               smoother_hidden_layers=args.smoother_hidden_layers,
               dyn_hidden_layers=args.dyn_hidden_layers,
               smoother_particles=args.smoother_particles,
               dyn_particles=args.dyn_particles,
               smoother_training_steps=args.smoother_training_steps,
               dyn_training_steps=args.dyn_training_steps,
               smoother_weight_decay=args.smoother_weight_decay,
               dyn_weight_decay=args.dyn_weight_decay,
               smoother_train_share=args.smoother_train_share,
               dyn_train_share=args.dyn_train_share,
               smoother_type=args.smoother_type,
               dyn_type=args.dyn_type,
               logging_mode_wandb=args.logging_mode_wandb,
               x_src=args.x_src)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_name', type=str, default='DiffSmoother')
    parser.add_argument('--seed', type=int, default=12)
    parser.add_argument('--num_traj', type=int, default=1)
    parser.add_argument('--noise_level', type=float, default=0.01)
    parser.add_argument('--sample_points', type=int, default=64)
    parser.add_argument('--smoother_feature_size', type=int, default=64)
    parser.add_argument('--dyn_feature_size', type=int, default=128)
    parser.add_argument('--smoother_hidden_layers', type=int, default=2)
    parser.add_argument('--dyn_hidden_layers', type=int, default=2)
    parser.add_argument('--smoother_particles', type=int, default=12)
    parser.add_argument('--dyn_particles', type=int, default=6)
    parser.add_argument('--smoother_training_steps', type=int, default=4_000)
    parser.add_argument('--dyn_training_steps', type=int, default=32_000)
    parser.add_argument('--smoother_weight_decay', type=float, default=3e-4)
    parser.add_argument('--dyn_weight_decay', type=float, default=3e-4)
    parser.add_argument('--smoother_train_share', type=float, default=1.0)
    parser.add_argument('--dyn_train_share', type=float, default=0.8)
    parser.add_argument('--smoother_type', type=str, default='DeterministicEnsemble')
    parser.add_argument('--dyn_type', type=str, default='DeterministicFSVGDEnsemble')
    parser.add_argument('--logging_mode_wandb', type=int, default=2)
    parser.add_argument('--x_src', type=str, default='smoother')
    args = parser.parse_args()
    main(args)