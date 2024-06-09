import matplotlib.pyplot as plt
import jax.numpy as jnp
import chex
from bsm.utils.normalization import Data
from data_functions.data_handling import split_dataset

def plot_derivative_data(t: chex.Array,
                         x: chex.Array,
                         x_dot_true: chex.Array,
                         x_dot_est: chex.Array,
                         x_dot_est_std: chex.Array,
                         beta: chex.Array,
                         source: str = "",
                         x_dot_smoother: chex.Array = None,
                         x_dot_smoother_std: chex.Array = None,
                         num_trajectories_to_plot: int = 1,
                         state_labels: str = [r'$cos(\theta)$', r'$sin(\theta)$', r'$\omega$'],
                         ) -> plt.figure:
    """Either pass all states and values with three dimensions (num_traj, num_samples, num_states)
    OR pass all states and values with only two dimensions (num_traj*num_samples, num_states)"""
    
    state_dim = x.shape[-1]
    assert x.shape == x_dot_true.shape == x_dot_est.shape == x_dot_est_std.shape
    assert num_trajectories_to_plot > 0, "The number of trajectories too plot must be more than zero"
    
    # Data has to be split again to be able to plot individual trajectories easier
    if x.ndim == 2:
        data = Data(inputs=t, outputs=jnp.concatenate([x, x_dot_true, x_dot_est, x_dot_est_std], axis=-1))
        data, num_trajectories = split_dataset(data)

        t = data.inputs
        x = data.outputs[:,:,:state_dim]
        x_dot_true = data.outputs[:,:,state_dim:state_dim*2]
        x_dot_est = data.outputs[:,:,state_dim*2:state_dim*3]
        x_dot_est_std = data.outputs[:,:,state_dim*3:]

    fig, axes = plt.subplots(state_dim, num_trajectories_to_plot, figsize=(16,9))
    for k01 in range(state_dim):
        if num_trajectories_to_plot > 1:
            for k02 in range(num_trajectories_to_plot):
                axes[k01][k02].plot(t[k02,:,0].reshape(-1), x_dot_est[k02,:,k01], color='blue', label=r'$\dot{x}_{%s}$'%(source))
                axes[k01][k02].fill_between(t[0,:,0].reshape(-1),
                                            (x_dot_est[k02,:,k01] - beta[k01] * x_dot_est_std[k02,:,k01]).reshape(-1),
                                            (x_dot_est[k02,:,k01] + beta[k01] * x_dot_est_std[k02,:,k01]).reshape(-1),
                                            label=r'$2\sigma$', alpha=0.3, color='blue')
                if x_dot_smoother is not None:
                    axes[k01][k02].plot(t[k02,:,0].reshape(-1), x_dot_smoother[k02,:,k01], color='red', label=r'$\dot{x}_{SMOOTHER}$')
                if x_dot_smoother_std is not None:
                    axes[k01][k02].fill_between(t[0,:,0].reshape(-1),
                                                (x_dot_smoother[k02,:,k01] - beta[k01] * x_dot_smoother_std[k02,:,k01]).reshape(-1),
                                                (x_dot_smoother[k02,:,k01] + beta[k01] * x_dot_smoother_std[k02,:,k01]).reshape(-1),
                                                label=r'$2\sigma$', alpha=0.3, color='red')
                axes[k01][k02].plot(t[0,:,0].reshape(-1), x_dot_true[k02,:,k01], color='green', label=r'$\dot{x}_{TRUE}$')
                axes[k01][k02].set_ylabel(state_labels[k01])
                axes[k01][k02].set_xlabel(r'Time [s]')
                axes[k01][k02].set_title(r'Trajectory %s'%(str(k02)))
                axes[k01][k02].grid(True, which='both')
        else:
            axes[k01].plot(t[0,:,0].reshape(-1), x_dot_est[0,:,k01], color='blue', label=r'$\dot{x}_{%s}$'%(source))
            axes[k01].fill_between(t[0,:,0].reshape(-1),
                                    (x_dot_est[0,:,k01] - beta[k01] * x_dot_est_std[0,:,k01]).reshape(-1),
                                    (x_dot_est[0,:,k01] + beta[k01] * x_dot_est_std[0,:,k01]).reshape(-1),
                                    label=r'$2\sigma$', alpha=0.3, color='blue')
            if x_dot_smoother is not None:
                axes[k01].plot(t[0,:,0].reshape(-1), x_dot_smoother[0,:,k01], color='red', label=r'$\dot{x}_{SMOOTHER}$')
            if x_dot_smoother_std is not None:
                axes[k01].fill_between(t[0,:,0].reshape(-1),
                                        (x_dot_smoother[0,:,k01] - beta[k01] * x_dot_smoother_std[0,:,k01]).reshape(-1),
                                        (x_dot_smoother[0,:,k01] + beta[k01] * x_dot_smoother_std[0,:,k01]).reshape(-1),
                                        label=r'$2\sigma$', alpha=0.3, color='red')
            axes[k01].plot(t[0,:,0].reshape(-1), x_dot_true[0,:,k01], color='green', label=r'$\dot{x}_{TRUE}$')
            axes[k01].set_ylabel(state_labels[k01])
            axes[k01].set_xlabel(r'Time [s]')
            axes[k01].grid(True, which='both')
    if num_trajectories_to_plot > 1:
        axes[-1][0].legend()
        axes[-1][1].legend()
    else:
        axes[-1].legend()
    fig.tight_layout()
    return fig

def plot_prediction_data(t: chex.Array,
                         x_true: chex.Array,
                         x_est: chex.Array,
                         x_est_std: chex.Array,
                         beta: chex.Array,
                         source: str = "",
                         state_labels: str = [r'$cos(\theta)$', r'$sin(\theta)$', r'$\omega$'],
                         ) -> plt.figure:
    """Either pass all states and values with three dimensions (num_traj, num_samples, num_states)
    OR pass all states and values with only two dimensions (num_traj*num_samples, num_states)"""
    
    state_dim = x_true.shape[-1]
    assert x_true.shape == x_est.shape == x_est_std.shape

    fig, axes = plt.subplots(state_dim, 1, figsize=(16,9))
    for k01 in range(state_dim):
        axes[k01].plot(t[:,0].reshape(-1), x_true[:,k01], label=r'${x}_{TRUE}$')
        axes[k01].plot(t[:,0].reshape(-1), x_est[:,k01], label=r'${x}_{%s}$'%(source))
        axes[k01].fill_between(t[:,0].reshape(-1),
                                (x_est[:,k01] - beta[k01] * x_est_std[:,k01]).reshape(-1),
                                (x_est[:,k01] + beta[k01] * x_est_std[:,k01]).reshape(-1),
                                label=r'$2\sigma$', alpha=0.3, color='blue')
        axes[k01].set_ylabel(state_labels[k01])
        axes[k01].set_xlabel(r'Time [s]')
        axes[k01].grid(True, which='both')
    plt.legend()
    fig.tight_layout()
    return fig


def plot_data(t: chex.Array,
              x: chex.Array,
              u: chex.Array = None,
              x_dot: chex.Array = None,
              title: str = '',
              state_labels: str = [r'$cos(\theta)$', r'$sin(\theta)$', r'$\omega$']) -> plt.figure:
    num_dim = x.ndim # If ndim = 3, there are different trajectories to plot
    if num_dim == 3:
        t1 = t[0,:,:]
        x1 = x[0,:,:]
        if u is not None:
            u1 = u[0,:,:]
        if x_dot is not None:
            x_dot1 = x_dot[0,:,:]
    state_dim = x.shape[-1]
    input_dim = u.shape[-1]
    if state_dim > 1 or input_dim > 1:
        fig = plt.figure(figsize=(10,8))
        axes = []
        for k01 in range(3):
            axes.append(plt.subplot2grid((3*input_dim,2),(k01,0), colspan=input_dim))
        for k01 in range(input_dim):
            axes.append(plt.subplot2grid((3*input_dim,2),(3*k01,1), rowspan=3))
        for k01 in range(state_dim):
            axes[0].plot(t1, x1[:,k01], label=state_labels[k01])
            axes[2].plot(t1, x_dot1[:,k01], label=r'$\dot{x}_{%s}$'%(str(k01)))
        axes[0].set_xlabel('Time')
        axes[0].set_ylabel('States')
        axes[0].set_title('One trajectory of the sampled data')
        axes[0].legend()
        axes[0].grid(True, which='both')
        for k01 in range(input_dim):
            axes[1].plot(t1, u1[:,k01], label=r'$u_{%s}$'%(str(k01)))
        axes[1].set_xlabel('Time')
        axes[1].set_ylabel('Inputs')
        axes[1].legend()
        axes[1].grid(True, which='both')
        axes[2].set_xlabel('Time')
        axes[2].set_ylabel('State Derivatives')
        axes[2].legend()
        axes[2].grid(True, which='both')
        for k01 in range(input_dim):
            for k02 in range(u.shape[0]):
                axes[3+k01].plot(t[k02,:,k01], u[k02,:,k01], label=r'$u_{%s, traj%s}$'%(str(k01),str(k02)))
            axes[3+k01].set_xlabel('Time')
            axes[3+k01].set_ylabel('Inputs')
            plt.legend()
            axes[3].set_title('Control inputs for all trajectories')

    fig.suptitle(title)
    return fig

def calc_derivative_RMSE(x_dot_true: chex.Array,
                         x_dot_est: chex.Array,
                         x_dot_est_std: chex.Array) -> float:
    pass