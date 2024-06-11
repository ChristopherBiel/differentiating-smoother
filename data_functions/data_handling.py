import jax.numpy as jnp
import jax.random as jr
from jax import vmap
from bsm.utils.normalization import Data

def split_dataset(data: Data) -> (list[Data], int):
        """Splits the full Dataset into the individual trajectories,
        based on the timestamps (every time there is a jump backwards in the timestamp the data is cut)
        The output is still only one dataset, but now with an additional dimension, which is the number of trajectories."""

        t = data.inputs
        assert t.shape[1] == 1

        # Split based on the distance between the timesteps (split at negative values)
        delta_t = jnp.diff(t, axis=0)
        indices = jnp.where(delta_t < 0.0)[0] + 1
        ts = jnp.split(t, indices)
        xs = jnp.split(data.outputs, indices)

        # To be able to stack, all the arrays need to have the same shape
        # For trajectories with different lengths (which may happen) this is not the case
        # Therefore, we need to pad by wrapping around (so as to not favor certain samples or add new ones)
        max_length = max([len(traj) for traj in ts])
        for i in range(len(ts)):
            ts[i] = jnp.pad(ts[i], ((0, max_length - len(ts[i])), (0, 0)), mode='wrap')
            xs[i] = jnp.pad(xs[i], ((0, max_length - len(xs[i])), (0, 0)), mode='wrap')
        
        inputs = jnp.stack(ts, axis=0)
        outputs = jnp.stack(xs, axis=0)

        return Data(inputs=inputs, outputs=outputs), len(ts)

def analyse_data(t, x, u, x_dot):
    """Check the sampled data for correct dynamics (Something seems to be wrong)"""
    
    state_dim = x.shape[-1]
    control_dim = u.shape[-1]

    # Reshape if only one trajectory is given.
    if x.ndim != 3:
        t.reshape(1, -1, 1)
        x.reshape(1, -1, state_dim)
        x_dot.reshape(1, -1, state_dim)
        u.reshape(1, -1, control_dim)
    
    print(f"Analysing data of the shape: {t.shape[0]} trajectories, {t.shape[1]} samples per traj., {state_dim} state dim, {control_dim} control dim")
    # Integrate the x_dot
    dt = (t[0,-1,0] - t[0,0,0]) / (t.shape[1]-1)
    print(f"dt = {dt}")
    def integrate(dot, dt, offset):
        return jnp.cumsum(dot*dt, axis=0) + offset
         
    x_integrated = vmap(integrate, in_axes=(0, None, 0))(x_dot, dt, x[:,0,:])
    
    def mse(a, b):
        return jnp.mean((a - b) ** 2, axis=0)
    state_mse = vmap(mse, in_axes=(0, 0))(x_integrated, x)
    print(f"Difference between integration and actual state: {state_mse}")
    return x_integrated


if __name__ == "__main__":
    from data_functions.data_creation import sample_random_pendulum_data
    import matplotlib.pyplot as plt

    sample_points = 32
    noise_level = None
    key = jr.PRNGKey(0)
    num_traj = 1
    t, x, u, x_dot = sample_random_pendulum_data(num_points=sample_points,
                                                 noise_level=noise_level,
                                                 key=key,
                                                 num_trajectories=num_traj,
                                                 initial_states=None,)
    print(f"Time data with {t.shape}: {t}")
    x_integ = analyse_data(t, x, u, x_dot)

    print(f"True state 1: {x[0,:,0]}")
    print(f"Integrated state 1: {x_integ[0,:,0]}")
    plt.plot(t.reshape(-1), x_integ[:,:,0].reshape(-1), label="Integrated state 1")
    plt.plot(t.reshape(-1), x_integ[:,:,1].reshape(-1), label="Integrated state 2")
    plt.plot(t.reshape(-1), x[:,:,0].reshape(-1), label="True state 1")
    plt.plot(t.reshape(-1), x[:,:,1].reshape(-1), label="True state 2")
    plt.title("Comparing the true state to an integration of the state derivative")
    plt.xlabel("Time [s]")
    plt.ylabel("States 1 and 2 [-]")
    plt.legend()
    plt.show()