import jax.numpy as jnp
from jax import vmap

def fitMultiStatePolynomial(t, x, regtype, lambda_, init_offset=None):
    """Fit Tikhonov regularization to multiple state data
    t: (m, 1) - m different samples of data
    x: (m, d) - d different states
    regtype: 'none', 'first', 'second'
    lambda_: Regularization parameter
    init_offset: (d)
    """
    # Use vmap to calculate for single states
    print(f"Input shapes: {t.shape} - {x.shape} - {init_offset.shape}")
    if init_offset is None:
        init_offset = jnp.zeros((x.shape[1],))
    v_apply = vmap(fitSingleStateTikhonov, in_axes=(None, 1, None, None, 0), out_axes=(1))
    x_fit, x_dot_fit = v_apply(t, x, regtype, lambda_, init_offset)
    return x_fit, x_dot_fit

def fitSinglePolynomial(t, x, regtype, lambda_, init_offset=0):
    """Fit a polynomial to the data with regularisation
    t: (m, 1) - m different samples of data
    x: (m, 1)
    regtype: 'zero', 'first', 'second'
    lambda_: Regularization parameter
    init_offset: true function value at the lower bound
    """
    n = t.shape[0]
    x = x - init_offset
    dt = (t[-1] - t[0]) / (n-1)

    # Calculate the derivative by solving the least squares problem
    A = jnp.tri(n, n, -1) * dt
    if regtype == 'zero':
        D = jnp.eye(n)
    elif regtype == 'first':
        D1 = jnp.zeros((n - 1, n))# Set the right hand diagonal to 1
        D1 = D1.at[jnp.arange(n-1), jnp.arange(n-1)].set(-1)
        D1 = D1.at[jnp.arange(n-1), jnp.arange(1, n)].set(1)
        D = jnp.vstack((jnp.eye(n), D1))
    elif regtype == 'second':
        D1 = jnp.zeros((n - 1, n))
        D1 = D1.at[jnp.arange(n-1), jnp.arange(n-1)].set(-1)
        D1 = D1.at[jnp.arange(n-1), jnp.arange(1, n)].set(1)
        D2 = jnp.zeros((n - 2, n))
        D2 = D2.at[jnp.arange(n-2), jnp.arange(n-2)].set(1)
        D2 = D2.at[jnp.arange(n-2), jnp.arange(1, n-1)].set(-2)
        D2 = D2.at[jnp.arange(n-2), jnp.arange(2, n)].set(1)
        D = jnp.vstack((jnp.eye(n), D1, D2))
    else:
        raise ValueError('Unknown regtype')
    
    x_dot_fit = jnp.linalg.solve(A.T @ A + lambda_ * D.T @ D, jnp.dot(A.T, x))

    # Calculate the integral by summing up the derivatives
    x_fit = jnp.cumsum(x_dot_fit*dt)
    x_fit += init_offset

    return x_fit, x_dot_fit