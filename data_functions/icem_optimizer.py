"""Generate colored noise. Taken from: https://github.com/felixpatzelt/colorednoise/blob/master/colorednoise.py"""
import chex
import numpy as np
from jax.numpy import sqrt, newaxis
from jax.numpy.fft import irfft, rfftfreq
import jax
import jax.numpy as jnp
from functools import partial
from typing import NamedTuple, Generic

@partial(jax.jit, static_argnums=(0, 1, 3))
def powerlaw_psd_gaussian(exponent: float, size: int, rng: jax.random.PRNGKey, fmin: float = 0) -> jax.Array:
    """Gaussian (1/f)**beta noise.
    Based on the algorithm in:
    Timmer, J. and Koenig, M.:
    On generating power law noise.
    Astron. Astrophys. 300, 707-710 (1995)
    Normalised to unit variance
    Parameters:
    -----------
    exponent : float
        The power-spectrum of the generated noise is proportional to
        S(f) = (1 / f)**beta
        flicker / pink noise:   exponent beta = 1
        brown noise:            exponent beta = 2
        Furthermore, the autocorrelation decays proportional to lag**-gamma
        with gamma = 1 - beta for 0 < beta < 1.
        There may be finite-size issues for beta close to one.
    shape : int or iterable
        The output has the given shape, and the desired power spectrum in
        the last coordinate. That is, the last dimension is taken as time,
        and all other components are independent.
    fmin : float, optional
        Low-frequency cutoff.
        Default: 0 corresponds to original paper.

        The power-spectrum below fmin is flat. fmin is defined relative
        to a unit sampling rate (see numpy's rfftfreq). For convenience,
        the passed value is mapped to max(fmin, 1/samples) internally
        since 1/samples is the lowest possible finite frequency in the
        sample. The largest possible value is fmin = 0.5, the Nyquist
        frequency. The output for this value is white noise.
    random_state :  int, numpy.integer, numpy.random.Generator, numpy.random.RandomState,
                    optional
        Optionally sets the state of NumPy's underlying random number generator.
        Integer-compatible values or None are passed to np.random.default_rng.
        np.random.RandomState or np.random.Generator are used directly.
        Default: None.
    Returns
    -------
    out : array
        The samples.
    Examples:
    ---------
    # generate 1/f noise == pink noise == flicker noise
    """

    # Make sure size is a list so we can iterate it and assign to it.
    try:
        size = list(size)
    except TypeError:
        size = [size]

    # The number of samples in each time series
    samples = size[-1]

    # Calculate Frequencies (we asume a sample rate of one)
    # Use fft functions for real output (-> hermitian spectrum)
    f = rfftfreq(samples)

    # Validate / normalise fmin
    if 0 <= fmin <= 0.5:
        fmin = max(fmin, 1. / samples)  # Low frequency cutoff
    else:
        raise ValueError("fmin must be chosen between 0 and 0.5.")

    # Build scaling factors for all frequencies
    s_scale = f
    ix = jnp.sum(s_scale < fmin)  # Index of the cutoff

    def cutoff(x, idx):
        x_idx = jax.lax.dynamic_slice(x, start_indices=(idx,), slice_sizes=(1,))
        y = jnp.ones_like(x) * x_idx
        indexes = jnp.arange(0, x.shape[0], step=1)
        first_idx = indexes < idx
        z = (1 - first_idx) * x + first_idx * y
        return z

    def no_cutoff(x, idx):
        return x

    s_scale = jax.lax.cond(
        jnp.logical_and(ix < len(s_scale), ix),
        cutoff,
        no_cutoff,
        s_scale,
        ix
    )
    s_scale = s_scale ** (-exponent / 2.)

    # Calculate theoretical output standard deviation from scaling
    w = s_scale[1:].copy()
    w = w.at[-1].set(w[-1] * (1 + (samples % 2)) / 2.)  # correct f = +-0.5
    sigma = 2 * sqrt(jnp.sum(w ** 2)) / samples

    # Adjust size to generate one Fourier component per frequency
    size[-1] = len(f)

    # Add empty dimension(s) to broadcast s_scale along last
    # dimension of generated random power + phase (below)
    dims_to_add = len(size) - 1
    s_scale = s_scale[(newaxis,) * dims_to_add + (Ellipsis,)]

    # prepare random number generator
    key_sr, key_si, rng = jax.random.split(rng, 3)
    sr = jax.random.normal(key=key_sr, shape=s_scale.shape) * s_scale
    si = jax.random.normal(key=key_si, shape=s_scale.shape) * s_scale

    # If the signal length is even, frequencies +/- 0.5 are equal
    # so the coefficient must be real.
    if not (samples % 2):
        si = si.at[..., -1].set(0)
        sr = sr.at[..., -1].set(sr[..., -1] * sqrt(2))  # Fix magnitude

    # Regardless of signal length, the DC component must be real
    si = si.at[..., 0].set(0)
    sr = sr.at[..., 0].set(sr[..., 0] * sqrt(2))  # Fix magnitude

    # Combine power + corrected phase to Fourier components
    s = sr + 1J * si

    # Transform to real time series & scale to unit variance
    y = irfft(s, n=samples, axis=-1) / sigma
    return y


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    exponent_list = [0.0, 0.25, 1.0, 2.0, 3.0, 5.0, 10.0]
    fig, axs = plt.subplots(len(exponent_list))
    fig.suptitle('Colored noise with varying correlation coefficients')
    for i, exponent in enumerate(exponent_list):
        rng = jax.random.PRNGKey(seed=0)
        size = 128
        samples = powerlaw_psd_gaussian(
            exponent=exponent,
            size=size,
            rng=rng,
        )
        x = np.arange(0, size)
        y = np.asarray(samples)
        axs[i].plot(x, y, label="coefficient " + str(exponent))
        axs[i].legend()
    plt.show()
