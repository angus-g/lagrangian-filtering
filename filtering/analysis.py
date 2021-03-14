"""Analysis routines for Lagrangian particle data.

These routines take an already-configured 'LagrangeFilter' and produce
diagnostic output.

"""

import dask.array as da


def power_spectrum(filter, time):
    """Compute the mean power spectrum over all particles at a given time.

    This routine gives the power spectrum (power spectral density) for
    each of the sampled variables within ''filter'', as a mean over
    all particles. It will run a single advection step at the specified time.

    Args:
        filter (filtering.LagrangeFilter): The pre-configured filter object
            to use for running the analysis.
        time (float): The time at which to perform the analysis.

    Returns:
        Dict[str, numpy.array]: A dictionary of power spectra for each of
            the sampled variables on the filter.

    """

    psds = {}
    advection_data = filter.advection_step(time, output_time=False)

    for v, a in advection_data.items():
        spectra = da.fft.fft(a[1].rechunk((-1, "auto")), axis=0)
        mean_spectrum = da.nanmean(da.absolute(spectra) ** 2, axis=1)
        psds[v] = mean_spectrum.compute()

    return psds
