===========================
 Exploring Lagrangian Data
===========================

While the main aim of the `lagrangian-filtering` library is the
filtering of the Lagrangian-transformed data, it can be useful to work
with the transformed data in an exploratory capacity. Suppose we have
a new dataset on which we'd like to perform the filtering. Two choices
need to be made straight away: what is the high-pass cutoff frequency,
and what is a sensible advection timestep? Certainly prior experience
and an idea of the source model parameters like Coriolis parameter and
timestep are useful, however directly interrogating the data may lead
to better results.

Under the hood (see also the :doc:`algorithm <algorithm>` description), the
Lagrangian transformation can be performed in isolation with
:meth:`~filtering.filtering.LagrangeFilter.advection_step`. For example,
to compute the mean Lagrangian velocity, which could then be used to
compute a spectrum for determining an ideal cutoff frequency:

.. code-block:: python

   f = LagrangeFilter(...)
   data = f.advection_step(time)
   mean_u = np.mean(data["var_U"][1], axis=1)


Using analysis functions
========================

As an alternative to using ad-hoc explorations as above, there are
predefined functions available to give more robust and efficient ways
to interrogate your data. For example, a mean kinetic energy spectrum
over all particles could be computed at a specified time using
:meth:`~filtering.analysis.power_spectrum`:

.. code-block:: python

   from filtering import analysis
   f = LagrangeFilter(..., sample_variables["U", "V"], ...)
   spectra = analysis.power_spectrum(f, time)
   ke_spectrum = spectra["var_U"] + spectra["var_V"]


Eulerian filtering
==================

It may be useful to compare Lagrangian-filtered data to
Eulerian-filtered data, i.e. simply take the time series at each
point, and apply the usual highpass filtering. To make the most direct
comparison, this can be easily achieved within the same framework as
the Lagrangian filtering. After constructing the filtering object,
change the particle kernel to have only the sampling component. In
effect, this deletes the advection component of the kernel, leaving a
purely Eulerian filtering pipeline.

.. code-block:: python

   f = LagrangeFilter(...)
   f.kernel = f.sample_kernel
   ...


Obtaining particle trajectories
===============================

As a bit of a sanity check, we could verify that the particles are
taking sensible paths through our data. Usually, this is discarded,
because it takes up extra memory, and is not actually used in the
final result. We can ask for the position to be retained, so that we
can examine the advection data:

.. code-block:: python

   f = LagrangeFilter(...)
   f.sample_variables += ["lat", "lon"]
   data = f.advection_step(time)
   lat, lon = data["lat"][1], data["lon"][1]
