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
