===================
 The Filter Object
===================

Aside from the parameters we pass to the
:py:class:`~filtering.filtering.LagrangeFilter` initialiser
controlling the particle advection portion of the algorithm (described
in :doc:`data`), there are also some parameters to do with filtering
itself.


Times and Windows
=================

To accurately capture enough data to filter the low-frequency
components of the signal, there is the concept of the *window*. This
is the length of time on either side of a sample over which particles
are advected. This window is allowed to optionally be *uneven*, which
allows filtering to be performed, even if there isn't a full
timeseries on either side of a sample. This happens at the edges of
data availability, or at in individual particle level when particles
run into topography or out of the domain. The threshold for including
particles without a full trajectory is determined by the
``minimum_window`` parameter.

Control over the temporal resolution within the filtering window is
given by the ``advection_dt`` parameter. This defaults to a 5-minute
advection timestep, but this may need to be adjusted depending on the
spatial and temporal resolution of the input data. Ideally, this
parameter is set to some divisor of the input data timestep, allowing
for well-resolved particle paths.



Filter types
============

There are a couple of types of filters available for attenuating the
low-frequency component of particle trajectories. The default filter
is used when ``highpass_frequency`` is specified, which gives a 3dB
cutoff over the entire domain using a 4th-order Butterworth filter (as
obtained by :py:func:`scipy.signal.butter`).

There are other filters available in the :py:mod:`filtering.filter`
module, such as one that performs the filtering in frequency space
(may give a sharper cutoff, at the expense of possible ringing), or
that allows variation of the cutoff frequency over the domain.

If an alternate filter is constructed, it can be attached to the
:py:class:`~filtering.filtering.LagrangeFilter`::

    ff = filtering.LagrangeFilter(...)
    f = filtering.filter.Filter(...)
    ff.inertial_filter(f)
