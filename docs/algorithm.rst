===========
 Algorithm
===========

The `lagrangian-filtering` algorithm has a fairly straightforward
goal: to take a set of observations and a velocity field on a
temporally-invariant grid, and remove low-frequency components from
those observations in a Lagrangian frame of reference. This operation
may be useful in many contexts, such as distinguishing stationary wave
signals from the sub-inertial flow on which they are imposed. In an
Eulerian frame of reference, these stationary waves would be
considered part of the mean flow and ignored!

The crux of the algorithm is, unsurprisingly, the transformation to a
Lagrangian frame of reference. Given the velocity field, we can
perform this transformation by simply advecting point particles along
the flow. Interpolating any observations of interest onto the particle
positions from the underlying gridded dataset gives us the discretised
approximation to the Lagrangian transformation. Consider that the
given velocity field may have regions of convergence and
divergence. It becomes important to ensure that in particular, regions
of divergent flow are spatially well-sampled.

We get around this problem of ensuring the entire domain is
well-sampled by a comprehensive set of particle advections: for each
time slice `t` of the source data, one particle is seeded at every
grid point. By running the particle advection both forwards and
backwards in time, a timeseries of the Lagrangian transformation is
obtained for each grid point. Considering this timeseries, we know
that by construction, the domain is well sampled at time `t`. Assuming
that the timeseries is sufficiently long to filter sub-inertial
frequencies, we can easily compute the Lagrangian-transformed,
high-pass filtered observation at time `t`. Note that we don't make
use of the filtered timeseries at any times preceeding or following
`t`. The reason for this is threefold:

1. we don't know whether these fields are spatially well-sampled
2. if they are well-sampled, we would have to run an unstructured
   interpolation algorithm to reconstruct the data on the source grid
3. due to the finite sampling window, these times are more likely to
   be impacted by ringing at the ends of the timeseries

Bearing in mind the aforementioned points, the forward-backward
particle advection must then be performed at every time slice `t`, to
obtain filtered data only at that slice. Indeed, the times at the
beginning and end of the source data may be skipped as well, since
there may not be enough data to filter the desired frequencies without
ringing.

Implementation
==============

The above description breaks down the algorithm at a high level. The
implementation follows quite naturally:

1. use OceanParcels_ to perform forward and backward particle advection
   at each time slice
2. construct a custom sampling kernel to record observations at particle
   locations during advection
3. high-pass filter the Lagrangian-transformed observations
4. save the resultant filtered fields to disk in a convenient format

The heavy lifting is done by OceanParcels, which combines the
expressive nature of Python with the performance of machine code by
compiling the advection and sampling kernels to a C library on the
fly, giving a vast speedup over native Python. To handle large
datasets, the `lagrangian-filtering` library uses a custom fork of
OceanParcels that uses a structure-of-arrays representation for
particle data. This allows for efficient, vectorised access to the
particle data on the Python side, which tends to bottleneck
performance. Advection is also parallelised using OpenMP, to take
advantage of systems with multiple CPU cores.

.. _OceanParcels: http://oceanparcels.org
