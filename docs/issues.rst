===============
 Common issues
===============

Unfortunately, the complexity of supporting all sorts of model outputs
on different systems can sometimes lead to obscure errors. This page
may help to identify a common issue.

Index errors with a curvilinear grid
------------------------------------

Curvilinear grids, i.e. those with 2D lat/lon fields, are somewhat
more complex when it comes to the particle advection. If a particle
isn't in its correct cell, the underlying index search algorithm in
Parcels is linear in the number of cells from its present location to
its target location. When we initially seed the particles for
filtering, Parcels doesn't know where they should be on the grid, so
the index search starts in the corner. For a rectilinear grid this is
no problem: we calculate the X and Y indices directly. However, for a
curvilinear grid, this could mean thousands of steps for each
particle, which quickly becomes intractable.

To prevent Parcels from hopelessly marching along with curvilinear
index searches, we have limited the number of steps it can take
to 10. This should present no issue once a particle is moving
according to the prescribed velocity field, but it could be
insufficient for the initial condition. To help things along, we give
Parcels a hint about the initial indices for a given parcel, using a
KD tree, through the pykdtree_ library. Unfortunately, if this not
installed or otherwise unimportable, the initial index population will
silently fail, leading to index search errors during the initial
particle sampling. Luckily, the fix is simple enough: install the
library (potentially forcing a from-source install if the binary
distribution is incompatible with your system).

.. _pykdtree: https://github.com/storpipfugl/pykdtree
