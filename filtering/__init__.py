from pkg_resources import get_distribution, DistributionNotFound

try:
    __version__ = get_distribution("lagrangian-filtering").version
except DistributionNotFound:
    pass

from .filtering import LagrangeFilter
from . import filter
