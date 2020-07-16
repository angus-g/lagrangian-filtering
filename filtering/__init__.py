from pkg_resources import get_distribution, DistributionNotFound

try:
    __version__ = get_distribution("lagrangian-filtering").version
except DistributionNotFound:
    pass

from filtering.filtering import LagrangeFilter
import filtering.filter
