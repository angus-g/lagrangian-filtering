from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("lagrangian-filtering")
except PackageNotFoundError:
    pass

from filtering.filtering import LagrangeFilter
import filtering.analysis
import filtering.filter
