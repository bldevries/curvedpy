
import importlib.metadata
__version__ = importlib.metadata.version('curvedpy')

from curvedpy.geodesics.blackhole import BlackholeGeodesicIntegrator
from curvedpy.cameras.camera import RelativisticCamera
from curvedpy.utils.projections import projection