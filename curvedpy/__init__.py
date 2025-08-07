
import importlib.metadata
__version__ = importlib.metadata.version('curvedpy')

from curvedpy.geodesics.blackhole import BlackholeGeodesicIntegrator
from curvedpy.geodesics.pointpoint import BlackholeGeodesicPointPointIntegrator
from curvedpy.cameras.camera import RelativisticCamera
from curvedpy.utils.projections import projection
from curvedpy.rendering.polygon.toy_vertex_shader import CurvedVertexShader