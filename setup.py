# https://betterscientificsoftware.github.io/python-for-hpc/tutorials/python-pypi-packaging/
# https://packaging.python.org/en/latest/tutorials/packaging-projects/

from setuptools import setup, find_packages

VERSION = '0.0.3'
DESCRIPTION = 'General Relativistic Geodesic Integrator'
LONG_DESCRIPTION = 'Curvedpy can be used to calculate geodesics (trajectories) for different metrics and for both photons and massive particles.'

# Setting up
setup(
       # the name must match the folder name 'verysimplemodule'
        name="curvedpy", 
        version=VERSION,
        author="B.L. de Vries",
        author_email="<bldevries@protonmail.com>",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=["numpy>=2.0.2", "scipy>=1.13.1", "sympy>=1.13.3", "multiprocess>=0.70.17"], # add any additional packages that 
        license = " GPL-3.0",
        # needs to be installed along with your package. Eg: 'caer'
        url= "https://github.com/bldevries/curvedpy",
        keywords=['python', 'relativistic', 'ray', 'tracer', 'blackhole', 'astronomy', 'physics', 'numerical'],
        classifiers= [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Education",
            "Programming Language :: Python :: 3",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
        ]
)