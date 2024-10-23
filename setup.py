from setuptools import setup, find_packages
from Cython.Build import cythonize
import os


def read(rel_path: str) -> str:
    here = os.path.abspath(os.path.dirname(__file__))
    # intentionally *not* adding an encoding option to open, See:
    #   https://github.com/pypa/virtualenv/issues/201#issuecomment-3145690
    with open(os.path.join(here, rel_path)) as fp:
        return fp.read()


def get_version(rel_path: str) -> str:
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            # __version__ = "0.9"
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    raise RuntimeError("Unable to find version string.")


# read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='vis_nav_game',
    version=get_version("vis_nav_core.py"),
    author="AI4CE Lab",
    author_email="cfeng@nyu.edu",
    description="a simple visual navigation game platform for embodied AI education",
    keywords='embodied AI navigation robotics education',
    url="https://github.com/ai4ce/vis_nav_game",
    packages=find_packages(),
    license='MIT',
    ext_modules=cythonize(['vis_nav_game/core.pyx'], language_level="3"),
    python_requires='>3.10',
    install_requires=[
        'pybullet',  # note: pybullet should be installed from conda-forge
        'numpy~=1.25.2',
        'cython~=3.0.2',
        'matplotlib>=3.5.3',
        'cryptography~=41.0.3',
        'opencv-python~=4.8.0.76',
        'gdown==4.7.3',
        'ntplib~=0.4.0'
    ],
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Beta',

        # Indicate who your project is intended for
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Topic :: Education',
        'Topic :: Games/Entertainment :: Simulation',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ]
)
