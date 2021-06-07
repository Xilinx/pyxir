# Copyright 2020 Xilinx Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" Setup PyXIR package """

import os
import sys
import glob
import sysconfig
import unittest
import setuptools
import platform
import subprocess
import multiprocessing

from shutil import copyfile, copymode
from distutils.version import LooseVersion
from distutils.command.install_headers import install_headers
from distutils.command.build_py import build_py
from setuptools.command.install_lib import install_lib
from distutils.command.install_data import install_data
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext

from pathlib import Path

__version__ = '0.2.1'

FILE_DIR = os.path.dirname(os.path.abspath(__file__))

#################
# SYS ARGUMENTS #
#################


if '--debug' in sys.argv:
    debug = True
    sys.argv.remove('--debug')
else:
   	debug = False

# DPUCADX8G/DPUv1 build
if '--use_vai_rt' in sys.argv:
    use_vai_rt_dpucadx8g = True
    sys.argv.remove('--use_vai_rt')
elif '--use_vai_rt_dpucadx8g' in sys.argv:
    use_vai_rt_dpucadx8g = True
    sys.argv.remove('--use_vai_rt_dpucadx8g')
else:
    use_vai_rt_dpucadx8g = False

# DPUCZDX8G/DPUv2 build
if '--use_vai_rt_dpuczdx8g' in sys.argv:
    use_vai_rt_dpuczdx8g = True
    sys.argv.remove('--use_vai_rt_dpuczdx8g')
elif '--use_vai_rt_dpuv2' in sys.argv:
    use_vai_rt_dpuczdx8g = True
    sys.argv.remove('--use_vai_rt_dpuv2')
else:
    use_vai_rt_dpuczdx8g = False

if '--use_dpuczdx8g_vart' in sys.argv:
    use_dpuczdx8g_vart = True
    sys.argv.remove('--use_dpuczdx8g_vart')
else:
    use_dpuczdx8g_vart = False

# CLOUD DPU with VART runtime build
# '--use_vai_rt_dpucahx8h' is here for backward compatibility
if '--use_vai_rt_dpucahx8h' in sys.argv:
    use_vai_rt_dpucahx8h = True
    use_vart_cloud_dpu = False
    sys.argv.remove('--use_vai_rt_dpucahx8h')
elif '--use_vart_cloud_dpu' in sys.argv:
    use_vart_cloud_dpu = True
    use_vai_rt_dpucahx8h = False
    sys.argv.remove('--use_vart_cloud_dpu')  
else:
    use_vai_rt_dpucahx8h = False
    use_vart_cloud_dpu = False

###############
# STATIC DATA #
###############

package_data = glob.glob("include/pyxir/**/*.hpp", recursive=True)
headers = package_data
static_data = glob.glob("python/pyxir/**/*.json", recursive=True)
static_data.extend(glob.glob("python/pyxir/**/*.dcf", recursive=True))


# From PyBind11: https://github.com/pybind/pybind11/blob/master/setup.py
class InstallHeaders(install_headers):

    """ Use custom header installer because the default one flattens
        subdirectories"""

    def run(self):
        if not self.distribution.headers:
            return

        for header in self.distribution.headers:
            subdir = os.path.dirname(
                os.path.relpath(header, 'include/pyxir'))
            install_dir = os.path.join(self.install_dir, subdir)
            self.mkpath(install_dir)

            (out, _) = self.copy_file(header, install_dir)
            self.outfiles.append(out)


# From PyBind11: https://github.com/pybind/pybind11/blob/master/setup.py
# Install the headers inside the package as well
class BuildPy(build_py):

    def build_package_data(self):
        build_py.build_package_data(self)
        for header in package_data:
            target = os.path.join(self.build_lib, 'pyxir', header)
            self.mkpath(os.path.dirname(target))
            self.copy_file(header, target, preserve_mode=False)
        for static_file in static_data:
            # strip python
            target = os.path.join(self.build_lib, static_file[7:])
            self.copy_file(static_file, target, preserve_mode=False)


###################
# CMAKE EXTENSION #
###################

# Based on official Pybind11 example:
#   https://github.com/pybind/cmake_example/blob/master/setup.py
class CMakeExtension(Extension):
    def __init__(self, name, sourcedir='.'):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


# Based on official Pybind11 example:
#   https://github.com/pybind/cmake_example/blob/master/setup.py
class CMakeBuild(build_ext):

    def initialize_options(self):
        build_ext.initialize_options(self)
        self.debug = debug

    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following"
                               " extensions: " +
                               ", ".join(e.name for e in self.extensions))

        if platform.system() == "Windows":
            cmake_version = LooseVersion(re.search(r'version\s*([\d.]+)',
                                         out.decode()).group(1))
            if cmake_version < '3.3.0':
                raise RuntimeError("CMake >= 3.3.0 is required on Windows")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(
            self.get_ext_fullpath(ext.name)))
        # required for auto-detection of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                      '-DPYTHON_EXECUTABLE=' + sys.executable]

        if use_vai_rt_dpucadx8g:
            cmake_args.append('-DUSE_VAI_RT_DPUCADX8G=ON')
        if use_vai_rt_dpuczdx8g:
            cmake_args.append('-DUSE_VAI_RT_DPUCZDX8G=ON')
        if use_vai_rt_dpucahx8h:
            cmake_args.append('-DUSE_VAI_RT_DPUCAHX8H=ON')
        if use_vart_cloud_dpu:
            cmake_args.append('-DUSE_VART_CLOUD_DPU=ON')
        if use_dpuczdx8g_vart:
            cmake_args.append('-DUSE_DPUCZDX8G_VART=ON')
        if self.debug:
            cmake_args.append('-DDEBUG=ON')
            # cmake_args.append('-DCMAKE_VERBOSE_MAKEFILE:BOOL=ON')

        cfg = 'Debug' if self.debug else 'Release'
        print("Cfg: {}".format(cfg))
        build_args = ['--config', cfg]

        if platform.system() == "Windows":
            cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'
                           .format(cfg.upper(), extdir)]
            if sys.maxsize > 2**32:
                cmake_args += ['-A', 'x64']
            build_args += ['--', '/m']
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            build_args += ['--', '-j{}'.format(multiprocessing.cpu_count())]

        env = os.environ.copy()
        # env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'\
        #     .format(env.get('CXXFLAGS', ''), self.distribution.get_version())
        env['CXXFLAGS'] = '{}'.format(env.get('CXXFLAGS', ''))
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args,
                              cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args,
                              cwd=self.build_temp)

        print(self.build_lib)
        # Copy libpyxir.so to python/ next to python/pyxir for rapid
        #   prototyping
        lib_bin = os.path.join(self.build_lib, 'libpyxir.so')
        lib_dest_dir = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), 'python')
        self.copy_file(lib_bin, lib_dest_dir)
        # lib_bin = os.path.join(self.build_lib, 'libpyxir.so.' + str(__version__))
        # lib_dest_dir = os.path.join(os.path.dirname(
        #     os.path.abspath(__file__)), 'python')
        # self.copy_file(lib_bin, lib_dest_dir)

        # Create symlink
        # lib_symlink = lib_dest_dir + "/libpyxir.so"
        # if os.path.exists(lib_symlink):
        #     os.remove(lib_symlink)
        # os.symlink(lib_dest_dir + '/libpyxir.so.' + str(__version__), lib_symlink)

        # Copy *_test file to tests directory
        # test_bin = os.path.join(self.build_temp, 'tests/pyxir_test')
        # test_dest_dir = os.path.join(os.path.dirname(
        #     os.path.abspath(__file__)), 'tests', 'cpp')
        # self.copy_file(test_bin, test_dest_dir)
        print()  # Add empty line for nicer output

    # From https://www.benjack.io/2018/02/02/python-cpp-revisited.html
    def copy_file(self, src_file, dest_dir):
        '''
        Copy ``src_file`` to `dest_dir` directory, ensuring parent directory
        exists. Messages like `creating directory /path/to/package` and
        `copying directory /src/path/to/package -> path/to/package` are
        displayed on standard output. Adapted from scikit-build.
        '''
        # Create directory if needed
        if dest_dir != "" and not os.path.exists(dest_dir):
            print("creating directory {}".format(dest_dir))
            os.makedirs(dest_dir)

        # Copy file
        dest_file = os.path.join(dest_dir, os.path.basename(src_file))
        print("copying {} -> {}".format(src_file, dest_file))
        copyfile(src_file, dest_file)
        copymode(src_file, dest_file)


setuptools.setup(
    name="pyxir",
    version=__version__,
    author="Xilinx Inc",
    author_email="jornt@xilinx.com",
    description=open(os.path.join(FILE_DIR, 'README.md'), encoding='utf-8').read(),
    long_description=open(os.path.join(FILE_DIR, 'README.md'), encoding='utf-8').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Xilinx/pyxir",
    packages=setuptools.find_packages("python"),
    package_dir={"": "python"},
    include_package_data=True,
    # package_data={
    #     # If any package contains *.txt files, include them:
    #     "/proj/rdi/staff/jornt/pyxir/include": ["*.hpp"]
    # },
    # data_files=[('../include/pyxir', '.pyxir.hpp')],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        'numpy',
        'packaging',
        'pydot==1.4.1',
        'h5py>=2.8.0'],
    extra_require={'full': ['tensorflow>=1.12.0,<2']},
    # cmdclass={'build_ext': BuildExt},
    cmdclass={
        'install_headers': InstallHeaders,
        'build_py': BuildPy,
        'build_ext': CMakeBuild
    },
    # ext_modules=ext_modules,
    ext_modules=[CMakeExtension('pyxir')],
    # headers=['../include/pyxir/pyxir.hpp'],
    zip_safe=False
)
