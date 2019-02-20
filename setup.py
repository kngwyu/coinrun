import os
import pkgconfig
from setuptools import find_packages, setup, Extension


CXX_FLAGS = '-std=c++11 -Wall -Wno-unused-variable -Wno-unused-function'
CXX_FLAGS += ' -Wno-deprecated-register -fPIC'

QT5 = pkgconfig.parse('Qt5Widgets')
os.environ['QT_SELECT'] = '5'


extension_module = Extension(
    'coinrun.cpplib',
    sources=['coinrun/coinrun.cpp'],
    libraries=QT5['libraries'],
    include_dirs=['/usr/include'] + QT5['include_dirs'],
    extra_compile_args=CXX_FLAGS.split(),
)

setup(
    name='coinrun',
    version='0.01',
    description='This is a demo package with a compiled C extension.',
    ext_modules=[extension_module],
    packages=find_packages(),
)
