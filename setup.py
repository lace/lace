# Conversion from Markdown to pypi's restructured text: https://coderwall.com/p/qawuyq -- Thanks James.
import os

try:
    import pypandoc
    long_description = pypandoc.convert('README.md', 'rst')
except (IOError, ImportError):
    long_description = ''
    print('warning: pandoc or pypandoc does not seem to be installed; using empty long_description')

import importlib

try:  # pip >= 10
    from pip._internal.req import parse_requirements
except ImportError:  # pip <= 9.0.3
    from pip.req import parse_requirements

from setuptools import setup, Extension
import numpy as np

install_requires = parse_requirements('requirements.txt', session=False)
install_requires = [str(ir.req) for ir in install_requires]

setup(
    name='lace',
    version=importlib.import_module('lace').__version__,
    author='Body Labs',
    author_email='alex@bodylabs.com',
    description='3d mesh handling',
    long_description=long_description,
    url='https://github.com/bodylabs/lace',
    license='MIT',
    packages=[
        'lace',
        'lace.serialization',
        'lace.serialization.obj',
        'lace.serialization.ply'
    ],
    ext_modules=[
        Extension('lace.serialization.ply.plyutils',
            sources=[os.path.join('lace/serialization/ply', x) for x in ['plyutils.c', 'rply.c']],
            depends=[os.path.join('lace/serialization/ply', x) for x in ['plyutils.c', 'plyutils.h', 'rply.c', 'rply.h', 'rplyfile.h']],
        ),
        Extension('lace.serialization.obj.objutils',
            sources=['lace/serialization/obj/objutils.cpp'],
            depends=['lace/serialization/obj/objutils.cpp'],
            include_dirs=[np.get_include()],
            extra_compile_args=['-O2', '-Wno-write-strings', '-Wno-c++11-narrowing', '-std=c++0x'], # maybe skip these on Windows?
        ),
    ],
    install_requires=install_requires,
    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
    ]
)
