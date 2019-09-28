import os
import importlib

with open("README.md") as f:
    readme = f.read()

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
    description='Polygonal mesh library developed at Body Labs',
    long_description=readme,
    long_description_content_type="text/markdown",
    url='https://github.com/lace/lace',
    license='BSD-2-Clause',
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
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
    ]
)
