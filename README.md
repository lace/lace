lace
====

[![version](https://img.shields.io/pypi/v/lace?style=flat-square)][pypi]
[![python versions](https://img.shields.io/pypi/pyversions/lace?style=flat-square)][pypi]
[![version](https://img.shields.io/pypi/l/lace?style=flat-square)][pypi]

Polygonal mesh library developed at Body Labs. **This library is deprecated.**

**There are five successor libraries:**

* **[lacecore][]** provides production-ready polygonal meshes optimized for cloud
  computation.
    * Supports triangles and quads.
    * Provides OBJ loading via the `obj` extra.
* **[polliwog][]** provides low-level, production-ready functions for working with
  triangles.
* **[entente][]** provides functions for working with vertexwise correspondence.
* **[proximity][]** provides proximity queries.
* **[hobart][]** obtains planar cross sections.
* **[tri-again][]** provides simple 3D scenegraphs for debugging 3D meshes,
  polylines, and points.
* **[meshlab-pickedpoints][]** loads and saves MeshLab picked point (.pp) files.

As an altenative for batteries-included prototyping, **[Trimesh][]** is
recommended.


[pypi]: https://pypi.org/project/lace/
[lacecore]: https://github.com/lace/lacecore
[tinyobjloader]: https://github.com/tinyobjloader/tinyobjloader
[entente]: https://github.com/lace/entente/
[hobart]: https://github.com/lace/hobart
[meshlab-pickedpoints]: https://github.com/lace/meshlab-pickedpoints
[proximity]: https://github.com/lace/proximity
[trimesh]: https://trimsh.org/
[tri-again]: https://github.com/lace/tri-again/
[polliwog]: https://github.com/lace/polliwog/


Installation
------------

### Install dependencies

Mac OS:
```sh
brew update && brew install boost
pip install numpy==1.13.1
pip install lace
```

Linux:
```sh
apt-get install -y --no-install-recommends libsuitesparse-dev libboost-dev
pip install numpy==1.13.1
pip install lace
```

Docker:
```
docker build .
```

### Install the library

```sh
pip install lace
```


Development
-----------

```sh
pip install -r requirements_dev.txt
pip install -e .
rake test
rake lint
```


Acknowledgements
----------------

This library was refactored from legacy code at Body Labs by [Alex Weiss][],
with portions by [Eric Rachlin][], [Paul Melnikow][], [Victor Alvarez][],
and others. It was extracted from the Body Labs codebase and open-sourced by
[Guillaume Marceau][]. In 2018 it was [forked by Paul Melnikow][fork] and
published as [metabolace][fork pypi]. Thanks to a repository and package
transfer from Body Labs, the fork has been merged back into the original.

[alex weiss]: https://github.com/algrs
[eric rachlin]: https://github.com/eerac
[paul melnikow]: https://github.com/paulmelnikow
[victor alvarez]: https://github.com/yangmillstheory
[guillaume marceau]: https://github.com/gmarceau
[fork]: https://github.com/metabolize/lace
[fork pypi]: https://pypi.org/project/metabolace/


License
-------

The project is licensed under the two-clause BSD license.

This project uses the RPly library to read and write PLY files, by Diego Nehab,
IMPA, distributed under the MIT License.
 * http://www.impa.br/~diego/software/rply
