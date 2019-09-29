lace
====

[![version](https://img.shields.io/pypi/v/lace?style=flat-square)][pypi]
[![python versions](https://img.shields.io/pypi/pyversions/lace?style=flat-square)][pypi]
[![version](https://img.shields.io/pypi/l/lace?style=flat-square)][pypi]
[![build status](https://img.shields.io/circleci/project/github/lace/lace/master?style=flat-square)][circle]

Polygonal mesh library developed at Body Labs.

The library is under active maintenance, and the goals are compatible with that:

- Keep the library working in current versions of Python and other tools.
- Make bug fixes.
- Provide API stability and backward compatibility with the upstream version.
- Respond to community contributions.

The eventual goal is to perform a rewrite of the loader and core mesh
functionality with quad support as part of a ["lace-core" project][lacecore]
and consider how to handle the extensive mesh manipulation functions which
remain.

[circle]: https://circleci.com/gh/lace/lace
[pypi]: https://pypi.org/project/lace/
[lacecore]: https://github.com/lace/lacecore-sketches


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


Contribute
----------

- Issue Tracker: https://github.com/lace/lace/issues
- Source Code: https://github.com/lace/lace

Pull requests welcome!


Support
-------

If you are having issues, please let us know.


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


Similar projects
----------------

There is an unrelated permissively licensed mesh manipulation library called
[Trimesh][] which provides some similar functionality.

[trimesh]: https://github.com/mikedh/trimesh


License
-------

The project is licensed under the two-clause BSD license.

This project uses the RPly library to read and write PLY files, by Diego Nehab,
IMPA, distributed under the MIT License.
 * http://www.impa.br/~diego/software/rply
