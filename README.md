metabolace
==========

[![pip install](https://img.shields.io/badge/pip%20install-metabolace-f441b8.svg?style=flat-square)][pypi]
[![version](https://img.shields.io/pypi/v/metabolace.svg?style=flat-square)][pypi]
[![python versions](https://img.shields.io/pypi/pyversions/metabolace.svg?style=flat-square)][pypi]
[![build status](https://img.shields.io/circleci/project/github/metabolize/lace/master.svg?style=flat-square)][circle]
[![last commit](https://img.shields.io/github/last-commit/metabolize/lace.svg?style=flat-square)][commits]
[![open pull requests](https://img.shields.io/github/issues-pr/metabolize/lace.svg?style=flat-square)][pull requests]

This is an active fork of [lace][upstream], the Body Labs-developed polygonal
mesh library.

The fork's goals are ambitious:

- Keep the library working in current versions of Python and other tools.
- Make bug fixes.
- Provide API stability and backward compatibility with the upstream version.
- Expand functionality to fully support quad meshes, and additional
  analysis and manipulation functionality where appropriate.
- Respond to community contributions.

[upstream]: https://github.com/bodylabs/lace
[circle]: https://circleci.com/gh/metabolize/lace
[pypi]: https://pypi.org/project/metabolace/
[pull requests]: https://github.com/metabolize/lace/pulls
[commits]: https://github.com/metabolize/lace/commits/master


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
pip install metabolace
```

And import it just like the upstream library:

```py
from lace.mesh import Mesh
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

- Issue Tracker: https://github.com/metabolize/lace/issues
- Source Code: https://github.com/metabolize/lace

Pull requests welcome!


Support
-------

If you are having issues, please let us know.


Acknowledgements
----------------

This library was refactored from legacy code at Body Labs by [Alex Weiss][],
with portions by [Eric Rachlin][], [Paul Melnikow][], [Victor Alvarez][],
and others. It was extracted from the Body Labs codebase and open-sourced by
[Guillaume Marceau][].

[alex weiss]: https://github.com/algrs
[eric rachlin]: https://github.com/eerac
[paul melnikow]: https://github.com/paulmelnikow
[victor alvarez]: https://github.com/yangmillstheory
[guillaume marceau]: https://github.com/gmarceau


License
-------

The project is licensed under the two-clause BSD license.

This project uses the RPly library to read and write PLY files, by Diego Nehab,
IMPA, distributed under the MIT License.
 * http://www.impa.br/~diego/software/rply
