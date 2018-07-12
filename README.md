lace
===========

A mesh class loaded with useful geometric and analysis functionality.


Installation
------------

Mac OS:
```sh
brew update && brew install boost
pip install pip==9.0.1 numpy==1.13.1
pip install lace
```

Linux:
```sh
apt-get install -y --no-install-recommends libsuitesparse-dev libboost-dev
pip install pip==9.0.1 numpy==1.13.1
pip install lace
```

Docker:
```
docker build .
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

- Issue Tracker: github.com/bodylabs/example/issues
- Source Code: github.com/bodylabs/example

Pull requests welcome!


Support
-------

If you are having issues, please let us know.


License
-------

The project is licensed under the two-clause BSD license.

This project uses the RPly library to read and write PLY files, by Diego Nehab,
IMPA, distributed under the MIT License.
 * http://www.impa.br/~diego/software/rply
