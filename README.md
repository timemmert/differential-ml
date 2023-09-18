Tensorflow and Pytorch implementation of differential machine learning (https://arxiv.org/abs/2005.02347, by Brian Huge and Antoine Savine).

The implementation is still in progress.

## Installing this repo
### Pulling the latest version of DML

```bash
git clone git@github.com:tum-ai/differential-ml.git
cd differential-ml
```

### Setting up the virtual environment with Python 3.11.0

Assumes a working installation of [pyenv](https://github.com/pyenv/pyenv) and [poetry](https://github.com/python-poetry/poetry)

```bash
pyenv install 3.11.0
pyenv virtualenv 3.11.0 differential-ml
pyenv local differential-ml
```

### Installing dependencies

```bash
poetry install
```

