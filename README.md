# 📊 Python Data Science Snippets

[![Downloads](https://img.shields.io/packagecontrol/dt/Python%20Data%20Science%20Snippets)](https://packagecontrol.io/packages/Python%20Data%20Science%20Snippets)
[![Tag](https://img.shields.io/github/v/tag/futureprogrammer360/Python-Data-Science-Snippets?sort=semver)](https://github.com/futureprogrammer360/Python-Data-Science-Snippets/tags)
[![Repo size](https://img.shields.io/github/repo-size/futureprogrammer360/Python-Data-Science-Snippets)](https://github.com/futureprogrammer360/Python-Data-Science-Snippets)
[![License](https://img.shields.io/github/license/futureprogrammer360/Python-Data-Science-Snippets?style=flat-square)](https://github.com/futureprogrammer360/Python-Data-Science-Snippets/blob/master/LICENSE)

[Python Data Science Snippets](https://github.com/futureprogrammer360/Python-Data-Science-Snippets) is a collection of [Sublime Text](https://www.sublimetext.com/) snippets for data science and machine learning in Python.

## 💻 Installation

The easiest way to install Python Data Science Snippets is through [Package Control](https://packagecontrol.io/packages/Python%20Data%20Science%20Snippets). After it is enabled inside Sublime Text, open the command palette and find **Package Control: Install Package** and press `ENTER`. Then, find **Python Data Science Snippets** in the list. Press `ENTER` again, and this package is installed!

## 📈 Snippets

* [Imports](#imports)
* [NumPy](#numpy)
* [Pandas](#pandas)
* [Matplotlib](#matplotlib)
* [Scikit-learn](#scikit-learn)
* [Keras](#keras)
* [PyTorch](#pytorch)

### Imports

Import snippets start with `i` followed by the package/module's import alias.

| Trigger    | Description                               |
|------------|-------------------------------------------|
| `ikeras`   | `from tensorflow import keras`            |
| `inp`      | `import numpy as np`                      |
| `ipd`      | `import pandas as pd`                     |
| `iplt`     | `import matplotlib.pyplot as plt`         |
| `isklearn` | `from sklearn.$1 import $2`               |
| `isns`     | `import seaborn as sns`                   |
| `itf`      | `import tensorflow as tf`                 |
| `itorch`   | `import torch`                            |
| `inn`      | `from torch import nn`                    |
| `idl`      | `from torch.utils.data import DataLoader` |

### NumPy

| Trigger    | Description    |
|------------|----------------|
| `arange`   | `np.arange`    |
| `array`    | `np.array`     |
| `linspace` | `np.linspace`  |
| `logspace` | `np.logspace`  |
| `ones`     | `np.ones`      |
| `zeros`    | `np.zeros`     |

### Pandas

| Trigger       | Description      |
|---------------|----------------  |
| `apply`       | `df.apply`       |
| `columns`     | `df.columns`     |
| `describe`    | `df.describe`    |
| `df`          | `pd.DataFrame`   |
| `dropna`      | `df.dropna`      |
| `fillna`      | `df.fillna`      |
| `groupby`     | `df.groupby`     |
| `head`        | `df.head`        |
| `read_csv`    | `pd.read_csv`    |
| `rename`      | `df.rename`      |
| `reset_index` | `df.reset_index` |
| `sample`      | `df.sample`      |
| `ser`         | `pd.Series`      |
| `tail`        | `df.tail`        |
| `to_csv`      | `df.to_csv`      |
| `to_datetime` | `pd.to_datetime` |

### Matplotlib

| Trigger        | Description        |
|----------------|--------------------|
| `annotate`     | `plt.annotate`     |
| `bar_label`    | `plt.bar_label`    |
| `bar`          | `plt.bar`          |
| `barh`         | `plt.barh`         |
| `fill_between` | `plt.fill_between` |
| `hist`         | `plt.hist`         |
| `imread`       | `plt.imread`       |
| `imsave`       | `plt.imsave`       |
| `imshow`       | `plt.imshow`       |
| `legend`       | `plt.legend`       |
| `pie`          | `plt.pie`          |
| `plot`         | `plt.plot`         |
| `savefig`      | `plt.savefig`      |
| `scatter`      | `plt.scatter`      |
| `show`         | `plt.show`         |
| `stackplot`    | `plt.stackplot`    |
| `subplot`      | `plt.subplot`      |
| `subplots`     | `plt.subplots`     |
| `suptitle`     | `plt.suptitle`     |
| `text`         | `plt.text`         |
| `tight_layout` | `plt.tight_layout` |
| `title`        | `plt.title`        |
| `xlabel`       | `plt.xlabel`       |
| `xlim`         | `plt.xlim`         |
| `ylabel`       | `plt.ylabel`       |
| `ylim`         | `plt.ylim`         |

### Scikit-learn

| Trigger  | Description              |
|----------|--------------------------|
| `knn`    | `KNeighborsClassifier`   |
| `linreg` | `LinearRegression`       |
| `logreg` | `LogisticRegression`     |
| `rfc`    | `RandomForestClassifier` |
| `tts`    | `train_test_split`       |

### Keras

| Trigger      | Description               |
|--------------|---------------------------|
| `compile`    | `model.compile`           |
| `evaluate`   | `model.evaluate`          |
| `fit`        | `model.fit`               |
| `layer`      | `keras.layers.layer`      |
| `load_model` | `keras.models.load_model` |
| `predict`    | `model.predict`           |
| `save`       | `model.save`              |
| `sequential` | `keras.Sequential`        |

### PyTorch

| Trigger      | Description                   |
|--------------|-------------------------------|
| `dataloader` | `torch.utils.data.DataLoader` |
| `device`     | `torch.device (cuda/cpu)`     |
| `module`     | `torch.nn.Module`             |

The snippet files are in the [`snippets`](https://github.com/futureprogrammer360/Python-Data-Science-Snippets/tree/master/snippets) folder of [this GitHub repository](https://github.com/futureprogrammer360/Python-Data-Science-Snippets).
