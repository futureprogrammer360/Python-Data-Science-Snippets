# 📊 Python Data Science Snippets

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

Import snippets start with `i` followed by the package's import alias.

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

| Trigger    | Description    |
|------------|----------------|
| `columns`  | `df.columns`   |
| `describe` | `df.describe`  |
| `df`       | `pd.DataFrame` |
| `head`     | `df.head`      |
| `read_csv` | `pd.read_csv`  |
| `ser`      | `pd.Series`    |
| `tail`     | `df.tail`      |

### Matplotlib

| Trigger    | Description    |
|------------|----------------|
| `bar`      | `plt.bar`      |
| `legend`   | `plt.legend`   |
| `pie`      | `plt.pie`      |
| `plot`     | `plt.plot`     |
| `scatter`  | `plt.scatter`  |
| `show`     | `plt.show`     |
| `subplots` | `plt.subplots` |
| `title`    | `plt.title`    |
| `xlabel`   | `plt.xlabel`   |
| `ylabel`   | `plt.ylabel`   |

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
| `fit`        | `model.fit`               |
| `layer`      | `keras.layers.layer`      |
| `load_model` | `keras.models.load_model` |
| `save`       | `model.save`              |
| `sequential` | `keras.Sequential`        |

### PyTorch

| Trigger      | Description                   |
|--------------|-------------------------------|
| `dataloader` | `torch.utils.data.DataLoader` |
| `device`     | `torch.device (cuda/cpu)`     |
| `module`     | `torch.nn.Module`             |

The snippet files are in the `snippets` folder of [this GitHub repository](https://github.com/futureprogrammer360/Python-Data-Science-Snippets).
