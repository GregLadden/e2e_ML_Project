## Welcome to Machine Learning Housing Corp.

This repository provides a machine learning project focused on predicting median house values in Californian districts. The project utilizes various district features to build and evaluate predictive models for estimating housing prices.

## Prerequisites

Before you begin, make sure you have the necessary tools and dependencies installed. For detailed setup instructions, please refer to the [prerequisites guide](docs/prerequisites.md).

This project outputs results via the command line. If you prefer to run the application in Jupyter Notebooks, please follow the instructions [here](docs/jupyter_notebooks.md).

## Installation

Clone this repository using either HTTPS or SSH:

**HTTPS:**

```bash
git clone https://github.com/GregLadden/ml_housing_corp.git
```

**SSH:**

```bash
git clone https://github.com/GregLadden/ml_housing_corp.git
```

> **Note:** Before installation, it’s recommended to activate your virtual environment before installing the packages. Please refer to the [environment setup](#) for instructions on how to set up and activate it.

```bash
cd ml_housing_corp
pip install -r requirements.txt
```

## Documentation

**Running Make Commands**

To streamline and automate various tasks in this project, we use a `Makefile` that allows you to easily run predefined commands. The `Makefile` provides a set of targets that correspond to different functionalities, such as data visualization and correlation analysis. To execute a command, simply use the `make` command followed by the target name. For example, to generate a histogram of median house values, you would run:

```bash
make plot_histogram
```

## Acknowledgments

This project is based on chapter 2 in the book:

- **Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, 2nd Edition**  
  _Aurélien Géron_

We would like to thank Aurélien Géron for providing such a comprehensive resource that guided the development of this project.
