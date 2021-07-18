# Machine learning, but not understanding

This repository explores some of the concepts from the book "Artificial Intelligence, a guide for thinking humans", by Melaine Mitchell.

More specifically, it shows with some experiments that despite calling it "machine learning", the machines are not really "learning" in the sense that we, humans, use the term.

> "Learning in neural networks simply consists in gradually modifying the weights on connections so that each output’s error gets as close to 0 as possible on all training examples."

(Quoted text blocks, like the one just above, are from Mitchell's book).

An important consequence of this "learning" process:

> The machine learns what it observes in the data rather than what you (the human) might observe. If there are statistical associations in the training data, even if irrelevant to the task at hand, the machine will happily learn those instead of what you wanted it to learn.

The [Jupyter notebook in this repository](machine_learning_but_not_understanding.ipynb) demonstrates how neural networks can fail because they are not in fact "learning".

If you are interested only in the concepts, see this [blog post](https://cgarbin.github.io/machine-learning-but-not-understanding/).

## Exploring the concepts with code

To explore the concepts with code, configure the Python environment as described below, then open [this Jupyter notebook](machine_learning_but_not_understanding.ipynb).

## Setting up the Python environment

### Install Python 3

The project uses Python 3.

Verify that you have Python 3.x installed: `python --version` should print `Python 3.x.y`. If
it prints `Python 2.x.y`, try `python3 --version`. If that still doesn't work, please install
Python 3.x before proceeding. The official Python download site is
[here](https://www.python.org/downloads/).

From this point on, the instructions assume that **Python 3 is installed as `python3`**.

### Cloning the repository

```bash
git clone https://github.com/fau-masters-collected-works-cgarbin/machine-learning-but-not-understanding.git
```

The repository is now in the directory `machine-learning-but-not-understanding`.

### Creating a Python virtual environment

Execute these commands to create and activate a [virtual environment](https://docs.python.org/3/tutorial/venv.html) for the project:

```bash
#  switch to the directory where the cloned repository is
cd machine-learning-but-not-understanding

python3 -m venv env
source env/bin/activate
# or in Windows: env\Scripts\activate.bat
```

### Installing the dependencies

It's important to update `pip` first. Older `pip` versions fail to install Tensorflow.

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Running the code

`jupyter lab machine_learning_but_not_understanding.ipynb`
