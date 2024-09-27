# Tutorials on Tensor Networks and Matrix product states

This is a set of tutorials used in various (Winter-/Summer-)Schools on Tensor Networks, e.g. from the [European Tensor Network](http://quantumtensor.eu/). Each school has it's on branch with the final notebooks used.

This "main" branch keeps a general set of notebooks that can be used as an introduction and that forms the basis for the tutorials in the various schools, and is also included into the main TeNPy documentation.

The tutorials are split into two parts as follows.

In the first part, the `exercise_1_*.ipynb` notebooks use a set of small "toy codes", small python scripts that require only [Python](https://python.org) with [numpy](https://numpy.org) + [scipy](https://scipy.org) + [matplotlib](https://matplotlib.org). They should give you a good idea how the algorithms work without the need to understand a full library like TeNPy. 
These python files for this are in the folder `tenpy_toycodes`, and you need to look into them during the tutorials to see how they work. It should not be necessary to modify the python files, however; you can define all functions etc in the notebooks itself. This will ensure that the other notebooks using them keep working.
The exercises itself are in interactive [jupyter notebooks](https://jupyter.org), where you can have code, output and plots together. 

The `exercise_tenpy.ipynb` notebook is for the second part, and uses the [TeNPy](https://github.com/tenpy/tenpy) library to demonstrate first examples of calling TeNPy rather than the toycodes.

**DISCLAIMER**: The toycodes and examples used are not optimized, and we only use very small bond dimensions here to make sure everything runs quickly on a normal laptop. For state-of-the-art MPS calculations (especially for cylinders towards 2D), `chi` should be significantly larger, often on the order of several 1000s (and significantly more CPU time).

## Some References

- [White, PRL 69, 2863 (1992)](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.69.2863), the original DMRG paper!
- [Hauschild, Pollmann, arXiv:1805.0055](https://arxiv.org/abs/1805.0055), review with focus on TeNPy use cases
- [Schollwoeck, arXiv:1008.3477](https://arxiv.org/abs/1008.3477), a classic review
- [Haegeman et al, arXiv:1103.0936](https://arxiv.org/abs/1103.0936), the original application of TDVP to MPS
- [Haegeman et al, arXiv:1408.5056](https://arxiv.org/abs/1408.5056), discussed in the TDVP lecture
- [Vanderstraeten et al, arXiv:1810.07006](https://arxiv.org/abs/1810.07006), a good review of the tangent space for infinite MPS
- [Paeckel et al, arXiv:1901.05824](https://arxiv.org/abs/1901.05824), a nice review comparing various MPS time evolution methods
- [More references in the TeNPy docs](https://tenpy.readthedocs.io/en/latest/literature.html)


## Setup

**Running locally**: If you have a working Python installation, feel free to solve all the exercises locally on your own computer. For the most part, you only need the `*.ipynb` notebooks and the `tenpy_toycodes/` folder.
For the second part, you need to [install TeNPy](https://tenpy.readthedocs.io/en/latest/INSTALL.html), which is often just a `conda install physics-tenpy` or `pip install physics-tenpy`, depending on your setup.

**Jupyter notebooks**: We recommend solving the exercises interactively with [jupyter notebooks](https//jupyter.org). You can get it with `conda install jupyterlab` or `pip install jupyterlab` and then run `jupyter-lab`, which opens an interactive coding session in your web browser.

**Running notebooks on Google colab**: You can also use [Google's colab cloud service](https://colab.research.google.com) to run the jupyter notebooks **without any local installation**. Use this option if you have any trouble with your local installation. However, you have to perform addiontal installs:
- For the first part, `exercise_1_*.ipynb`, you need to make sure that you not only copy the notebooks itself onto google colab, but also the `tenpy_toycodes/` folder (including the `__init__.py` file). 
  Alternatively, install them by adding and executing a notebook cell `!pip install git+https://github.com/tenpy/tenpy_toycodes.git` at the top of the notebooks.
- For the second part, `exercise_tenpy.ipynb`, you need to install TeNPy. On google colab, this can be done by adding and executing a notebook cell `!pip install physics-tenpy` at the top of the notebook.


## License

All codes are released under the Apache license given in the file `LICENSE`, which means you can freely copy, share and distribute the code.
They toycodes in the folder `tenpy_toycodes` have formerly been distributed directly from the [TeNPy](https://github.com/tenpy/tenpy) repository - originally under the GPL v3 license, see also [this issue on the license change](https://github.com/tenpy/tenpy/issues/462).
