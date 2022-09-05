# Tutorials on Tensor Networks and Matrix product states

This is a set of tutorials used in various (Winter-/Summer-)Schools on Tensor Networks, e.g. from the [European Tensor Network](http://quantumtensor.pks.mpg.de/).

The tutorials are split into two parts.

In the first part, we will use very small "toy codes" that require only [Python](https://python.org) with [numpy](https://numpy.org) + [scipy](https://scipy.org) + [matplotlib](https://matplotlib.org), and should give you a good idea how the algorithms work.
All files for this are in the folder `toycodes`, and you need to look into them during the tutorials to see how they work. (It should not be necessary to modify them.)

In the second part, we will use the [TeNPy](https://github.com/tenpy/tenpy) library to setup more advanced calculations in the folder `tenpy`.

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

**Running locally**: If you have a working Python installation, feel free to solve all the exercises locally on your own computer.
For the second part, you need to [install TeNPy](https://tenpy.readthedocs.io/en/latest/INSTALL.html), which is often just a `conda install physics-tenpy` or `pip install physics-tenpy`, depending on your setup.

**Jupyter notebooks**: We recommend solving the exercises interactively with [jupyter notebooks](https//jupyter.org). You can get it with ``conda install jupyterlab`` or ``pip install jupyterlab`` and then run``jupyter-lab``, which opens an interactive coding session in your web browser.

**Running notebooks on Google colab**: You can also use [Google's colab cloud service](https://colab.research.google.com) to run the jupyter notebooks **without any local installation**. Use this option if you have any trouble with your local installation.
In this case, you need to ``pip install git+https://github.com/tenpy/tenpy_toycodes.git`` to allow the notebooks to find the toy codes.
(It's already as a comment at the top of the notebooks.)

## License

All codes are released under GPL (v3) given in the file `LICENSE`, which means you can freely copy, share and distribute the code.
They toycodes in the folder `toycodes` are based on the toycodes distributed with TeNPy (also under the GPL).
