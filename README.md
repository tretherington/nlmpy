# NLMpy <img src="images/logo.png" align="right" width="200" />

`NLMpy` is a Python package for the creation of neutral landscape models that 
are widely used by landscape ecologists to model ecological patterns across 
landscapes.

`NLMpy` aims to:

- be open-source so it can be easily adapted or developed for specific modelling requirements.
- be cross-platform it can be used on any computer system.
- bring together a wide range of neutral landscape model algorithms.
- be easily integrated with geographic information system data.
- enable novel combinations and integrations of different neutral landscape model algorithms.

A full description of the package can be found in the accompnying 
[software paper](https://besjournals.onlinelibrary.wiley.com/doi/full/10.1111/2041-210X.12308).

## Quick examples

Using `NLMpy` to create and export a midpoint displacement NLM can be achieved with 
only three lines of code:

```python
from nlmpy import nlmpy
nlm = nlmpy.mpd(nRow=50, nCol=50, h=0.75)
nlmpy.exportASCIIGrid("raster.asc", nlm)
```

## Citation

If you use `NLMpy` in your research we would be very grateful if you could please cite the 
software using the following freely available software paper:

[Etherington TR, Holland EP, O'Sullivan D (2015) NLMpy: a Python software package for 
the creation of neutral landscape models within a general numerical framework. Methods in 
Ecology and Evolution 6:164-168](https://besjournals.onlinelibrary.wiley.com/doi/full/10.1111/2041-210X.12308)

## Installation

`NLMpy` is available on the [Python Package Index](https://pypi.python.org/pypi/nlmpy), so it can be installed using:

```
pip install nlmpy
```

If that does not work you could also simply move the `NLMpy.py` file to the same location 
on your computer as a Python script that wants to import `NLMpy`, then when those scripts are 
executed they will import all the `NLMpy` functions.  So while this approach doesn’t 
actually install `NLMpy` onto your computer, it does at least allow you to make use of the 
functionality of `NLMpy` within a neighbouring Python script.

## Package dependencies

- numpy
- scipy
- numba

## Community guidelines

We very much welcome input from others\! If you find a bug, need some
help, or can think of some extra functionality that would be useful,
please raise an
[issue](https://github.com/tretherington/nlmpy/issues). Better
still, please feel free to fork the project and raise a pull request if
you think and can fix a bug, clarify the documentation, or improve the
functionality yourself.
