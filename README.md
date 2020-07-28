# NLMpy

NLMpy is a Python package for the creation of neutral landscape models that
are widely used in the modelling of ecological patterns and processes across
landscapes.

NLMpy aims to provide several advantages over existing NLM software:

* it is open-source so it can be easily adapted or developed for specific modelling requirements.
* being cross-platform it can be used on any computer system.
* it brings together a much wider range of NLM algorithms, including some that are not available elsewhere.
* it can be combined with geographic information system (GIS) data.
* it enables novel combinations and integrations of different NLM algorithms.
* it can be embedded into larger modelling frameworks based on software that is capable of executing a Python script.  

## Example use

Using NLMpy to create and export a midpoint displacement NLM can be achieved with 
only three lines of code:

```python
from nlmpy import nlmpy
nlm = nlmpy.mpd(nRow=50, nCol=50, h=0.75)
nlmpy.exportASCIIGrid("raster.asc", nlm)
```

## New developments

Having forked the project into manaakiwhenua the following have been (or are being!) added:

### Done:
* additional NLMs; waveSurface, perlinNoise

### Doing:
* numba integration to leverage jit compliation where useful
* general coding improvements to speed slow functions up
* error check and handling

### To do:
* wiki pages containing more detailed examples