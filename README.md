NLMpy
=====

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

Example use
===========

Using NLMpy to create and export a midpoint displacement NLM can be achieved with 
only three lines of code:

.. code:: python

    >>> import nlmpy
    >>> nlm = nlmpy.mpd(nRow=50, nCol=50, h=0.75)
    >>> nlmpy.exportASCIIGrid("raster.asc", nlm)

Python 3.x example use
======================

Due to differences in absolute and relative importing between Python versions 2.x and 3.x NLMpy needs to be imported slightly differently in version 3.x:

.. code:: python

    >>> from nlmpy import nlmpy
    >>> nlm = nlmpy.mpd(nRow=50, nCol=50, h=0.75)
    >>> nlmpy.exportASCIIGrid("raster.asc", nlm)
	
Please note that when executing the example scripts from the supplementary material provided with the published paper, if you are running Python 3.x you will need to modify the way NLMpy is imported by the script.

Otherwise NLMPy should work fine in a 3.x environment and has been successfully tested with Python 3.5.2, Numpy 1.11.3, SciPy 0.18.1, Matplotlib 2.0.0, and NLMpy 0.1.4.

Documentation
=============


A full description of NLMpy is published in: Etherington TR, Holland EP, and 
O'Sullivan D (2015) NLMpy: a Python software package for the creation of 
neutral landscape models within a general numerical framework. Methods in 
Ecology and Evolution 6(2):164-168 , which is freely available online  
(http://bit.ly/14i4x7n).  

The journal website also holds example scripts and GIS data
(http://bit.ly/1XUXjOF) that generate the figures in the paper.  There are 
also some tutorial videos that provide some advice about installing 
(http://bit.ly/1qLfMjt) and using (http://bit.ly/2491u9n) NLMpy.


Dependencies
============

NLMpy was developed using:

* Python 2.7.3
* NumPy 1.8.0
* SciPy 0.13.2
* Matplotlib 1.3.1

Version History
===============

* 0.1.5 - added some advice to the read me about imprting NLMpy with Python 3.x
* 0.1.4 - updated code to add compatability with Python 3.x
* 0.1.3 - updated documentation
* 0.1.2 - updated documentation
* 0.1.1 - added the exportASCIIGrid function

Installation notes
==================

Getting a properly working installation of Python and associated packages can be a rather 
unintuitive task – especially if you are not overly familiar with software distribution 
methods.  It is possible to build a Python installation by downloading Python itself 
(https://www.python.org/downloads/) and then adding required Python packages from the 
Python Package Index (https://pypi.python.org/pypi).  However, a much simpler approach 
that I would recommend is using a scientific distribution of Python that comes with all 
the packages you are most likely to require for scientific applications.

Perhaps the two most popular scientific distributions of Python are the Anaconda 
distribution (https://store.continuum.io/cshop/anaconda/) and the Canopy distribution 
(https://www.enthought.com/products/canopy/).  I would recommend either, as they are 
both: cross-platform, free to download, and come with all the Python packages most 
commonly required for scientific computing.  My personal preference is for the Anaconda 
distribution, as it comes with Spyder (https://pythonhosted.org/spyder/) that is in my 
opinion the best environment for writing Python code (it is similar to Rstudio for people 
more used to using R), and because in Canopy access to some of the Python packages I use 
a lot requires additional licencing.  But in order to get NLMpy working either Anaconda or 
Canopy will work, as they both come with the NumPy (http://www.numpy.org/) and SciPy 
(http://www.scipy.org/) packages on which NLMpy depends, the Matplotlib 
(http://matplotlib.org/) package that the example scripts use to plot results, and the pip 
(https://pypi.python.org/pypi/pip) package that makes installation of NLMpy easy.

Once you have installed the Python version 2.7.x of either Anaconda or Canopy, the 
installation of NLMpy can be done from the command line using either a Terminal on 
OSX/UNIX or a Command Prompt on Windows.  At the command line you just need to run::

    pip install nlmpy

which will get the pip program to install NLMpy directly from the online Python Package 
Index.

You may have a problem using this approach if you connect to the Internet via a proxy 
server.  So while this approach works fine for me at home, I can’t use it at work.  There 
is however an alternative approach.  You can go to the NLMpy Python Package Index page 
(https://pypi.python.org/pypi/nlmpy) and download the nlmpy-0.1.1.tar.gz package file.  
With the file downloaded to your computer you can then use pip to install NLMpy from this 
local file by running::

    pip install /Users/username/Downloads/nlmpy-0.1.5.tar.gz

though you will obviously have to specify the directory location for the package file that 
is correct on your computer!

If you are unable to get either of those approaches to work, there is an easy workaround.  
If you decompress the nlmpy-0.1.5.tar.gz package file, you will find inside a nlmpy.py file 
that contains all the NLMpy functions.  If you simply move this file to the same location 
on your computer as a Python script that wants to import nlmpy, then when those scripts are 
executed they will import all the NLMpy functions.  So while this approach doesn’t 
actually install NLMpy onto your computer, it does at least allow you to make use of the 
functionality of NLMpy within a neighbouring Python script.
