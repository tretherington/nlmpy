#------------------------------------------------------------------------------

# The MIT License (MIT)

# Copyright (c) 2014 Thomas R. Etherington, E. Penelope Holland, and David O'Sullivan.
# Copyright (c) 2019 Pierre Vigier
# Copyright (c) 2022 Landcare Research New Zealand Ltd

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

#------------------------------------------------------------------------------

import sys
import numpy as np
from scipy import ndimage
from numba import jit

#------------------------------------------------------------------------------
# REQUIRED FUNCTIONS:
#------------------------------------------------------------------------------

def linearRescale01(array):
    """    
    A rescale in which the values in the array are linearly rescaled to range
    between 0 and 1.

    Parameters
    ----------
    array : array
        2D array of data values.
        
    Returns
    -------
    out : array
        2D array with rescaled values.
    """
    rescaledArray = (array - np.nanmin(array)) / np.nanmax(array - np.nanmin(array))
    return(rescaledArray)

#------------------------------------------------------------------------------

# A function to insert nan cells into an array based on a binary mask array.
def maskArray(array, maskArray):
    """    
    Return the array with nan values inserted where present in the mask array.
    It is assumed that both the arrays have the same dimensions.

    Parameters
    ----------
    array : array
        2D array of data values.
    maskArray : array
        2D array used as a binary mask.
        
    Returns
    -------
    out : array
        2D array with masked values.
    """
    np.place(array, maskArray==0, np.nan)
    return(array)

#------------------------------------------------------------------------------

def randomUniform01(nRow, nCol, mask=None):
    """    
    Create an array with random values ranging 0-1.

    Parameters
    ----------
    nRow : int
        The number of rows in the array.
    nCol : int
        The number of columns in the array.
    mask : array, optional
        2D array used as a binary mask to limit the elements with values.
        
    Returns
    -------
    out : array
        2D float array.
    """
    array = np.random.random((nRow, nCol))
    if mask is not None:
        array = maskArray(array, mask)
    rescaledArray = linearRescale01(array)
    return(rescaledArray)
    
#------------------------------------------------------------------------------

def nnInterpolate(array, missing):
    """    
    Two-dimensional array nearest-neighbour interpolation in which the elements
    in the positions indicated by the array "missing" are replaced by the
    nearest value from the "array" of data values.

    Parameters
    ----------
    array : array
        2D array of data values.
    missing: boolean array
        Values of True receive interpolated values.
        
    Returns
    -------
    out : array
        2D array with interpolated values.
    """
    # Get row column based index of nearest value
    rcIndex = ndimage.distance_transform_edt(missing, return_distances=False, 
                                             return_indices=True)
    # Create a complete array by extracting values based on the index
    interpolatedArray = array[tuple(rcIndex)]
    return(interpolatedArray)

#------------------------------------------------------------------------------

def w2cp(weights):
    """    
    Convert a list of category weights into a 1D NumPy array of cumulative 
    proportions.

    Parameters
    ----------
    weights : list
        A list of numeric values
        
    Returns
    -------
    out : array
        1D array of class cumulative proportions.
    """
    w = np.array(weights, dtype=float)
    proportions = w / np.sum(w)
    cumulativeProportions = np.cumsum(proportions)
    cumulativeProportions[-1] = 1 # to ensure the last value is 1
    return(cumulativeProportions)

#------------------------------------------------------------------------------

def calcBoundaries(array, cumulativeProportions, classifyMask=None):
    """    
    Determine upper class boundaries for classification of an array with values
    ranging 0-1 based upon an array of cumulative proportions.

    Parameters
    ----------
    array : array
        2D array of data values.
    cumulativeProportions : array
        1D array of class cumulative proportions.
    classifyMask : array, optional
        2D array used as a binary mask to limit the elements used to determine
        the upper boundary values for each class.
        
    Returns
    -------
    out : array
        1D float array.
    """
    if classifyMask is None:
        classifyMask = np.ones(np.shape(array))
    maskedArray = array * classifyMask
    np.place(maskedArray, classifyMask==0, np.nan)
    # Determine the number of cells that are in the classification mask.
    nCells = np.count_nonzero(np.isfinite(maskedArray))
    # Based on the number of cells, find the index of upper boundary element
    boundaryIndexes = (cumulativeProportions * nCells).astype(int) - 1
    # Index out the the upper boundary value for each class
    boundaryValues = np.sort(np.ndarray.flatten(maskedArray))[boundaryIndexes]
    # Ensure the maximum boundary value is equal to 1
    boundaryValues[-1] = 1
    return(boundaryValues)

#------------------------------------------------------------------------------
      
def classifyArray(array, weights, classifyMask=None):
    """    
    Classify an array with values ranging 0-1 into proportions based upon a 
    list of class weights.

    Parameters
    ----------
    array : array
        2D array of data values.
    weights : list
        A list of numeric values
    classifyMask : array, optional
        2D array used as a binary mask to limit the elements used to determine
        the upper boundary values for each class.
        
    Returns
    -------
    out : array
        2D array.
    """
    cumulativeProportions = w2cp(weights)
    boundaryValues = calcBoundaries(array, cumulativeProportions, classifyMask)
    # Classify the array
    classifiedArray = np.searchsorted(boundaryValues, array)
    # Replace any nan values
    classifiedArray = classifiedArray.astype(float)
    np.place(classifiedArray, np.isnan(array), np.nan)
    return(classifiedArray)

#------------------------------------------------------------------------------

def blendArrays(arrays, scalingFactors=None):
    """    
    Blend arrays weighted by scaling factors.

    Parameters
    ----------
    arrays : list
        List of 2D arrays of data values.
    scalingFactors : list
        List of scaling factors used to weight the arrays in the blend.
        
    Returns
    -------
    out : array
        2D array.
    """
    if scalingFactors is None:
        scalingFactors = np.ones(len(arrays))
    combinedArrays = np.zeros(arrays[0].shape)
    for n in range(len(arrays)):
        combinedArrays = combinedArrays + (arrays[n] * scalingFactors[n])
    blendedArray = combinedArrays / len(arrays)
    rescaledArray = linearRescale01(blendedArray)
    return(rescaledArray)
    
#------------------------------------------------------------------------------

def blendClusterArray(primaryArray, arrays, scalingFactors=None):
    """    
    Blend a primary cluster NLM with other arrays in which the mean value per 
    cluster is weighted by scaling factors.

    Parameters
    ----------
    primaryArray : array
        2D array of data values in which values are clustered.
    arrays : list
        List of 2D arrays of data values.
    scalingFactors : list
        List of scaling factors used to weight the arrays in the blend.
        
    Returns
    -------
    out : array
        2D array.
    """
    if scalingFactors is None:
        scalingFactors = np.ones(len(arrays))
    for n in range(len(arrays)):
        meanOfClusterArray = meanOfCluster(primaryArray, arrays[n])
        primaryArray = primaryArray + (meanOfClusterArray * scalingFactors[n])
    blendedArray = primaryArray / len(arrays)
    rescaledArray = linearRescale01(blendedArray)
    return(rescaledArray)
    
#------------------------------------------------------------------------------

def meanOfCluster(clusterArray, array):
    """    
    For each cluster of elements in an array, calculate the mean value for the
    cluster based on a second array.

    Parameters
    ----------
    clutserArray : array
        2D array of data values in which values are clustered.
    array : array
        2D array of data values.
        
    Returns
    -------
    out : array
        2D array.
    """
    meanClusterValues = np.zeros(np.shape(clusterArray))
    clusterValues = np.unique(clusterArray)
    for value in clusterValues:
        if np.isfinite(value):
            # Extract location of values
            valueLocs = clusterArray == value
            # Define clusters in array
            clusters, nClusters = ndimage.measurements.label(valueLocs)
            # Get mean for each cluster
            means = ndimage.mean(array, clusters, range(1,nClusters + 1))
            means = np.insert(means, 0, 0) # for background non-cluster
            # Apply mean values to clusters by index
            clusterMeans = means[clusters]
            # Add values for those clusters to array
            meanClusterValues = meanClusterValues + clusterMeans
    np.place(meanClusterValues, np.isnan(clusterArray), np.nan)
    rescaledArray = linearRescale01(meanClusterValues)
    return(rescaledArray)

#------------------------------------------------------------------------------

def exportASCIIGrid(outFile, nlm, xll=0, yll=0, cellSize=1):
    """
    Export a NLM array as a ASCII grid raster file.
    
    Parameters
    ----------
    outFile : string
        The path and name of the output raster file.
    nlm : 2D array
        The NLM to be exported.
    xll : number
        Raster lower left corner x coordinate.
    yll : number
        Raster lower left corner y coordinate.
    cellSize : number
        The size of the cells in the output raster.
    """
    # Get dimensions of the NLM
    nRow, nCol = nlm.shape
    # Convert any nan elements to null data value of -9999
    np.place(nlm, np.isnan(nlm), -9999)
    # Create raster out file
    textOut = open(outFile, 'w')
    # Write metadata
    textOut.write("NCOLS " + str(nCol) + "\n")
    textOut.write("NROWS " + str(nRow) + "\n")
    textOut.write("XLLCORNER " + str(xll) + "\n")
    textOut.write("YLLCORNER " + str(yll) + "\n")
    textOut.write("CELLSIZE " + str(cellSize) + "\n")
    textOut.write("NODATA_VALUE -9999\n")
    # Write NLM
    for row in range(nRow):
        lineout = ""
        for col in range(nCol):
            lineout = lineout + str(nlm[row,col]) + " "
        textOut.write(lineout[:-1] + "\n")
    textOut.close()

#------------------------------------------------------------------------------
# NEUTRAL LANDSCAPE MODELS:
#------------------------------------------------------------------------------

def random(nRow, nCol, mask=None):
    """    
    Create a spatially random neutral landscape model with values ranging 0-1.

    Parameters
    ----------
    nRow : int
        The number of rows in the array.
    nCol : int
        The number of columns in the array.
    mask : array, optional
        2D array used as a binary mask to limit the elements with values.
        
    Returns
    -------
    out : array
        2D array.
    """
    array = randomUniform01(nRow, nCol, mask)
    return(array)
    
#------------------------------------------------------------------------------

def planarGradient(nRow, nCol, direction=None, mask=None):
    """    
    Create a planar gradient neutral landscape model with values ranging 0-1.

    Parameters
    ----------
    nRow : int
        The number of rows in the array.
    nCol : int
        The number of columns in the array.
    direction: int, optional
        The direction of the gradient as a bearing from north, if unspecified
        the direction is randomly determined.
    mask : array, optional
        2D array used as a binary mask to limit the elements with values.
        
    Returns
    -------
    out : array
        2D array.
    """
    if direction is None:
        direction = np.random.uniform(0, 360, 1) # a random direction
    # Create arrays of row and column index
    rowIndex, colIndex = np.indices((nRow, nCol))
    # Determine the eastness and southness of the direction
    eastness = np.sin(np.deg2rad(direction))
    southness = np.cos(np.deg2rad(direction)) * -1
    # Create gradient array
    gradient = (southness * rowIndex + eastness * colIndex)
    if mask is not None:
        gradient = maskArray(gradient, mask)
    rescaledArray = linearRescale01(gradient)
    return(rescaledArray)

#------------------------------------------------------------------------------

def edgeGradient(nRow, nCol, direction=None, mask=None):
    """    
    Create an edge gradient neutral landscape model with values ranging 0-1.

    Parameters
    ----------
    nRow : int
        The number of rows in the array.
    nCol : int
        The number of columns in the array.
    direction: int, optional
        The direction of the gradient as a bearing from north, if unspecified
        the direction is randomly determined.
    mask : array, optional
        2D array used as a binary mask to limit the elements with values.
        
    Returns
    -------
    out : array
        2D array.
    """
    # Create planar gradient
    gradient = planarGradient(nRow, nCol, direction, mask)
    # Transform to a central gradient
    edgeGradient = (np.abs(0.5 - gradient) * -2) + 1
    rescaledArray = linearRescale01(edgeGradient)
    return(rescaledArray)

#------------------------------------------------------------------------------

def distanceGradient(source, mask=None):
    """    
    Create a distance gradient neutral landscape model with values ranging 0-1.

    Parameters
    ----------
    source : array
        2D array binary array that defines the source elements from which
        distance will be measured.  The dimensions of source also specify
        the output dimensions of the distance gradient.
    mask : array, optional
        2D array used as a binary mask to limit the elements with values.
        
    Returns
    -------
    out : array
        2D array.
    """
    gradient = ndimage.distance_transform_edt(1 - source)
    if mask is not None:
        gradient = maskArray(gradient, mask)
    rescaledArray = linearRescale01(gradient)
    return(rescaledArray)

#------------------------------------------------------------------------------

def waveSurface(nRow, nCol, periods, direction=None, mask=None):
    """
    Create a waves neutral landscape model with values ranging 0-1.

    Parameters
    ----------
    nRow : int
        The number of rows in the array.
    nCol : int
        The number of columns in the array.
    periods: int
        The number of periods in the landscape, where a period consists of a 
        complete wave cycle of one crest and one trough.
    direction: int, optional
        The direction of the waves as a bearing from north, if unspecified
        the direction is randomly determined.
    mask : array, optional
        2D array used as a binary mask to limit the elements with values.
        
    Returns
    -------
    out : array
        2D array.
    """
    gradient = planarGradient(nRow, nCol, direction)
    waves = np.sin(gradient * (2 * np.pi * periods))
    if mask is not None:
        waves = maskArray(waves, mask)
    rescaledArray = linearRescale01(waves)
    return(rescaledArray)

#------------------------------------------------------------------------------

def mpd(nRow, nCol, h, mask=None):
    """    
    Create a midpoint displacement neutral landscape model with values ranging 
    0-1.

    Parameters
    ----------
    nRow : int
        The number of rows in the array.
    nCol : int
        The number of columns in the array.
    h: float
        The h value controls the level of spatial autocorrelation in element
        values.
    mask : array, optional
        2D array used as a binary mask to limit the elements with values.
        
    Returns
    -------
    out : array
        2D array.
    """
    # Determine the dimension of the smallest square
    maxDim = np.max(np.array([nRow, nCol]))
    N = np.int(np.ceil(np.log2(maxDim - 1)))
    dim = 2 ** N + 1
    # Create surface and extract required array size if needed
    surface = diamondsquare(dim, h)
    if (nRow, nCol) != surface.shape:
        surface = extractRandomArrayFromSquareArray(surface, nRow, nCol)
    # Apply mask and rescale 0-1
    if mask is not None:
        surface = maskArray(surface, mask)
    rescaledArray = linearRescale01(surface)
    return(rescaledArray)

def extractRandomArrayFromSquareArray(array, nRow, nCol):
    # Extract a portion of the array to match the dimensions
    dim = array.shape[0]
    randomStartRow = np.random.choice(range(dim - nRow))
    randomStartCol = np.random.choice(range(dim - nCol))
    return(array[randomStartRow:randomStartRow + nRow,
                 randomStartCol:randomStartCol + nCol])

def diamondsquare(dim, h):
    # Create a surface consisting of random displacement heights average value
    # 0, range from [-0.5, 0.5] x displacementheight
    disheight = 2.0
    randomValues = np.random.random(size=dim*dim)
    surface = np.reshape(randomValues, (dim, dim))
    surface = surface * disheight -0.5 * disheight
    # Set square size to cover the whole array
    inc = dim-1
    while inc > 1: # while considering a square/diamond at least 2x2 in size
            i2 = int(inc/2) # what is half the width (i.e. where is the centre?)
            # SQUARE step
            for x in range(0,dim-1,inc):
                for y in range(0,dim-1,inc):
                    # this adjusts the centre of the square 
                    surface[x+i2,y+i2] = displacevals(np.array([surface[x,y],surface[x+inc,y],surface[x+inc,y+inc],surface[x,y+inc]]), disheight, np.random.random(2))
            # DIAMOND step
            for x in range(0, dim-1, inc):
                for y in range(0, dim-1,inc):
                    diaco = check_diamond_coords(x+i2,y,dim,i2)
                    diavals = np.zeros((len(diaco),))
                    for c in range(len(diaco)):
                        diavals[c] = surface[diaco[c]]
                    surface[x+i2,y] = displacevals(diavals,disheight,np.random.random(2))
                    diaco = check_diamond_coords(x,y+i2,dim,i2)
                    diavals = np.zeros((len(diaco),))
                    for c in range(len(diaco)):
                        diavals[c] = surface[diaco[c]]
                    surface[x,y+i2] = displacevals(diavals,disheight,np.random.random(2))
                    diaco = check_diamond_coords(x+inc,y+i2,dim,i2)
                    diavals = np.zeros((len(diaco),))
                    for c in range(len(diaco)):
                        diavals[c] = surface[diaco[c]]
                    surface[x+inc,y+i2] = displacevals(diavals,disheight,np.random.random(2))
                    diaco = check_diamond_coords(x+i2,y+inc,dim,i2)
                    diavals = np.zeros((len(diaco),))
                    for c in range(len(diaco)):
                        diavals[c] = surface[diaco[c]]
                    surface[x+i2,y+inc] = displacevals(diavals,disheight,np.random.random(2))
            # Reduce displacement height
            disheight = disheight * 2 ** (-float(h))
            inc = int(inc / 2)
    return(surface)
    
@jit(nopython=True)
def displacevals(p, disheight, r):
    # Calculate the average value of the 4 corners of a square (3 if up
    # against a corner) and displace at random.
    if len(p) == 4:
        pcentre = 0.25 * np.sum(p) + randomdisplace(disheight, r[0])
    elif len(p) == 3:
        pcentre = np.sum(p) / 3 + randomdisplace(disheight, r[1])	
    return(pcentre)

@jit(nopython=True)
def randomdisplace(disheight, r):
    # Returns a random displacement between -0.5 * disheight and 0.5 * disheight
    return(r * disheight -0.5 * disheight)
    
def check_diamond_coords(diax,diay,dim,i2):
    # get the coordinates of the diamond centred on diax, diay with radius i2
    # if it fits inside the study area
    if diax < 0 or diax > dim or diay <0 or diay > dim:
        return([])
    if diax-i2 < 0:
        return([(diax+i2,diay),(diax,diay-i2),(diax,diay+i2)])
    if diax + i2 >= dim:
        return([(diax-i2,diay),(diax,diay-i2),(diax,diay+i2)])
    if diay-i2 < 0:
        return([(diax+i2,diay),(diax-i2,diay),(diax,diay+i2)])
    if diay+i2 >= dim:
        return([(diax+i2,diay),(diax-i2,diay),(diax,diay-i2)])
    return([(diax+i2,diay),(diax-i2,diay),(diax,diay-i2),(diax,diay+i2)])    

#------------------------------------------------------------------------------

def perlinNoise(nRow, nCol, periods, octaves=1, lacunarity=2, persistence=0.5,
                rescale='linear', mask=None):
    """    
    Create a Perlin noise neutral landscape model with values ranging 0-1.

	For a full description of the method see Etherington TR (2022) Perlin noise 
	as a hierarchical neutral landscape model. Web Ecology 22:1-6. 
	http://doi.org/10.5194/we-22-1-2022
	
    Parameters
    ----------
    nRow : int
        The number of rows in the array.
    nCol : int
        The number of columns in the array.
    periods: tuple
        Integers for the number of periods of Perlin noise across row and 
        column dimensions for the first octave.
    octaves : int
        The number of octaves that will form the Perlin noise.
    lacunarity : int
        The rate at which the frequency of periods increases for each octive.
    persistance : float
        The rate at which the amplitude of periods decreases for each octive.
    rescale: string
        How the Perlin noise values are rescaled between 0 and 1.  Options 
		includes: 'linear' (the default), 'absolute', and 'squared'.
    mask : array, optional
        2D array used as a binary mask to limit the elements with values.
        
    Returns
    -------
    out : array
        2D array.
    """
    # nRow must equal nCol so determine the dimension of the smallest square
    dim = max(nRow, nCol)
    # Check the dim is a multiple of each octives maximum number of periods and 
    # expand dim if needed
    rPeriodsMax = periods[0] * (lacunarity ** (octaves - 1))
    cPeriodsMax = periods[1] * (lacunarity ** (octaves - 1))
    periodsMultiple = np.lcm(rPeriodsMax, cPeriodsMax) # lowest commom multiple
    if dim % periodsMultiple != 0:
        dim = int(np.ceil(dim / periodsMultiple) * periodsMultiple)

    # Generate the Perlin noise
    noise = np.zeros((dim, dim))
    for octive in range(octaves):
        noise = noise + octave(dim, dim, periods, octive, lacunarity, persistence)
    # If needed randomly extract the desired array size
    if (nRow, nCol) != noise.shape:
        noise = extractRandomArrayFromSquareArray(noise, nRow, nCol)    
    
    # Rescale the Perlin noise
    if rescale == 'linear':
        surface = linearRescale01(noise)
    if rescale == 'absolute':
        surface = linearRescale01(np.abs(noise))
    if rescale == 'squared':
        surface = linearRescale01(noise ** 2)

    # Apply mask
    if mask is not None:
        surface = maskArray(surface, mask)
    return(surface)

def octave(nRow, nCol, periods, octive, lacunarity, persistence):        
    rP, cP = periods
    rP = rP * (lacunarity ** octive)
    cP = cP * (lacunarity ** octive)
    delta = (rP / nRow, cP / nCol)
    d = (nRow // rP, nCol // cP)
    grid = np.mgrid[0:rP:delta[0],0:cP:delta[1]].transpose(1, 2, 0) % 1
    # Gradients
    angles = 2 * np.pi * np.random.rand(rP + 1, cP + 1)
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    gradients = gradients.repeat(d[0], 0).repeat(d[1], 1)
    g00 = gradients[    :-d[0],    :-d[1]]
    g10 = gradients[d[0]:     ,    :-d[1]]
    g01 = gradients[    :-d[0],d[1]:     ]
    g11 = gradients[d[0]:     ,d[1]:     ]
    # Ramps
    n00 = np.sum(np.dstack((grid[:,:,0]  , grid[:,:,1]  )) * g00, 2)
    n10 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1]  )) * g10, 2)
    n01 = np.sum(np.dstack((grid[:,:,0]  , grid[:,:,1]-1)) * g01, 2)
    n11 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1]-1)) * g11, 2)
    # Interpolation
    t = f(grid)
    n0 = n00 * (1 - t[:,:,0]) + t[:,:,0] * n10
    n1 = n01 * (1 - t[:,:,0]) + t[:,:,0] * n11
    octave = np.sqrt(2) * ((1 - t[:,:,1]) * n0 + t[:,:,1] * n1)
    return(octave * (persistence ** octive))
    
@jit(nopython=True)
def f(t):
    return(6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3)

#------------------------------------------------------------------------------

def randomRectangularCluster(nRow, nCol, minL, maxL, mask=None):
    """    
    Create a random rectangular cluster neutral landscape model with 
    values ranging 0-1.

    Parameters
    ----------
    nRow : int
        The number of rows in the array.
    nCol : int
        The number of columns in the array.
    minL: int
        The minimum possible length of width and height for each random 
        rectangular cluster.
    maxL: int
        The maximum possible length of width and height for each random 
        rectangular cluster.
    mask : array, optional
        2D array used as a binary mask to limit the elements with values.
        
    Returns
    -------
    out : array
        2D array.
    """    
    # Create an empty array of correct dimensions
    array = np.zeros((nRow, nCol)) - 1
    # Keep applying random clusters until all elements have a value
    while np.min(array) == -1:
        width = np.random.choice(range(minL, maxL))
        height = np.random.choice(range(minL, maxL))
        row = np.random.choice(range(-maxL, nRow))
        col = np.random.choice(range(-maxL, nCol))
        array[row:row + width, col:col + height] = np.random.random()   
    # Apply mask and rescale 0-1
    if mask is not None:
        array = maskArray(array, mask)
    rescaledArray = linearRescale01(array)
    return(rescaledArray)

#------------------------------------------------------------------------------

def randomElementNN(nRow, nCol, n, mask=None, categorical=False):
    """    
    Create a random element nearest-neighbour neutral landscape model with 
    values ranging 0-1.

    Parameters
    ----------
    nRow : int
        The number of rows in the array.
    nCol : int
        The number of columns in the array.
    n: int
        The number of elements randomly selected to form the basis of
        nearest-neighbour clusters.
    mask : array, optional
        2D array used as a binary mask to limit the elements with values.
        
    Returns
    -------
    out : array
        2D array.
    """
    # Create an empty array of correct dimensions
    array = np.zeros((nRow, nCol))
    if mask == None:
        mask = np.ones((nRow, nCol))
    # Insert value for n elements
    i = 1
    while np.max(array) < n:
        randomRow = np.random.choice(range(nRow))
        randomCol = np.random.choice(range(nCol))
        if array[randomRow, randomCol] == 0 and mask[randomRow, randomCol] == 1:
            array[randomRow, randomCol] = i
            i = i + 1
    # Interpolate the values
    interpolatedArray = nnInterpolate(array, array==0) - 1 # -1 to index from 0
    # Apply mask and rescale 0-1
    if mask is not None:
        interpolatedArray = maskArray(interpolatedArray, mask)
    if categorical == False:
        rescaledArray = linearRescale01(interpolatedArray)
        return(rescaledArray)
    else:
        return(interpolatedArray.astype('int'))
    
 
#------------------------------------------------------------------------------

def randomClusterNN(nRow, nCol, p, n='4-neighbourhood', mask=None):
    """    
    Create a random cluster nearest-neighbour neutral landscape model with 
    values ranging 0-1.

    Parameters
    ----------
    nRow : int
        The number of rows in the array.
    nCol : int
        The number of columns in the array.
    p: float
        The p value controls the proportion of elements randomly selected to
        form clusters.
    n: string, optional
        Clusters are defined using a set of neighbourhood structures that 
        include:
                            [0,1,0]
        '4-neighbourhood' = [1,1,1]
                            [0,1,0]
                            
                            [1,1,1]
        '8-neighbourhood' = [1,1,1]
                            [1,1,1]
                            
                     [0,1,1]
        'diagonal' = [1,1,1]
                     [1,1,0]
                     
        The default parameter setting is '4-neighbourhood'.
        
    mask : array, optional
        2D array used as a binary mask to limit the elements with values.
        
    Returns
    -------
    out : array
        2D array.
    """
    # Define a dictionary of possible neighbourhood structures:
    neighbourhoods = {}
    neighbourhoods['4-neighbourhood'] = np.array([[0,1,0],
                                                  [1,1,1],
                                                  [0,1,0]])
    neighbourhoods['8-neighbourhood'] = np.array([[1,1,1],
                                                  [1,1,1],
                                                  [1,1,1]])
    neighbourhoods['diagonal'] = np.array([[0,1,1],
                                           [1,1,1],
                                           [1,1,0]])
    # Create percolation array
    randomArray = random(nRow, nCol, mask)
    percolationArray = classifyArray(randomArray, [1 - p, p])
    # As nan not supported in cluster algorithm replace with zeros
    np.place(percolationArray, np.isnan(percolationArray), 0)
    # Define clusters
    clusters, nClusters = ndimage.measurements.label(percolationArray, 
                                                     neighbourhoods[n])
    # Create random set of values for each the clusters
    randomValues = np.random.random(nClusters)
    randomValues = np.insert(randomValues, 0, 0) # for background non-cluster
    # Apply values by indexing by cluster
    clusterArray = randomValues[clusters]
    # Gap fill with nearest neighbour interpolation
    interpolatedArray = nnInterpolate(clusterArray, clusterArray==0)
    # Apply mask and rescale 0-1
    if mask is not None:
        interpolatedArray = maskArray(interpolatedArray, mask)
    rescaledArray = linearRescale01(interpolatedArray)
    return(rescaledArray)

#------------------------------------------------------------------------------

def bsp(nRow, nCol, n, partOrient='longest', maxLWRatio=2, p=None):
    """    
    Create a binary space partitioning (BSP) neutral landscape model (NLM).
    
    For a full description of the method see Etherington et al. (2022) Binary 
    space partitioning generates hierarchical and rectilinear neutral landscape 
    models suitable for human-dominated landscapes. Landscape Ecology 
    37:1761–1769. http://doi.org/10.1007/s10980-022-01452-6

    Parameters
    ----------
    nRow : int
        The number of rows in the NLM array.
    nCol : int
        The number of columns in the NLM array.
    n: int
        The number of patches desired in the resulting NLM array.
    partOrient: string
        One of 'longest', 'random', 'horizontal', 'vertical' that specifies the
        prefered orientation of the patch BSP, default value is 'longest'
    maxLWRatio: integer or float
        The maximum permitted length to width ratio of patches created by BSP.
        This parameter must be equal to greater than the default value is 2 to 
        allow a square to be partitioned.
    p: array, optional
        2D array used to determine the probability of BSP occurring across the
        NLM array.  This parameter is optional, and if unspecified the 
        probability of BSP = 1 everywhere within the array.
        
    Returns
    -------
    out : tuple 
        The ouput tuple contains a series of data structures:
            (0) 2D array: An NLM of intger values that denote the location of
                each patch in the landscape.
            (1) Dictionary: the BSP tree keyed by parent patch that returns a
                tuple of the parent patch's child patches.
            (2) Dictionary: keyed by leaf patches, this dictionary records the
                tree height of the BSP leaf nodes/patches.
            (3) Dictionary: keyed by patch, this dictionary returns a tuple of
                tuples that record the array indices for the patch dimensions 
                by row and column ((min row, max row), (min col, max col)).
            (4) Dictionary: keyed by tree height, this dictionary returns a 
                list of the patches at the specified BSP tree height.
    """
    # Extract inputs and build default inputs
    if p is None:
        p = np.ones((nRow, nCol))
    # If a probability array is supplied, check its dimensions
    else:
        if ((nRow, nCol)) != p.shape:
            print("The probability array is not the same shape as the specified NLM")
            sys.exit()
        
    # Initiate data structures and progress counters
    bspSpace = np.zeros((nRow, nCol)).astype('int')
    patchDims = {}
    patchDims[0] = ((0, nRow), (0, nCol))    
    bspTree = {}
    treeHeights = [[0]]
    leafNodes = {}
    leafNodes[0] = 0
    # Run binary space partitioning
    n = n - 1 # as have created root
    bspResults = bspPartitioning(n, bspSpace, bspTree, leafNodes, patchDims, treeHeights, 
                                 partOrient, maxLWRatio, p)
    return(bspResults)

#------------------------------------------------------------------------------

def bspPartitioning(n, bspSpace, bspTree, leafNodes, patchDims, treeHeights, 
                    partOrient='longest', maxLWRatio=2, p=None):
    """    
    This function conducts binary space partitioning (BSP), and can be used to
    add further partions to an existing BSP neutral landscape model (NLM).
    
    For a full description of the method see Etherington et al. (2022) Binary 
    space partitioning generates hierarchical and rectilinear neutral landscape 
    models suitable for human-dominated landscapes. Landscape Ecology 
    37:1761–1769. http://doi.org/10.1007/s10980-022-01452-6

    Parameters
    ----------
    n: int
        The number of partitions to add.
    bspSpace: 2D array
        An NLM of intger values that denote the location of each patch in the 
        landscape.
    bspTree: dictionary
        The BSP tree keyed by parent patch that returns a tuple of the parent 
        patch's child patches.
    leafNodes: dictionary
        Keyed by leaf patches, this dictionary records the tree height of the 
        BSP leaf nodes/patches.
    patchDims: dictionary
        Keyed by patch, this dictionary returns a tuple of tuples that record 
        the array indices for the patch dimensions by row and column 
        ((min row, max row), (min col, max col)).
    treeHeights: dictionary
        Keyed by tree height, this dictionary returns a list of the patches at 
        the specified BSP tree height.  
    partOrient: string
        One of 'longest', 'random', 'horizontal', 'vertical' that specifies the
        prefered orientation of the patch BSP, default value is 'longest'
    maxLWRatio: integer or float
        The maximum permitted length to width ratio of patches created by BSP.
        This parameter must be equal to greater than the default value is 2 to 
        allow a square to be partitioned.
    p: array, optional
        2D array used to determine the probability of BSP occurring across the
        NLM array.  This parameter is optional, and if unspecified the 
        probability of BSP = 1 everywhere within the array.
        
    Returns
    -------
    out : tuple 
        The ouput tuple contains a series of data structures:
            (0) 2D array: An NLM of intger values that denote the location of
                each patch in the landscape.
            (1) Dictionary: the BSP tree keyed by parent patch that returns a
                tuple of the parent patch's child patches.
            (2) Dictionary: keyed by leaf patches, this dictionary records the
                tree height of the BSP leaf nodes/patches.
            (3) Dictionary: keyed by patch, this dictionary returns a tuple of
                tuples that record the array indices for the patch dimensions 
                by row and column ((min row, max row), (min col, max col)).
            (4) Dictionary: keyed by tree height, this dictionary returns a 
                list of the patches at the specified BSP tree height.
    """    
    # Extract inputs and build default inputs
    nRow, nCol = bspSpace.shape
    if p is None:
        p = np.ones((nRow, nCol))
    # If a probability array is supplied, check its dimensions
    else:
        if ((nRow, nCol)) != p.shape:
            print("The probability array is not the same shape as the existing BSP space")
            sys.exit()    
    # Check keyword and ratio is appropriate, if not end with error message
    if maxLWRatio < 2:
        print("maxLWRatio parameter must be >=2")
        sys.exit()
    if partOrient not in ['longest', 'random', 'horizontal', 'vertical']:
        print("The partOrient parameter must be one of: 'longest', 'random', 'horizontal', 'vertical'")
        sys.exit()
    maxLeafNode = np.nanmax(bspSpace)
    patchCount = len(np.unique(bspSpace))
    totalPatches = patchCount + n
    # Create dictionary to record tree heights of all partitions
    treeHeight = {}
    for height in range(len(treeHeights)):
        for partition in treeHeights[height]:
            treeHeight[partition] = height
    
    # Start creating patches
    while patchCount < totalPatches:
        # Pick a random leaf node
        randomRow = np.random.randint(low=0, high=nRow, size=1)
        randomCol = np.random.randint(low=0, high=nCol, size=1)
        randomLeafNode = bspSpace[randomRow, randomCol][0]
        # Get dimensions for random leaf node cells
        minRow, maxRow = patchDims[randomLeafNode][0]
        minCol, maxCol = patchDims[randomLeafNode][1]
        dimCols = maxCol - minCol
        dimRows = maxRow - minRow        
        # Check probability of partition
        patch = p[minRow:maxRow, minCol:maxCol]
        if np.sum(np.isfinite(patch)) > 0: # check not all nan
            meanP = np.nanmean(patch)
            if meanP >= np.random.uniform(size=1):
                # Determine split direction
                orientation = np.random.choice(['horizontal', 'vertical'])
                if partOrient == 'longest':
                    # If there is a long axis use that
                    if (maxRow - minRow) > (maxCol - minCol):
                        orientation = 'horizontal'
                    if (maxRow - minRow) < (maxCol - minCol):
                        orientation = 'vertical'
                else:
                    # Apply the defined orientation if there is one
                    if partOrient != 'random':
                        orientation = partOrient        
                    # Check the resulting length width ratio is within the limit, 
                    # and if not then switch the partition direction
                    if orientation == 'horizontal':
                        if dimCols / (dimRows / 2) > maxLWRatio:
                            orientation = 'vertical'
                    else:
                        if dimRows / (dimCols / 2) > maxLWRatio:
                            orientation = 'horizontal'
                # Check partition can occur
                partitionCheck = False
                if orientation == 'horizontal':
                    if dimRows >= 2:
                        partitionCheck = True
                if orientation == 'vertical':
                    if dimCols >= 2:
                        partitionCheck = True            
                # Partion patch
                if partitionCheck == True:
                    patchCount = patchCount + 1
                    height = leafNodes[randomLeafNode]
                    del leafNodes[randomLeafNode]               
                    if orientation == 'horizontal':
                        partitionRow = int(minRow + (dimRows / 2))
                        bspSpace[minRow:partitionRow, minCol:maxCol] = maxLeafNode + 1
                        patchDims[maxLeafNode + 1] = ((minRow, partitionRow), (minCol, maxCol))
                        bspSpace[partitionRow:maxRow, minCol:maxCol] = maxLeafNode + 2
                        patchDims[maxLeafNode + 2] = ((partitionRow, maxRow), (minCol, maxCol))
                    if orientation == 'vertical':
                        partitionCol = int(minCol + (dimCols / 2))
                        bspSpace[minRow:maxRow, minCol:partitionCol] = maxLeafNode + 1
                        patchDims[maxLeafNode + 1] = ((minRow, maxRow), (minCol, partitionCol))
                        bspSpace[minRow:maxRow, partitionCol:maxCol] = maxLeafNode + 2
                        patchDims[maxLeafNode + 2] = ((minRow, maxRow), (partitionCol, maxCol))
                    # Update tree
                    bspTree[randomLeafNode] = (maxLeafNode + 1, maxLeafNode + 2)
                    leafNodes[maxLeafNode + 1] = height + 1
                    leafNodes[maxLeafNode + 2] = height + 1
                    treeHeight[maxLeafNode + 1] = treeHeight[randomLeafNode] + 1
                    treeHeight[maxLeafNode + 2] = treeHeight[randomLeafNode] + 1
                    if len(treeHeights) <= treeHeight[randomLeafNode] + 1:
                        treeHeights.append([])
                    treeHeights[treeHeight[randomLeafNode] + 1].append(maxLeafNode + 1)
                    treeHeights[treeHeight[randomLeafNode] + 1].append(maxLeafNode + 2)
                    maxLeafNode = maxLeafNode + 2
    return(bspSpace, bspTree, leafNodes, patchDims, treeHeights)

#------------------------------------------------------------------------------