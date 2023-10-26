#------------------------------------------------------------------------------

# The MIT License (MIT)

# Copyright (c) 2023 Landcare Research New Zealand Ltd

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

# Developed with:
# Python 3.10.6 | packaged by conda-forge | (main, Aug 22 2022, 20:30:19) [MSC v.1929 64 bit (AMD64)]
import numpy as np # version 1.22.3
from scipy import sparse, ndimage, spatial # version 1.9.0

#------------------------------------------------------------------------------

def createCellVertices(costsurface):
    """
    Create unique values for each cell in a cost-surface that has a cost value 
    that will represent the vertices in the landscape graph.
    
    Parameters
    ----------
    costsurface : array
        2D array of cost values.
    Returns
    -------
    cellVertices : integer array
        2D array of same dimensions as the costsurface in which each cell has 
        a unique value with no data cells having the last value.
    verticesRowCol : integer array
        An n x 2 array, where each row is a non-null costsurface cell, and the 
        columns give the row and column locations of each cell.
    
    """
    nRow, nCol = costsurface.shape
    rowIndex, colIndex = np.mgrid[0:nRow, 0:nCol]
    finiteCells = np.isfinite(costsurface)
    cellID = np.array(range(np.sum(finiteCells)))
    cellIndexRow = rowIndex[finiteCells]
    cellIndexCol = colIndex[finiteCells]
    cellVertices = np.zeros((nRow, nCol), dtype='int64') + np.max(cellID) + 1
    cellVertices[cellIndexRow, cellIndexCol] = cellID
    verticesRowCol = np.vstack((cellIndexRow, cellIndexCol)).T
    return(cellVertices, verticesRowCol)

#------------------------------------------------------------------------------

def createLandscapeGraph(costsurface):
    """
    Create a landscape graph for a cost-surface.
    
    Parameters
    ----------
    costsurface : array
        2D array of cost values.
    Returns
    -------
    landscapeGraph : graph
        A landscape graph as a SciPy sparse matrix in COOrdinate format in 
        which the vertices of the graph are the cost-surface cell indices.
    cellVertices : integer array
        2D array of same dimensions as the costsurface in which each cell has 
        a unique value with no data cells having the last value.
    verticesRowCol : integer array
        An n x 2 array, where each row is a non-null costsurface cell, and the 
        columns give the row and column locations of each cell.
    """
    nRow, nCol = costsurface.shape
    # Create cell to vertex mappings
    cellVertices, verticesRowCol = createCellVertices(costsurface)
    # Create empty arrays to store edge information
    i = np.empty((0), dtype='int64') # from cell ID
    j = np.empty((0), dtype='int64') # to cell ID
    w = np.empty((0), dtype='float') # edge weight
    # Extend cost-surface and cm by one cell around each edge
    costsurfaceExt = np.empty((nRow + 2, nCol + 2))
    costsurfaceExt[:] = np.nan
    costsurfaceExt[1:-1, 1:-1] = costsurface
    cellVerticesExt = np.zeros((nRow + 2, nCol + 2), dtype='int32')
    cellVerticesExt[1:-1, 1:-1] = cellVertices
    # Create list containing neighbour cell offsets and orth/diag multiplier
    orth = 1.0
    diag = np.sqrt(2)
    cellNeighOffsets = [(0, 0, 0, 0, diag), # Above left
                        (0, 0, 1, 1, orth), # Above
                        (0, 0, 2, 2, diag), # Above right
                        (1, 1, 0, 0, orth), # Left
                        (1, 1, 2, 2, orth), # Right
                        (2, 2, 0, 0, diag), # Below left
                        (2, 2, 1, 1, orth), # Below
                        (2, 2, 2, 2, diag)] # Below right
    # For the eight neighbouring cells
    for cellNeighOffset in cellNeighOffsets:
        offset1, offset2, offset3, offset4, cellDistance = cellNeighOffset
        # Extract shifted grids
        spcs = costsurfaceExt[offset1:nRow + offset2, offset3:nCol + offset4]
        spcm = cellVerticesExt[offset1:nRow + offset2, offset3:nCol + offset4]
        # Calculate the mean cost multipled by distance
        costDistance = ((spcs + costsurface) / 2) * cellDistance
        # Extract i, j, and d
        finiteCostDistance = np.isfinite(costDistance)
        i = np.append(i, np.extract(finiteCostDistance, cellVertices))
        j = np.append(j, np.extract(finiteCostDistance, spcm))
        w = np.append(w, np.extract(finiteCostDistance, costDistance))
    # Create a landscape graph as a sparse matrix
    landscapeGraph = sparse.coo_matrix((w, (i, j)))
    landscapeGraph = landscapeGraph.tolil()
    return(landscapeGraph, cellVertices, verticesRowCol)

#------------------------------------------------------------------------------

def vanDerCorput(n, base):
    """
    Calculate the number from a Van Der Corput sequence
        
    Parameters
    ----------
    n : integer
        The position of the number in the Van Der Corput sequence.
    base : integer
        The base used in the calculation.
    Returns
    -------
    out : float
        The Van Der Corput number value.
    """
    vdc, denom = 0.0, 1.0
    while n:
        denom *= base
        n, remainder = divmod(n, base)
        vdc += remainder / denom
    return(vdc)

#------------------------------------------------------------------------------

def haltonSample(array, n):
    """
    Create a subsample of landscape cells based on a Halton point pattern
        
    Parameters
    ----------
    n : integer
        The number of Halton points required.
    array : 2D array
        Cost-surafce array in which each cell has a cost value.
    Returns
    -------
    out : 2D array
        Row and column locations of Halton points.
    """    
    nRow, nCol = array.shape
    row = []
    col = [] 
    i = 0
    while len(row) < n:
        i = i + 1
        x = vanDerCorput(i, 2)
        y = vanDerCorput(i, 7)
        c = int(x * (nCol - 1))
        r = int((1 - y) * (nRow - 1))
        if array[r, c] == True:
            row.append(r)
            col.append(c)
    return(np.vstack((row, col)).T)

#------------------------------------------------------------------------------

def generateSites(nSites, siteRegions):
    """
    Generate a number of sites in landscape cells within a region matching the
    index value in landClasses
        
    Parameters
    ----------
    nSites : list
        The number of sites within the network. Either a list of one to locate
        sites across all finite costsurface cells, or a list of two or more to
        locate cells in different regions of the landscape as specificed by the
        siteRegions parameter.
    siteRegions: array
        2D array that provides the spatial region available for site 
        generation as an index that matches the values in nSites.
    Returns
    -------
    out : 2D array
        Row and column locations of site locations.
    """
    for i in np.arange(len(nSites)):
        siteRegion = siteRegions == i
        if i == 0:
            sites = haltonSample(siteRegion, nSites[i])
        else:
            sites = np.vstack((sites, haltonSample(siteRegion, nSites[i])))
    return(sites)      

#------------------------------------------------------------------------------

def orderSiteLocations(cells, landscapeGraph, cellVertices):
    """
    Order sites based on those with the largest Voronoi cells.
        
    Parameters
    ----------
    cells : 2D array
        Row and column locations of cells from which to order sites.
    landscapeGraph : graph
        A landscape graph as a SciPy sparse matrix in COOrdinate format in 
        which the vertices of the graph are the cost-surface cell indices.
    cellVertices : integer array
        2D array of same dimensions as the costsurface in which each cell has 
        a unique value with no data cells having the last value.
    Returns
    -------
    sites : 2D integer array
        The array gives the locations of sites ordered by least-cost Voronoi
        area.
    leastCostVoronoi : 2D array
        Array of the least-cost Voronoi diagram of the sites as an index of
        the sites.
    """ 
    # Get the graph indices for the candidate sites
    cellIndices = cellVertices[cells[:,0], cells[:,1]]
    # Calculate least-cost distance from the candidate sites
    dijkstraCosts = sparse.csgraph.dijkstra(csgraph = landscapeGraph, 
                                            indices = cellIndices,
                                            min_only = True,
                                            return_predecessors = True)
    # Create a Voronoi diagram and calculate area of each Voronoi cell
    voronoi = np.append(dijkstraCosts[2], np.nan)[cellVertices]   
    voronoiAreas = []
    leastCostVoronoi = np.empty(voronoi.shape)
    leastCostVoronoi[:] = np.nan
    for i in range(cells.shape[0]):
        cellIndex = cellVertices[cells[i,0], cells[i,1]]
        voronoiAreas.append(np.sum(voronoi == cellIndex)) 
        np.place(leastCostVoronoi, voronoi==cellIndex, i)
    # Extract the desired number of sites with the largest Voronoi cells
    voronoiSorted = np.flip(np.argsort(voronoiAreas))
    sites = cells[voronoiSorted,:]
    return(sites, leastCostVoronoi)

#------------------------------------------------------------------------------

def landEdgeCells(costsurface):
    """
    Identify finite cells on the edge of a landscape
        
    Parameters
    ----------
    array : 2D array
        2D array of cost values.
    Returns
    -------
    edgeCells : 2D array
        Row and column locations of landscape edge cells.
    edgeArray : 2D array
        2D array of same dimensions as costsurface that identifies the edge cells
    """  
    valued = 1 - np.isnan(costsurface)
    neighbourhood = np.array([[1,1,1],
                              [1,0,1],
                              [1,1,1]])
    filled = ndimage.binary_fill_holes(valued, structure=neighbourhood)
    edgeArray = (1 - ndimage.binary_erosion(filled, np.ones((3,3)))) * filled
    np.place(edgeArray, np.isinf(costsurface), 0)
    edges = np.where(edgeArray == 1)
    edgeCells = np.array([edges[0], edges[1]]).T
    return(edgeCells, edgeArray)

#------------------------------------------------------------------------------

def gabrielGraph(xy):
    """
    Identify neighbouring cells on the basis of a Gabriel graph
        
    Parameters
    ----------
    xy : 2D array
        Row and column locations of cells.
    Returns
    -------
    gabrielEdges : list
        Each tuple gives the indices of cells that are Gabriel neighbours.
    """  
    distanceMatrix = spatial.distance.squareform(spatial.distance.pdist(xy))
    delTri = spatial.Delaunay(xy)
    # Create Delaunay triagulation edge set
    delTriEdges = set()
    # For each Delaunay triangle,
    for s in range(delTri.nsimplex):
        # for each edge of the triangle, sort the vertices (to avoid duplicated 
        # edges being added to the set) and add to the edges set.
        edge = sorted([delTri.vertices[s,0], delTri.vertices[s,1]])
        delTriEdges.add((edge[0], edge[1]))
        edge = sorted([delTri.vertices[s,0], delTri.vertices[s,2]])
        delTriEdges.add((edge[0], edge[1]))
        edge = sorted([delTri.vertices[s,1], delTri.vertices[s,2]])
        delTriEdges.add((edge[0], edge[1]))
    delTriEdges = list(delTriEdges)
    delTriEdges.sort()
    # Subset out Gabriel edges
    gabrielEdges = []
    for delTriEdge in delTriEdges:
        ABdistanceSquared = distanceMatrix[delTriEdge[0], delTriEdge[1]] ** 2
        ABCdistanceSquared = (distanceMatrix[delTriEdge[0],] ** 2) + (distanceMatrix[delTriEdge[1],] ** 2)
        if np.min(ABCdistanceSquared) >= ABdistanceSquared:
            gabrielEdges.append(delTriEdge)
    return(gabrielEdges)

#------------------------------------------------------------------------------

def dijkstraShortestPath(endVertex, predecessors):
    """
    Find the Dijkstra shortest path
        
    Parameters
    ----------
    endVertex : integer
        The vertex that is at the end of the path.
    predecessors : 1D array
        For each vertex in the graph the preceeding vertex of the shortest path.
    Returns
    -------
    path : list
        Integers of the graph vertices that make up the shorest path.
    """  
    path = [endVertex]
    step = predecessors[path[-1]]  
    while step >= 0:
        path.append(step)    
        step = predecessors[path[-1]]
    return(path)

#------------------------------------------------------------------------------

def updateLandscapeGraphWithRoute(dijkstraPath, landscapeGraph, value):
    """
    Update a landscape graph given a route represented by a Dijkstra shortest 
    path
        
    Parameters
    ----------
    dijkstraPath : list
        Integers of the graph vertices that make up the shorest path.
    landscapeGraph : graph
        A landscape graph as a SciPy sparse matrix in COOrdinate format in 
        which the vertices of the graph are the cost-surface cell indices.
    value : real number
        The value to update the path edges with
    Returns
    -------
    landscapeGraph : graph
        A landscape graph as a SciPy sparse matrix in COOrdinate format in 
        which the vertices of the graph are the cost-surface cell indices.
    """  
    for step in range(len(dijkstraPath) - 1):
        oldValue = landscapeGraph[dijkstraPath[step], dijkstraPath[step + 1]]
        if value < oldValue:
            landscapeGraph[dijkstraPath[step], dijkstraPath[step + 1]] = value
            landscapeGraph[dijkstraPath[step + 1], dijkstraPath[step]] = value
    return(landscapeGraph)

#------------------------------------------------------------------------------

def calculateLCPs(sites, landscapeGraph, cellVertices, verticesRowCol, 
                  edgeCells, edgeArray, routeCost = 1):
    """
    Calculate least-cost paths as routes between sites.
        
    Parameters
    ----------
    sites : 2D integer array
        Gives the row and column locations of sites.
    landscapeGraph : graph
        A landscape graph as a SciPy sparse matrix in COOrdinate format in 
        which the vertices of the graph are the cost-surface cell indices.
    cellVertices : integer array
        2D array of same dimensions as the costsurface in which each cell has 
        a unique value with no data cells having the last value.
    verticesRowCol : integer array
        An n x 2 array, where each row is a non-null costsurface cell, and the 
        columns give the row and column locations of each cell.
    edgeCells : 2D array
        Row and column locations of landscape edge cells.
    edgeArray : 2D array
        2D array of same dimensions as costsurface that identifies the edge cells
    routeCost : float
        The replacement cost value for routes.
    Returns
    -------
    lcps : list
        A list containing 2D arrays of the row and column locations of routes.
    """  
    lcps = []
    # Get the landscape edge cell and site cell landscape graph vertices
    edgeVertices = cellVertices[edgeCells[:,0], edgeCells[:,1]]
    sitesVertices = cellVertices[sites[:,0], sites[:,1]]
    
    #----------------------------------------------------------------------
    
    # Create Gabriel graph to determine neighbouring sites
    if sites.shape[0] == 1:
        gabrielEdges = []
    if sites.shape[0] == 2:
        gabrielEdges = [(0, 1)]
    if sites.shape[0] >= 3:
        gabrielEdges = gabrielGraph(sites)
    # Redefine Gabriel graph edges in terms of landscape graph vertices
    gabrielVertices = []
    for gabrielEdge in gabrielEdges:
        gabrielVertices.append((sitesVertices[gabrielEdge[0]], sitesVertices[gabrielEdge[1]]))

    #----------------------------------------------------------------------
    
    # Connect each neighbour with a LCP if needed
    for neighbour in gabrielVertices:
        dijkstraResults = sparse.csgraph.dijkstra(csgraph = landscapeGraph, 
                                                  indices = neighbour[0],
                                                  return_predecessors = True,
                                                  min_only = True)
        # Create route
        dijkstraPath = dijkstraShortestPath(neighbour[1], dijkstraResults[1])
        # Update graph values based on route
        landscapeGraph = updateLandscapeGraphWithRoute(dijkstraPath, 
                                                       landscapeGraph, 
                                                       value = routeCost)
        # Create least-cost path as rows and columns of cells in path
        lcps.append(verticesRowCol[dijkstraPath,:])

    #----------------------------------------------------------------------
    
    # Connect each site to the landscape edge if it is not an edge cell
    for startVertex in sitesVertices:
        r, c = verticesRowCol[startVertex,]
        if edgeArray[r, c] == 0:
            # Connect to outer landscape
            dijkstraResults = sparse.csgraph.dijkstra(csgraph = landscapeGraph, 
                                                      indices = startVertex,
                                                      return_predecessors = True,
                                                      min_only = True)
            exitPoint = edgeVertices[np.argmin(dijkstraResults[0][edgeVertices])]
            # Create route
            dijkstraPath = dijkstraShortestPath(exitPoint, dijkstraResults[1])
            # Update graph values based on route
            landscapeGraph = updateLandscapeGraphWithRoute(dijkstraPath, 
                                                           landscapeGraph, 
                                                           value = routeCost)
            # Create least-cost path as rows and columns of cells in path
            lcps.append(verticesRowCol[dijkstraPath,:])
        
    #----------------------------------------------------------------------
            
    return(lcps)

#------------------------------------------------------------------------------

def createNetwork(costsurface, lcps):
    """
    Create an integer array of routes in each landscape cell.
        
    Parameters
    ----------
    costsurface : array
        2D array of cost values.
    lcps : list
        A list containing 2D arrays of the row and column locations of 
        least-cosy paths representing routes.
    Returns
    -------
    network : 2D array
        2D array of a landscape in which routes forming are network have a 
        value of one, non-network landscape cells have a value of zero, and any
        non-landscape cells have a value of nan.
    """ 
    # Create array for routes
    network = np.zeros(costsurface.shape)
    np.place(network, np.isnan(costsurface), np.nan)
    for lcp in lcps:
        for step in lcp:
            network[step[0], step[1]] = 1
    return(network)

#------------------------------------------------------------------------------

def leastCostNetwork(costsurface, nSites, siteRegions = None):
    """
    Calculate a network of sites and routes based on a cost-surface.
        
    Parameters
    ----------
    costsurface : array
        2D array of cost values.
    nSites : list
        The number of sites within the network. Either a list of one to locate
        sites across all finite costsurface cells, or a list of two or more to
        locate cells in different regions of the landscape as specificed by the
        siteRegions parameter.
    siteRegions: array
        Optional 2D array that provides the spatial regions available for site 
        generation as an index that matches the index of values in nSites.
    Returns
    -------
    network : 2D array
        Array that records the network in each landscape cell.
    sites : 2D integer array
        Gives the row and column locations of sites.
    routes : list of lists
        A list that contains 2D arrays of the row and column locations of routes.
    """ 
    # Create siteRegions if needed
    if siteRegions is None:
        siteRegions = np.empty(costsurface.shape)
        siteRegions[:] = np.nan
        np.place(siteRegions, np.isfinite(costsurface), 0)
    # Generate landscape graph
    landscapeGraph, cellVertices, verticesRowCol = createLandscapeGraph(costsurface)
    # Generate sites
    sites = generateSites(nSites, siteRegions)
    # Put those sites in order based on least-cost catchments
    sites, leastCostVoronoi = orderSiteLocations(sites, landscapeGraph, cellVertices)
    # Determine edge cells for inter-landscape connections
    edgeCells, edgeArray = landEdgeCells(costsurface)
    # Generate least-cost paths and a routes array
    routes = calculateLCPs(sites, landscapeGraph, cellVertices, verticesRowCol,
                           edgeCells, edgeArray)
    network = createNetwork(costsurface, routes)
    return(network, sites, routes)

#------------------------------------------------------------------------------
