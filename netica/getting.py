"""
Getting methods that do not directly call the Netica C Library.

None of these methods are called by methods in netica.py.
"""

import pandas as pd
import itertools


def getnodedataframe(self, node_p):
    """Get node conditional probabilities (CPT) in a Pandas DataFrame."""
    node_p = self.getnodenamed(node_p)  # Verify pointer.
    states = tuple(self.getnodemetadata(node_p)['states'])

    parentmeta = self.getnodeparentmetadata(node_p)
    pnames = tuple([meta['name'] for meta in parentmeta])
    pstates = tuple([tuple(meta['states']) for meta in parentmeta])

    data = self.getnodeprobs(node_p)
    if len(parentmeta) > 0:
        indlist = list(itertools.product(*pstates))
        idx = pd.MultiIndex.from_tuples(indlist, names=pnames)
        df = pd.DataFrame(data=data, index=idx, columns=states)
    else:  # Return a pandas series.
        idx = states
        df = pd.Series(data=data, index=idx)

    return df


def getnodeparentmetadata(self, node_p=None):
    """Get metadata of all node parents."""
    node_p = self.getnodenamed(node_p)  # Verify pointer.
    parents = self.getnodeparents(node_p)
    return self.getnodelistmetadata(parents)


def getnodelistdata(self, nl_p=None):
    """
    Get all data from all nodes listed in the input.

    Input is a nodelist_bn object.
    Output is a list of dictionaries.
    """
    nodelistdata = []
    if not nl_p:
        nl_p = self.getnetnodes()

    nnodes = self.lengthnodelist(nl_p)
    for idx in range(nnodes):
        node_p = self.nthnode(nl_p, idx)
        nodelistdata.append(self.getnodedata(node_p))
    return nodelistdata


def getnodelistmetadata(self, nl_p=None):
    """
    Get all metadata from all nodes listed in the input.

    Input is a nodelist_bn object.
    Output is a list of dictionaries.
    """
    nodelistmetadata = []
    if not nl_p:
        nl_p = self.getnetnodes()

    nnodes = self.lengthnodelist(nl_p)
    for idx in range(nnodes):
        node_p = self.nthnode(nl_p, idx)
        nodelistmetadata.append(self.getnodemetadata(node_p))
    return nodelistmetadata


def getnodedata(self, node_p=None):
    """Get all the commonly used data from a specified node."""
    node_p = self.getnodenamed(node_p)  # Verify pointer.
    nodedata = self.getnodemetadata(node_p)
    if nodedata['discrete']:
        expval = (None, None)
    else:
        expval = self.getnodeexpectedvalue(node_p)

    prob_bn = self.getnodebeliefs(node_p)

    nodedata['expval'] = expval
    nodedata['probs'] = prob_bn
    return nodedata


def getnodemetadata(self, node_p=None):
    """Get node metadata."""
    node_p = self.getnodenamed(node_p)  # Verify pointer.
    nodenamestr = self.getnodename(node_p)
    nstates = self.getnodenumberstates(node_p)
    isdiscrete = self.getnodetype(node_p) != 1
    if isdiscrete:
        nodelevels = list(self.getnodelevels(node_p))

        nodestatenames = [self.getnodestatename(node_p, state=i)
                          for i in range(nstates)]

        if not nodestatenames[0]:
            nodestates = nodelevels
        else:
            nodestates = nodestatenames
    else:
        # First get array, then create list of strings, then tuple
        nodelevels = list(self.getnodelevels(node_p))
        nodestates = [str(nodelevels[i]) + ' to ' + str(nodelevels[i + 1])
                      for i in range(len(nodelevels) - 1)]
        nodestatenames = [self.getnodestatename(node_p, state=i)
                          for i in range(nstates)]

    nodemetadata = {'name': nodenamestr,
                    'discrete': isdiscrete,
                    'nstates': nstates,
                    'levels': nodelevels,
                    'states': nodestates,
                    'statenames': nodestatenames}
    return nodemetadata
