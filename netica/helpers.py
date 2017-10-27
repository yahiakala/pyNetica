"""Helper functions, temporarily named this way."""
import numpy as np
import sys


def getnodedata(self, nl_p):
    """
    Get all data from all nodes listed in the input.

    Input is a nodelist_bn object.
    Output is a list of dictionaries.
    """
    nodedata = []
    nnodes = self.lengthnodelist(nl_p)
    for idx in range(nnodes):
        node_p = self.nthnode(nl_p, idx)
        nodenamestr = self.getnodename(node_p)
        nstates = self.getnodenumberstates(node_p)
        isdiscrete = self.getnodetype(node_p) != 1
        if isdiscrete:
            nodelevels = [self.getnodestatename(node_p, state=i)
                          for i in np.arange(nstates)]
            expval = (None, None)
        else:
            nodelevels = self.getnodelevels(node_p)
            expval = self.getnodeexpectedvalue(node_p)
        prob_bn = self.getnodebeliefs(node_p)
        nodedata.append({'name': nodenamestr,
                         'discrete': isdiscrete,
                         'expval': expval,
                         'nstates': nstates,
                         'levels': nodelevels,
                         'probs': prob_bn})
    return nodedata


def ccharp(inpstr):
    """Make sure input strings are c_char_p bytes objects."""
    # https://stackoverflow.com/questions/23852311/different-behaviour-of-ctypes-c-char-p  # noqa
    if sys.version_info < (3, 0) or 'bytes' in str(type(inpstr)):
        outstr = inpstr
    else:
        outstr = inpstr.encode('utf-8')
    return outstr
