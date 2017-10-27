# -*- coding: utf-8 -*-  # noqa

"""Created in 2012-2016 by Kees Den Heijer (C.denheijer@tudelft.nl)"""

# Check out http://www.norsys.com/onLineAPIManual/index.html
# Check out https://docs.python.org/3.6/library/ctypes.html


import os
from ctypes import (CDLL, c_char, c_char_p, c_void_p, c_int, c_double,
                    create_string_buffer, c_bool, POINTER, byref)
from numpy.ctypeslib import ndpointer
import numpy as np
from numpy import array
import platform
import logging
import sys
import pdb

from helpers import getnodedata

logger = logging.getLogger(__name__)
c_double_p = POINTER(c_double)

# Constants
MESGLEN = 600
NO_VISUAL_INFO = 0
NO_WINDOW = 0x10
MINIMIZED_WINDOW = 0x30
REGULAR_WINDOW = 0x70

if 'window' in platform.system().lower():
    NETICA_LIB = os.path.join(os.path.split(__file__)[0], '..',
                              'lib', 'Netica.dll')
else:
    #    from ctypes import cdll
    NETICA_LIB = os.path.join(os.path.split(__file__)[0], '..',
                              'lib', 'libnetica.so')

# Check if the C Library is there.
if not os.path.exists(NETICA_LIB):
    # library Netica.dll or libnetica.so not found
    err = RuntimeError('"%s" NOT FOUND at\n %s' %
                       (os.path.split(NETICA_LIB)[-1], NETICA_LIB))
    logger.error(err)
    raise err
else:
    # Load the C Library.
    cnetica = CDLL(NETICA_LIB)

# Set argument and result types from Netica C Functions.
# ------------------------------------------------------------------------
cnetica.NewNeticaEnviron_ns.argtypes = [c_char_p, c_void_p, c_char_p]
cnetica.NewNeticaEnviron_ns.restype = c_void_p

cnetica.ReadNet_bn.argtypes = [c_void_p, c_int]
cnetica.ReadNet_bn.restype = c_void_p

cnetica.InitNetica2_bn.argtypes = [c_void_p, c_char_p]
cnetica.InitNetica2_bn.restype = c_int

cnetica.CloseNetica_bn.argtypes = [c_void_p, c_char_p]
cnetica.CloseNetica_bn.restype = c_int

cnetica.NewNet_bn.argtypes = [c_char_p, c_void_p]
cnetica.NewNet_bn.restype = c_void_p

cnetica.NewFileStream_ns.argtypes = [c_char_p, c_void_p, c_char_p]
cnetica.NewFileStream_ns.restype = c_void_p

cnetica.WriteNet_bn.argtypes = [c_void_p, c_void_p]
cnetica.WriteNet_bn.restype = None

cnetica.CompileNet_bn.argtypes = [c_void_p]
cnetica.CompileNet_bn.restype = None

cnetica.SetNetAutoUpdate_bn.argtypes = [c_void_p, c_int]
cnetica.SetNetAutoUpdate_bn.restype = None

cnetica.EnterNodeValue_bn.argtypes = [c_void_p, c_double]
cnetica.EnterNodeValue_bn.restype = None
# ------------------------------------------------------------------------


def ccharp(inpstr):
    """Make sure input strings are c_char_p bytes objects."""
    # https://stackoverflow.com/questions/23852311/different-behaviour-of-ctypes-c-char-p  # noqa
    if sys.version_info < (3, 0) or 'bytes' in str(type(inpstr)):
        outstr = inpstr
    else:
        outstr = inpstr.encode('utf-8')
    return outstr


class NeticaNetwork:
    """Netica Bayesian Network Class Object."""

    def __init__(self, openfile=None, license=None, *args, **kwargs):
        """Initialize the Bayesian Network class object."""
        # Environment pointer. First arg is license.
        self.env = cnetica.NewNeticaEnviron_ns(ccharp(license), None, None)
        # Initialize environment.
        mesg = create_string_buffer(MESGLEN)
        # (environ_ns* env, char* mesg)
        self.res = cnetica.InitNetica2_bn(self.env, mesg)
        logger.info(mesg.value)

        if openfile:
            # Read net from file.
            file_p = self._newstream(self.env, openfile)  # Create stream.
            # Net pointer.
            # (stream_ns* file, int options)
            self.net = cnetica.ReadNet_bn(file_p, REGULAR_WINDOW)
        else:
            # Create new empty net.
            # (const char* name, environ_ns* env)
            self.net = cnetica.NewNet_bn(ccharp('BayesNet'), self.env)

    def closeenv(self):
        """Close environment."""
        mesg = create_string_buffer(MESGLEN)
        # (environ_ns* env, char* mesg)
        res = cnetica.CloseNetica_bn(self.env, mesg)
        logger.info(mesg.value)
        return res

    def savenet(self, name):
        """Create new stream and write Netica file."""
        file_p = self._newstream(self.env, name)
        # (const net_bn* net, stream_ns* file)
        cnetica.WriteNet_bn(self.net, file_p)

    def _newstream(self, name):
        """
        Create stream.

        Streams are used to prepare for file read or write.
        """
        # (const char* filename, environ_ns* env, const char* access)
        name = create_string_buffer(ccharp(name))
        return cnetica.NewFileStream_ns(name, self.env, None)  # file_p

    def compilenet(self):
        """Compile net."""
        # (net_bn* net)
        cnetica.CompileNet_bn(self.net)

    def setautoupdate(self, auto_update=1):
        """Set the auto update feature."""
        # (net_bn* net, int auto_update)
        cnetica.SetNetAutoUpdate_bn(self.net, auto_update)

    def enternodevalue(self, node_p, value):
        """Enter node finding as value."""
        # (node_bn* node, double value)
        cnetica.EnterNodeValue_bn(node_p, value)

    def enterfinding(self, node_p, state):
        """

        Enters the discrete finding state for node.

        This means that in the case currently being analyzed, node is known
        with certainty to have value state.

        """
        # (	node_bn*  node,   state_bn  state )
        self.cnetica.EnterFinding_bn.argtypes = [c_void_p, c_int]
        self.cnetica.EnterFinding_bn.restype = None
        self.cnetica.EnterFinding_bn(node_p, state)

    def enternodelikelyhood(self, node_p, prob_bn):
        """

        Enters a likelihood finding for node.

        likelihood is a vector containing one probability for each
        state of node.

        """
        nstates = self.getnodenumberstates(node_p)
        prob_bn = array(prob_bn, dtype='float32')
        # (node_bn* node, const prob_bn* likelihood)
        self.cnetica.EnterNodeLikelihood_bn.argtypes = [c_void_p, ndpointer(
            'float32', ndim=1, shape=(nstates,), flags='C')]
        # (node_bn* node, const prob_bn* likelihood)
        self.cnetica.EnterNodeLikelihood_bn.restype = None
        self.cnetica.EnterNodeLikelihood_bn(node_p, prob_bn)

    def retractnodefindings(self, node_p):
        """Retract all findings from node."""
        # (node_bn* node)
        self.cnetica.RetractNodeFindings_bn.argtypes = [c_void_p]
        self.cnetica.RetractNodeFindings_bn.restype = None
        self.cnetica.RetractNodeFindings_bn(node_p)

    def retractnetfindings(self, net_p):
        """

        Retracts all findings from all nodes.

        (i.e., the current case), except "constant" nodes
        (use retractnodefindings for that)

        """
        # (net_bn* net)
        self.cnetica.RetractNetFindings_bn.argtypes = [c_void_p]
        self.cnetica.RetractNetFindings_bn.restype = None
        self.cnetica.RetractNetFindings_bn(net_p)

    def getnodenamed(self, nodename, net_p):
        """Get node by name."""
        # (const char* name, const net_bn* net)
        self.cnetica.GetNodeNamed_bn.argtypes = [c_char_p, c_void_p]
        self.cnetica.GetNodeNamed_bn.restype = c_void_p
        # nodename = create_string_buffer(nodename)
        node_p = self.cnetica.GetNodeNamed_bn(ccharp(nodename), net_p)
        if node_p is None:
            logger.warning('Node with name "%s" does not exist' % nodename)

        return node_p

    def getnodenumberstates(self, node_p):
        """Get number of states."""
        # (const node_bn* node)
        self.cnetica.GetNodeNumberStates_bn.argtypes = [c_void_p]
        self.cnetica.GetNodeNumberStates_bn.restype = c_int
        return self.cnetica.GetNodeNumberStates_bn(node_p)  # nstates

    def getnodelevels(self, node_p):
        """
        Get node levels.

        Returns the list of numbers used to enable a continuous node
        to act discrete, or enables a discrete node to provide real-valued
        numbers. Levels are used to discretize continuous nodes, or to map
        from discrete nodes to real numbers.

        """
        node_type = self.getnodetype(node_p)
        nstates = self.getnodenumberstates(node_p)
        if node_type == 1:
            # CONTINUOUS_TYPE
            nlevels = nstates + 1
        else:
            # DISCRETE_TYPE
            nlevels = nstates

        print(nstates, nlevels)
        # (const node_bn* node)
        self.cnetica.GetNodeLevels_bn.argtypes = [c_void_p]
        T = ndpointer('double', ndim=1, shape=(nlevels,), flags='C')
        self.cnetica.GetNodeLevels_bn.restype = c_void_p
        res = self.cnetica.GetNodeLevels_bn(node_p)  # node_levels
        if res:
            return np.array(T(res))
        else:
            return None

    def getnodestatename(self, node_p, state=0):
        """
        Get node state name.

        Given an integer index representing a state of node, this returns
        the associated name of that state, or the empty string
        (rather than NULL) if it does not have a name.

        Either all of the states have names, or none of them do.

        """
        # GetNodeStateName_bn (	const node_bn*  node,   state_bn  state )
        self.cnetica.GetNodeStateName_bn.argtypes = [c_void_p, c_int]
        self.cnetica.GetNodeStateName_bn.restype = c_char_p
        return self.cnetica.GetNodeStateName_bn(node_p, state)  # node_levels

    def getnodebeliefs(self, node_p):
        """Get node beliefs."""
        nstates = self.getnodenumberstates(node_p)
        # (node_bn* node)
        self.cnetica.GetNodeBeliefs_bn.argtypes = [c_void_p]
        self.cnetica.GetNodeBeliefs_bn.restype = ndpointer(
            'float32', ndim=1, shape=(nstates,), flags='C')

        return self.cnetica.GetNodeBeliefs_bn(node_p)  # prob_bn

    def getnetnodes(self, net_p):
        """
        Get net nodes.

        Input a net_bn object. Returns a nodelist_bn object.
        """
        # GetNetNodes2_bn is not listed in the API manual, but GetNetNodes_bn
        # is. Looks like an update to the API that is undocumented.
        # TODO: Figure out how to export a list of names from object.
        zerochar_type = c_char * 0
        # (const net_bn* net, const char options[])
        self.cnetica.GetNetNodes2_bn.argtypes = [c_void_p, zerochar_type]
        self.cnetica.GetNetNodes2_bn.restype = c_void_p
        return self.cnetica.GetNetNodes2_bn(net_p, zerochar_type())  # nl_p

    def lengthnodelist(self, nl_p):
        """Get number of nodes."""
        # (const nodelist_bn* nodes)
        self.cnetica.LengthNodeList_bn.argtypes = [c_void_p]
        self.cnetica.LengthNodeList_bn.restype = c_int
        return self.cnetica.LengthNodeList_bn(nl_p)  # nnodes

    def getnodename(self, node_p):
        """Return the node name as string."""
        # (const node_bn* node)
        self.cnetica.GetNodeName_bn.argtypes = [c_void_p]
        self.cnetica.GetNodeName_bn.restype = c_char_p
        return self.cnetica.GetNodeName_bn(node_p)  # name

    def getnodetype(self, node_p):
        """
        Get node type.

        Returns DISCRETE_TYPE if the variable corresponding to node is
        discrete (digital), and CONTINUOUS_TYPE if it is continuous (analog)

        """
        # (const node_bn* node)
        self.cnetica.GetNodeType_bn.argtypes = [c_void_p]
        self.cnetica.GetNodeType_bn.restype = c_int
        return self.cnetica.GetNodeType_bn(node_p)  # node_type

    def nthnode(self, nl_p, index):
        """
        Get the node pointer.

        Returns the node pointer at position "index" within list of
        nodes "nl_p"

        """
        # (const nodelist_bn* nodes, int index)
        self.cnetica.NthNode_bn.argtypes = [c_void_p, c_int]
        self.cnetica.NthNode_bn.restype = c_void_p
        return self.cnetica.NthNode_bn(nl_p, index)  # node_p

    def getnodeequation(self, node_p):
        """
        Get the node equation.

        Returns a null terminated C character string which contains the
        equation associated with node, or the empty string (rather than NULL),
        if node does not have an equation.

        """
        # (const node_bn* node)
        self.cnetica.GetNodeEquation_bn.argtypes = [c_void_p]
        self.cnetica.GetNodeEquation_bn.restype = c_char_p
        return self.cnetica.GetNodeEquation_bn(node_p)  # equation

    def getnodeexpectedvalue(self, node_p):
        """
        Get the node expected value.

        Returns the expected real value of node, based on the current beliefs
        for node, and if std_dev is non-NULL, *std_dev will be set to the
        standard deviation. Returns UNDEF_DBL if the expected value couldn't
        be calculated.

        """
        # (node_bn* node, double* std_dev, double* x3, double* x4)
        self.cnetica.GetNodeExpectedValue_bn.argtypes = [c_void_p, c_double_p,
                                                         c_double_p, c_double_p]

        self.cnetica.GetNodeExpectedValue_bn.restype = c_double
        stdev = c_double(9999)  # standard deviation
        x3 = c_double_p()
        x4 = c_double_p()

        # expected value
        expvalue = self.cnetica.GetNodeExpectedValue_bn(node_p,
                                                        byref(stdev), x3, x4)

        return expvalue, stdev.value

    def setnodeequation(self, node_p, eqn):
        """
        Set the node equation.

        This associates the equation eqn (a null terminated C character string)
        as the equation of node.

        """
        # (node_bn* node, const char* eqn)
        self.cnetica.SetNodeEquation_bn.argtypes = [c_void_p, c_char_p]
        self.cnetica.SetNodeEquation_bn.restype = None
        self.cnetica.SetNodeEquation_bn(node_p, ccharp(eqn))

    def setnodetitle(self, node_p, title):
        """Set the node title."""
        self.cnetica.SetNodeTitle_bn.argtypes = [c_void_p, c_char_p]
        self.cnetica.SetNodeTitle_bn.restype = None
        self.cnetica.SetNodeTitle_bn(node_p, ccharp(title))

    def setnodestatenames(self, node_p, state_names):
        """
        Set the node state names.

        state_names is a single string with commas and/or whitespace
        separators. It can also be newlines. Make sure the number
        of names is consistent with the actual number of states.
        """
        # (node_bn* node, const char* state_names)
        self.cnetica.SetNodeStateNames_bn.argtypes = [c_void_p, c_char_p]
        self.cnetica.SetNodeStateNames_bn.restype = None
        self.cnetica.SetNodeStateNames_bn(node_p, ccharp(state_names))

    def setnodestatetitle(self, node_p, state, state_title):
        """Set the node state title(s)."""
        self.cnetica.SetNodeStateTitle_bn.argtypes = [c_void_p, c_int, c_char_p]
        self.cnetica.SetNodeStateTitle_bn.restype = None
        self.cnetica.SetNodeStateTitle_bn(node_p, state, ccharp(state_title))

    def setnodelevels(self, node_p, num_states, levels):
        """Set the node levels."""
        # (node_bn* node, int num_states, const level_bn* levels)
        self.cnetica.SetNodeLevels_bn.argtypes = [c_void_p, c_int, ndpointer(
            'double', ndim=1, shape=(len(levels),), flags='C')]
        self.cnetica.SetNodeLevels_bn.restype = None
        self.cnetica.SetNodeLevels_bn(node_p, num_states, levels)

    def equationtotable(self, node_p, num_samples, samp_unc, add_exist):
        """
        Build table from equation.

        Builds the CPT for node based on the equation that has been
        associated with it

        """
        # (node_bn* node, int num_samples, bool_ns samp_unc, bool_ns add_exist)
        self.cnetica.EquationToTable_bn.argtypes = [c_void_p, c_int, c_bool, c_bool]
        self.cnetica.EquationToTable_bn.restype = None
        self.cnetica.EquationToTable_bn(node_p, num_samples, samp_unc, add_exist)

#    def revisecptsbyfindings(self, nl_p=None, updating=0, degree=0):

#        """

#        The current case (i.e., findings entered) is used to revise each
#        node's conditional probabilities.

#        """

#        env_p = self.newenv()

#        # (const nodelist_bn* nodes, int updating, double degree)


#        self.cnetica.ReviseCPTsByFindings_bn.argtypes = [c_void_p, c_void_p, c_int,
#                                                    c_double]

#        self.cnetica.ReviseCPTsByFindings_bn.restype = None

#        self.cnetica.ReviseCPTsByFindings_bn(nl_p, updating, degree)

    def revisecptsbycasefile(self, filename='', nl_p=None,
                             updating=0, degree=0):
        """
        Revise table by case file.

        Reads a file of cases from file and uses them to revise the
        experience and conditional probability tables (CPT) of each
        node in nodes.

        """
        env_p = self.newenv()
        file_p = self._newstream(env_p, filename)

        # (stream_ns* file, const nodelist_bn* nodes, int updating,
        #  double degree)
        self.cnetica.ReviseCPTsByCaseFile_bn.argtypes = [c_void_p, c_void_p,
                                                         c_int, c_double]

        self.cnetica.ReviseCPTsByCaseFile_bn.restype = None
        self.cnetica.ReviseCPTsByCaseFile_bn(file_p, nl_p, updating, degree)

    def setnodeprobs(self, node_p, parent_states, probs):
        """Set node probabilities."""
        parenttype = ndpointer('int', ndim=1, shape=len(parent_states,),
                               flags='C')

        self.cnetica.SetNodeProbs_bn.argtypes = [
            c_void_p,
            parenttype,
            ndpointer('float32', ndim=1, shape=(len(probs),), flags='C')
        ]

        self.cnetica.SetNodeProbs_bn.restype = None

        pdb.set_trace()
        self.cnetica.SetNodeProbs_bn(node_p, parent_states, probs)

    def getnodeprobs(self, node_p, parent_states):
        """Get node probabilities."""
        parenttype = ndpointer('int', ndim=1, shape=len(20,), flags='C')

        self.cnetica.GetNodeProbs_bn.argtypes = [
            c_void_p,
            parenttype
        ]

        self.cnetica.GetNodeProbs_bn.restype = c_void_p
        pdb.set_trace()
        probs = self.cnetica.GetNodeProbs_bn(node_p, parent_states)

        return probs

    def newnode(self, name=None, num_states=0, net_p=None):
        """Create and return a new node."""
        # (const char* name, int num_states, net_bn* net)
        self.cnetica.NewNode_bn.argtypes = [c_char_p, c_int, c_void_p]
        self.cnetica.NewNode_bn.restype = c_void_p
        return self.cnetica.NewNode_bn(ccharp(name), num_states, net_p)  # node_p

    def deletenode(self, node_p=None):
        """
        Remove node from net.

        Removes node from its net, and frees all resources (e.g. memory)
        it was using.

        """
        # (node_bn* node)
        self.cnetica.DeleteNode_bn.argtypes = [c_void_p]
        self.cnetica.DeleteNode_bn.restype = None
        self.cnetica.DeleteNode_bn(node_p)

    def addlink(self, parent=None, child=None):
        """
        Add a link from node parent to node child.

        Returns the index of the added link.
        """
        # (node_bn* parent, node_bn* child)

        self.cnetica.AddLink_bn.argtypes = [c_void_p, c_void_p]
        self.cnetica.AddLink_bn.restype = c_int
        return self.cnetica.AddLink_bn(parent, child)  # link_index

    def deletelink(self, link_index=1, child=None):
        """
        Remove link between parent and child.

        Removes the link going to child from the link_indexth
        parent node of child.
        """
        # (int link_index, node_bn* child)

        self.cnetica.DeleteLink_bn.argtypes = [c_int, c_void_p]
        self.cnetica.DeleteLink_bn.restype = None
        self.cnetica.DeleteLink_bn(link_index, child)

    def reverselink(self, parent=None, child=None):
        """Reverse the link between parent and child."""
        # (node_bn* parent, node_bn* child)

        self.cnetica.ReverseLink_bn.argtypes = [c_void_p, c_void_p]
        self.cnetica.ReverseLink_bn.restype = None
        return self.cnetica.ReverseLink_bn(parent, child)  # link_index

    def switchnodeparent(self, link_index=1, node_p=None, new_parent=None):
        """
        Switch the parent node for a given child node.

        Makes node new_parent a parent of node by replacing the existing
        parent at the link_indexth position, without modifying node's
        equation, or any of node's tables (such as CPT table or
        function table).
        """
        # (int link_index, node_bn* node, node_bn* new_parent)
        self.cnetica.ReverseLink_bn.argtypes = [c_int, c_void_p, c_void_p]
        self.cnetica.ReverseLink_bn.restype = None
        return self.cnetica.SwitchNodeParent_bn(link_index, node_p, new_parent)

    getnodedata = getnodedata
