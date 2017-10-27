# -*- coding: utf-8 -*-  # noqa
"""
Created in 2012-2016 by Kees Den Heijer (C.denheijer@tudelft.nl)

Refactored from 2017-present by Yahia Kala (ymzkala@gmail.com)
"""

# Check out http://www.norsys.com/onLineAPIManual/index.html
# Check out https://docs.python.org/3.6/library/ctypes.html


import os
from ctypes import (CDLL, c_char, c_char_p, c_void_p, c_int, c_double,
                    create_string_buffer, c_bool, POINTER, byref)
from numpy.ctypeslib import ndpointer
import numpy as np
import platform
import logging
import pdb
import types

from helpers import getnodedata, ccharp

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

# ------------------------------------------------------------------------
# Set argument and result types from Netica C API Functions.
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

cnetica.EnterFinding_bn.argtypes = [c_void_p, c_int]
cnetica.EnterFinding_bn.restype = None

cnetica.RetractNodeFindings_bn.argtypes = [c_void_p]
cnetica.RetractNodeFindings_bn.restype = None

cnetica.RetractNetFindings_bn.argtypes = [c_void_p]
cnetica.RetractNetFindings_bn.restype = None

cnetica.GetNodeNamed_bn.argtypes = [c_char_p, c_void_p]
cnetica.GetNodeNamed_bn.restype = c_void_p

cnetica.GetNodeNumberStates_bn.argtypes = [c_void_p]
cnetica.GetNodeNumberStates_bn.restype = c_int

cnetica.GetNodeLevels_bn.argtypes = [c_void_p]
cnetica.GetNodeLevels_bn.restype = c_void_p

cnetica.GetNodeStateName_bn.argtypes = [c_void_p, c_int]
cnetica.GetNodeStateName_bn.restype = c_char_p

zerochar_type = c_char * 0
cnetica.GetNetNodes2_bn.argtypes = [c_void_p, zerochar_type]
cnetica.GetNetNodes2_bn.restype = c_void_p

cnetica.LengthNodeList_bn.argtypes = [c_void_p]
cnetica.LengthNodeList_bn.restype = c_int

cnetica.GetNodeName_bn.argtypes = [c_void_p]
cnetica.GetNodeName_bn.restype = c_char_p

cnetica.GetNodeType_bn.argtypes = [c_void_p]
cnetica.GetNodeType_bn.restype = c_int

cnetica.NthNode_bn.argtypes = [c_void_p, c_int]
cnetica.NthNode_bn.restype = c_void_p

cnetica.GetNodeEquation_bn.argtypes = [c_void_p]
cnetica.GetNodeEquation_bn.restype = c_char_p

cnetica.GetNodeExpectedValue_bn.argtypes = [c_void_p, c_double_p,
                                            c_double_p, c_double_p]
cnetica.GetNodeExpectedValue_bn.restype = c_double

cnetica.SetNodeEquation_bn.argtypes = [c_void_p, c_char_p]
cnetica.SetNodeEquation_bn.restype = None

cnetica.SetNodeTitle_bn.argtypes = [c_void_p, c_char_p]
cnetica.SetNodeTitle_bn.restype = None

cnetica.SetNodeStateNames_bn.argtypes = [c_void_p, c_char_p]
cnetica.SetNodeStateNames_bn.restype = None

cnetica.SetNodeStateTitle_bn.argtypes = [c_void_p, c_int, c_char_p]
cnetica.SetNodeStateTitle_bn.restype = None

cnetica.EquationToTable_bn.argtypes = [c_void_p, c_int, c_bool, c_bool]
cnetica.EquationToTable_bn.restype = None

cnetica.NewNode_bn.argtypes = [c_char_p, c_int, c_void_p]
cnetica.NewNode_bn.restype = c_void_p

cnetica.DeleteNode_bn.argtypes = [c_void_p]
cnetica.DeleteNode_bn.restype = None

cnetica.ReviseCPTsByCaseFile_bn.argtypes = [c_void_p, c_void_p,
                                            c_int, c_double]
cnetica.ReviseCPTsByCaseFile_bn.restype = None

cnetica.AddLink_bn.argtypes = [c_void_p, c_void_p]
cnetica.AddLink_bn.restype = c_int

cnetica.DeleteLink_bn.argtypes = [c_int, c_void_p]
cnetica.DeleteLink_bn.restype = None

cnetica.ReverseLink_bn.argtypes = [c_void_p, c_void_p]
cnetica.ReverseLink_bn.restype = None

cnetica.ReverseLink_bn.argtypes = [c_int, c_void_p, c_void_p]
cnetica.ReverseLink_bn.restype = None

# The following Netica C functions were configured in the methods
# due to their input sizes being specific to the nodes.
# ---------------------------
# SetNodeProbs_bn
# EnterNodeLikelihood_bn
# GetNodeBeliefs_bn
# ------------------------------------------------------------------------
# End of argument and result types from Netica C API Functions.
# ------------------------------------------------------------------------


class NeticaNetwork:
    """Netica Bayesian Network Class Object."""

    def __init__(self, openfile=None, license=None, *args, **kwargs):
        """Initialize the Bayesian Network class object."""
        # Environment pointer. First arg is license.
        # (const char* license, environ_ns* env, const char* locn)
        self.env = cnetica.NewNeticaEnviron_ns(ccharp(license), None, None)
        # Most applications have only one environment, but we keep
        # this environment specific to the object instance.

        # Initialize environment.
        mesg = create_string_buffer(MESGLEN)
        # (environ_ns* env, char* mesg)
        self.res = cnetica.InitNetica2_bn(self.env, mesg)
        logger.info(mesg.value)

        # Create net.
        if openfile:
            # Read net from file.
            file_p = self._newstream(openfile)  # Create stream.
            # Net pointer.
            # (stream_ns* file, int options)
            self.net = cnetica.ReadNet_bn(file_p, REGULAR_WINDOW)
        else:
            # Create new empty net.
            # (const char* name, environ_ns* env)
            # TODO: Figure out significance of name arg.
            self.net = cnetica.NewNet_bn(ccharp('BayesNet'), self.env)
        self.setautoupdate()  # Auto update on by default.
    # --------------------------------------------------------------------
    # Methods involving file operations.
    # --------------------------------------------------------------------

    def closeenv(self):
        """
        Close environment.

        When a Netica environment is closed, no further operations can be
        done on the object.
        """
        mesg = create_string_buffer(MESGLEN)
        # (environ_ns* env, char* mesg)
        res = cnetica.CloseNetica_bn(self.env, mesg)
        logger.info(mesg.value)
        return res

    def savenet(self, name):
        """Create new stream and write Netica file."""
        file_p = self._newstream(name)
        # (const net_bn* net, stream_ns* file)
        cnetica.WriteNet_bn(self.net, file_p)

    def _newstream(self, name):
        """
        Create stream.

        Streams are used to prepare for file read or write.
        """
        name = create_string_buffer(ccharp(name))
        # (const char* filename, environ_ns* env, const char* access)
        return cnetica.NewFileStream_ns(name, self.env, None)  # file_p
    # --------------------------------------------------------------------
    # End of methods involving file operations.
    # --------------------------------------------------------------------

    # --------------------------------------------------------------------
    # Methods returning node list information.
    # --------------------------------------------------------------------
    def getnetnodes(self):
        """
        Get net nodes.

        Input a net_bn object. Returns a nodelist_bn object.
        """
        # GetNetNodes2_bn is not listed in the API manual, but GetNetNodes_bn
        # is. Looks like an update to the API that is undocumented.

        # (const net_bn* net, const char options[])
        return cnetica.GetNetNodes2_bn(self.net, zerochar_type())  # nl_p

    def lengthnodelist(self, nl_p=None):
        """
        Get number of nodes.

        Input is a node list object.
        """
        if not nl_p:
            nl_p = self.getnetnodes()
        # (const nodelist_bn* nodes)
        return cnetica.LengthNodeList_bn(nl_p)  # nnodes
    # --------------------------------------------------------------------
    # End of methods returning node list information.
    # --------------------------------------------------------------------

    # --------------------------------------------------------------------
    # Methods that create or delete nodes.
    # --------------------------------------------------------------------
    def newnode(self, name=None, num_states=0):
        """Create and return a new node."""
        # (const char* name, int num_states, net_bn* net)
        return cnetica.NewNode_bn(ccharp(name), num_states, self.net)

    def deletenode(self, node_p=None):
        """
        Remove node from net.

        Removes node from its net, and frees all resources (e.g. memory)
        it was using.
        """
        node_p = self.getnodenamed(node_p)  # Verify pointer.
        # (node_bn* node)
        cnetica.DeleteNode_bn(node_p)
    # --------------------------------------------------------------------
    # End of methods that create or delete nodes.
    # --------------------------------------------------------------------

    # --------------------------------------------------------------------
    # Methods that get values from nodes.
    # --------------------------------------------------------------------
    def getnodenamed(self, nodename):
        """
        Get node object/pointer by name.

        Node names are unique within a net, but node titles are not.
        Returns the nodename input if it is not a string (node_p checks)
        """
        # nodename = create_string_buffer(nodename)

        # Return the input if it is not a string (for node_p checks)
        if not isinstance(nodename, types.StringType):
            return nodename

        # (const char* name, const net_bn* net)
        node_p = cnetica.GetNodeNamed_bn(ccharp(nodename), self.net)
        if node_p is None:
            logger.warning('Node with name "%s" does not exist' % nodename)
        return node_p

    def getnodename(self, node_p):
        """Return the node name as string using node object as input."""
        # (const node_bn* node)
        return cnetica.GetNodeName_bn(node_p)  # name

    def nthnode(self, nl_p=None, index=0):
        """
        Get the node pointer.

        Returns the node pointer at position "index" within list of
        nodes "nl_p". Default for nl_p is the entire list in self.

        """
        # TODO: create a method called nthnodename
        if not nl_p:
            nl_p = self.getnetnodes()
        # (const nodelist_bn* nodes, int index)
        return cnetica.NthNode_bn(nl_p, index)  # node_p

    def getnodebeliefs(self, node_p=None):
        """
        Get node beliefs.

        Will default to using nodename if it exists.
        """
        node_p = self.getnodenamed(node_p)  # Verify pointer.

        nstates = self.getnodenumberstates(node_p)
        cnetica.GetNodeBeliefs_bn.argtypes = [c_void_p]
        cnetica.GetNodeBeliefs_bn.restype = ndpointer(
            'float32', ndim=1, shape=(nstates,), flags='C')
        # (node_bn* node)
        return cnetica.GetNodeBeliefs_bn(node_p)  # prob_bn

    def getnodenumberstates(self, node_p=None):
        """
        Get number of states in a node.

        Will accept node name or object.
        """
        node_p = self.getnodenamed(node_p)  # Verify pointer.

        # (const node_bn* node)
        return cnetica.GetNodeNumberStates_bn(node_p)  # nstates

    def getnodelevels(self, node_p=None):
        """
        Get node levels.

        Returns the list of numbers used to enable a continuous node
        to act discrete, or enables a discrete node to provide real-valued
        numbers. Levels are used to discretize continuous nodes, or to map
        from discrete nodes to real numbers.

        """
        node_p = self.getnodenamed(node_p)  # Verify pointer.

        node_type = self.getnodetype(node_p)
        nstates = self.getnodenumberstates(node_p)
        if node_type == 1:
            # CONTINUOUS_TYPE
            nlevels = nstates + 1
        else:
            # DISCRETE_TYPE
            nlevels = nstates

        print(nstates, nlevels)
        # TODO: Figure this out. Something might be off with T
        T = ndpointer('double', ndim=1, shape=(nlevels,), flags='C')
        # (const node_bn* node)
        res = cnetica.GetNodeLevels_bn(node_p)  # node_levels
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
        node_p = self.getnodenamed(node_p)  # Verify pointer.

        # GetNodeStateName_bn (	const node_bn*  node,   state_bn  state )
        return cnetica.GetNodeStateName_bn(node_p, state)  # node_levels

    def getnodetype(self, node_p):
        """
        Get node type.

        Returns DISCRETE_TYPE if the variable corresponding to node is
        discrete (digital), and CONTINUOUS_TYPE if it is continuous (analog)

        """
        node_p = self.getnodenamed(node_p)  # Verify pointer.

        # (const node_bn* node)
        return cnetica.GetNodeType_bn(node_p)  # node_type

    def getnodeequation(self, node_p):
        """
        Get the node equation.

        Returns a null terminated C character string which contains the
        equation associated with node, or the empty string (rather than NULL),
        if node does not have an equation.

        """
        node_p = self.getnodenamed(node_p)  # Verify pointer.

        # (const node_bn* node)
        return cnetica.GetNodeEquation_bn(node_p)  # equation

    def getnodeexpectedvalue(self, node_p):
        """
        Get the node expected value.

        Returns the expected real value of node, based on the current beliefs
        for node, and if std_dev is non-NULL, *std_dev will be set to the
        standard deviation. Returns UNDEF_DBL if the expected value couldn't
        be calculated.

        """
        node_p = self.getnodenamed(node_p)  # Verify pointer.

        # (node_bn* node, double* std_dev, double* x3, double* x4)
        stdev = c_double(9999)  # standard deviation
        x3 = c_double_p()
        x4 = c_double_p()

        # expected value. Also tweaks stdev.value
        expvalue = cnetica.GetNodeExpectedValue_bn(node_p,
                                                   byref(stdev), x3, x4)
        # TODO: test out stdev
        return expvalue, stdev.value

    def getnodeprobs(self, node_p=None, parent_states=None):
        """Get node probabilities."""
        node_p = self.getnodenamed(node_p)  # Verify pointer.

        parenttype = ndpointer('int', ndim=1, shape=len(20,), flags='C')
        cnetica.GetNodeProbs_bn.argtypes = [
            c_void_p,
            parenttype
        ]
        cnetica.GetNodeProbs_bn.restype = c_void_p

        pdb.set_trace()
        probs = cnetica.GetNodeProbs_bn(node_p, parent_states)
        # TODO: Test output of this.
        return probs
    # --------------------------------------------------------------------
    # End of methods that get values from nodes.
    # --------------------------------------------------------------------

    # --------------------------------------------------------------------
    # Methods that manipulate existing nodes.
    # --------------------------------------------------------------------
    def enternodelikelyhood(self, node_p=None, prob_bn=None):
        """
        Enters a likelihood finding for node.

        prob_bn is a vector containing one probability for each
        state of node.
        """
        node_p = self.getnodenamed(node_p)  # Verify pointer.

        nstates = self.getnodenumberstates(node_p)
        prob_bn = np.array(prob_bn, dtype='float32')
        # TODO: figure out configuration
        # Configuring ctypes inputs here because of array size spec.
        cnetica.EnterNodeLikelihood_bn.argtypes = [c_void_p, ndpointer(
            'float32', ndim=1, shape=(nstates,), flags='C')]
        cnetica.EnterNodeLikelihood_bn.restype = None
        # (node_bn* node, const prob_bn* likelihood)
        cnetica.EnterNodeLikelihood_bn(node_p, prob_bn)

    def enternodevalue(self, node_p=None, value=None):
        """Enter node finding as value."""
        node_p = self.getnodenamed(node_p)  # Verify pointer.

        # (node_bn* node, double value)
        cnetica.EnterNodeValue_bn(node_p, value)

    def enterfinding(self, node_p=None, state=None):
        """
        Enters the discrete finding state for node.

        This means that in the case currently being analyzed, node is known
        with certainty to have value state.
        """
        node_p = self.getnodenamed(node_p)  # Verify pointer.

        # (	node_bn*  node,   state_bn  state )
        cnetica.EnterFinding_bn(node_p, state)

    def retractnodefindings(self, node_p):
        """Retract all findings from node."""
        node_p = self.getnodenamed(node_p)  # Verify pointer.

        # (node_bn* node)
        cnetica.RetractNodeFindings_bn(node_p)

    def setnodeequation(self, node_p=None, eqn=None):
        """
        Set the node equation.

        This associates the equation eqn (a null terminated C character string)
        as the equation of the node.

        """
        node_p = self.getnodenamed(node_p)  # Verify pointer.

        # (node_bn* node, const char* eqn)
        cnetica.SetNodeEquation_bn(node_p, ccharp(eqn))

    def setnodetitle(self, node_p=None, title=None):
        """Set the node title. This can be non-unique."""
        node_p = self.getnodenamed(node_p)  # Verify pointer.

        # (node_bn* node, const char* title)
        cnetica.SetNodeTitle_bn(node_p, ccharp(title))

    def setnodestatenames(self, node_p=None, state_names=None):
        """
        Set the node state names.

        state_names is a single string with commas and/or whitespace
        separators. It can also be newlines. Make sure the number
        of names is consistent with the actual number of states.
        """
        node_p = self.getnodenamed(node_p)  # Verify pointer.

        # (node_bn* node, const char* state_names)
        cnetica.SetNodeStateNames_bn(node_p, ccharp(state_names))

    def setnodestatetitle(self, node_p=None, state=None, state_title=None):
        """Set the node state title(s)."""
        node_p = self.getnodenamed(node_p)  # Verify pointer.
        cnetica.SetNodeStateTitle_bn(node_p, state, ccharp(state_title))

    def setnodelevels(self, node_p=None, num_states=None, levels=None):
        """
        Set the node levels.

        If underlying variable is continuous, num_states = len(levels)-1
        If underlying variable is discrete, num_states = len(levels)

        On a fresh new node, it will assign the node type as continuous
        or discrete depending on the input.
        """
        node_p = self.getnodenamed(node_p)  # Verify pointer.

        # (node_bn* node, int num_states, const level_bn* levels)
        cnetica.SetNodeLevels_bn.argtypes = [c_void_p, c_int, ndpointer(
            'double', ndim=1, shape=(len(levels),), flags='C')]
        cnetica.SetNodeLevels_bn.restype = None
        cnetica.SetNodeLevels_bn(node_p, num_states, levels)

    def equationtotable(self, node_p=None, num_samples=None,
                        samp_unc=None, add_exist=None):
        """
        Build table from equation.

        Builds the CPT for node based on the equation that has been
        associated with it

        """
        node_p = self.getnodenamed(node_p)  # Verify pointer.

        # (node_bn* node, int num_samples, bool_ns samp_unc, bool_ns add_exist)
        cnetica.EquationToTable_bn(node_p, num_samples, samp_unc, add_exist)

    def setnodeprobs(self, node_p=None, parent_states=None, probs=None):
        """Set node probabilities."""
        node_p = self.getnodenamed(node_p)  # Verify pointer.

        parenttype = ndpointer('int', ndim=1, shape=len(parent_states,),
                               flags='C')

        cnetica.SetNodeProbs_bn.argtypes = [
            c_void_p,
            parenttype,
            ndpointer('float32', ndim=1, shape=(len(probs),), flags='C')
        ]

        cnetica.SetNodeProbs_bn.restype = None
        pdb.set_trace()
        cnetica.SetNodeProbs_bn(node_p, parent_states, probs)

    # --------------------------------------------------------------------
    # End of methods that manipulate existing nodes.
    # --------------------------------------------------------------------

    # --------------------------------------------------------------------
    # Methods involving net operations.
    # --------------------------------------------------------------------
    # def revisecptsbyfindings(self, nl_p=None, updating=0, degree=0):
    #     """
    #     Revise table with findings.
    #
    #     The current case (i.e., findings entered) is used to revise each
    #     node's conditional probabilities.
    #
    #     """
    #     env_p = self.newenv()
    #
    #     # (const nodelist_bn* nodes, int updating, double degree)
    #
    #     cnetica.ReviseCPTsByFindings_bn.argtypes = [c_void_p, c_void_p,
    #                                                 c_int, c_double]
    #
    #     cnetica.ReviseCPTsByFindings_bn.restype = None
    #
    #     cnetica.ReviseCPTsByFindings_bn(nl_p, updating, degree)

    def compilenet(self):
        """Compile net."""
        # (net_bn* net)
        cnetica.CompileNet_bn(self.net)

    def setautoupdate(self, auto_update=1):
        """Set the auto update feature."""
        # (net_bn* net, int auto_update)
        cnetica.SetNetAutoUpdate_bn(self.net, auto_update)

    def retractnetfindings(self):
        """

        Retracts all findings from all nodes.

        (i.e., the current case), except "constant" nodes
        (use retractnodefindings for that)

        """
        # (net_bn* net)
        cnetica.RetractNetFindings_bn(self.net)

    def revisecptsbycasefile(self, filename='', nl_p=None,
                             updating=0, degree=0):
        """
        Revise table by case file.

        Reads a file of cases from file and uses them to revise the
        experience and conditional probability tables (CPT) of each
        node in the input node list.

        """
        if not nl_p:
            nl_p = self.getnetnodes()

        file_p = self._newstream(filename)
        # (stream_ns* file, const nodelist_bn* nodes, int updating,
        #  double degree)
        cnetica.ReviseCPTsByCaseFile_bn(file_p, nl_p, updating, degree)
    # --------------------------------------------------------------------
    # End of methods involving net operations.
    # --------------------------------------------------------------------

    # --------------------------------------------------------------------
    # Methods involving links between nodes.
    # --------------------------------------------------------------------
    def addlink(self, parent=None, child=None):
        """
        Add a link from node parent to node child.

        Returns the index of the added link.
        """
        parent = self.getnodenamed(parent)  # verify pointer.
        child = self.getnodenamed(child)  # verify pointer.

        # (node_bn* parent, node_bn* child)
        return cnetica.AddLink_bn(parent, child)  # link_index

    def deletelink(self, link_index=1, child=None):
        """
        Remove link between parent and child.

        Removes the link going to child from the link_indexth
        parent node of child.
        """
        child = self.getnodenamed(child)  # Verify pointer.

        # (int link_index, node_bn* child)
        cnetica.DeleteLink_bn(link_index, child)

    def reverselink(self, parent=None, child=None):
        """Reverse the link between parent and child."""
        parent = self.getnodenamed(parent)  # verify pointer.
        child = self.getnodenamed(child)  # verify pointer.

        # (node_bn* parent, node_bn* child)
        return cnetica.ReverseLink_bn(parent, child)  # link_index

    def switchnodeparent(self, link_index=1, node_p=None, new_parent=None):
        """
        Switch the parent node for a given child node.

        Makes node new_parent a parent of node by replacing the existing
        parent at the link_indexth position, without modifying node's
        equation, or any of node's tables (such as CPT table or
        function table).
        """
        node_p = self.getnodenamed(node_p)
        new_parent = self.getnodenamed(new_parent)

        # (int link_index, node_bn* node, node_bn* new_parent)
        return cnetica.SwitchNodeParent_bn(link_index, node_p, new_parent)
    # --------------------------------------------------------------------
    # End of methods involving links between nodes.
    # --------------------------------------------------------------------

    # --------------------------------------------------------------------
    # Self-contained methods
    # --------------------------------------------------------------------
    getnodedata = getnodedata
    # --------------------------------------------------------------------
    # End of self-contained methods
    # --------------------------------------------------------------------
