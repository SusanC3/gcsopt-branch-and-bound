import numpy as np
import heapq
import cvxpy as cp
import copy

from gcsopt import (GraphOfConvexSets, GraphOfConicSets)
from gcsopt.graph_problems.shortest_path import bb_node_shortest_path
from gcsopt.graph_problems.utils import (define_variables,
    enforce_edge_programs, set_solution)



class Node:
    def __init__(self):
        self.value = None
        self.yv = []
        self.yv_fixed = {}
        self.ye = []
        self.ye_fixed = {}

        self.children = []
    
    def __init__(self, val, yv, ye):
        self.yv_fixed = {}
        self.ye_fixed = {}
        self.children = []
        self.set_solution(val, yv, ye)

    def set_solution(self, val, yv, ye):
        self.value = val
        self.yv = yv
        self.ye = ye

    def __lt__(self, other): return self #TODO with proper priority this should never be used


def shortest_path_conic(conic_graph, conic_source, conic_target, tol=1e-4):

    upper_bound = np.inf 
    lower_bound = -np.inf 

    root = None # in case this is useful

    # maintain tree data structure
    leaf_nodes = [] # not pruned or branched

    #0. Initialize: create initial node with no fixed variables
    bb_node_shortest_path(conic_graph, conic_source, conic_target, {}, {})
    heapq.heappush(leaf_nodes, 
                   (conic_graph.value, Node(conic_graph.value, # TODO different priority strategies
                        [v.binary_variable for v in conic_graph.vertices], 
                        [e.binary_variable for e in conic_graph.edges])))

    while (True):
        print("upper_bound =", upper_bound)
        print("queue length:", len(leaf_nodes))
        #1. Termination check: if leaf_nodes is empty 
        if (len(leaf_nodes) == 0):
            return #solution already set, anything else needs to be done?

        #2. Choose next node, pop it from leaf_nodes
        priority, n = heapq.heappop(leaf_nodes)
        print("priority/value:", priority)

        #3. solve the LP at this node, conditioned on fixed variables. 
            # unbounded --> stop, OG is unbounded
        bb_node_shortest_path(conic_graph, conic_source, conic_target, n.yv_fixed, n.ye_fixed)
        if (conic_graph.status == cp.UNBOUNDED):
            print("Problem is unbounded!")
            return
        
        n.set_solution(conic_graph.value, # TODO give nodes custom priority
                        [v.binary_variable for v in conic_graph.vertices], 
                        [e.binary_variable for e in conic_graph.edges])
        
        yv_values = [v.value for v in n.yv]
        ye_values = [e.value for e in n.ye]

        #4. Prune (GOTO Step 1) IF:
            # LP is infeasible
        if (conic_graph.status == cp.INFEASIBLE): continue
            # upper bound <= objective val
        if (upper_bound <= n.value): continue
            # if all vars are integers and val <= upper bound, update & prune Nodes whose val >= UB
        if (np.allclose(np.array(yv_values), np.rint(np.array(yv_values)), atol=tol) 
                and np.allclose(np.array(ye_values), np.rint(np.array(ye_values)), atol=tol)):
            upper_bound = n.value
            leaf_nodes = [leaf for leaf in leaf_nodes if not (leaf[1].value >= upper_bound)]
            continue
        

        #6. Branch. decrease feasible region by fixing >=1 addtl. variables, add >=1 new nodes
        #TODO allow multiple strategies. for now it picks closest to 0.5
        iv, best_v = -1, np.inf
        for i, v in enumerate(n.yv):
            if v.value is not None and abs(v.value-0.5) <= best_v:
                iv = i
                best_v = abs(v.value-0.5)

        ie, best_e = -1, np.inf
        for i, e in enumerate(n.ye):
            if e.value is not None and abs(e.value-0.5) <= best_e:
                ie = i
                best_e = abs(e.value-0.5)

        #TODO allow variable number of children
        child0 = copy.deepcopy(n)
        child1 = copy.deepcopy(n)

        import pdb
        pdb.set_trace()

        if best_v < best_e:
            child0.yv_fixed[iv] = 0
            child1.yv_fixed[iv] = 1
            print("best vertex", iv, "is this close to 0.5:", best_v)
        else:
            child0.ye_fixed[ie] = 0
            child1.ye_fixed[ie] = 1
            print("best edge", ie, "is this close to 0.5:", best_e)

        #TODO could do different priority
        heapq.heappush(leaf_nodes, (priority, child0))
        heapq.heappush(leaf_nodes, (priority, child1))
 

def shortest_path(graph, source, target):
    conic_graph = graph.to_conic()
    conic_source = conic_graph.get_vertex(source.name)
    conic_target = conic_graph.get_vertex(target.name)
    shortest_path_conic(conic_graph, conic_source, conic_target)
    graph._set_solution(conic_graph)