import numpy as np
import heapq
import cvxpy as cp
import copy
from functools import partial

import graphviz as gv

from gcsopt import (GraphOfConvexSets, GraphOfConicSets)
from gcsopt.graph_problems.shortest_path import bb_node_shortest_path
from gcsopt.graph_problems.utils import (define_variables,
    enforce_edge_programs, set_solution)


DEBUG = 1



def makedot(root, optval):
   
    dot = gv.Digraph()

    q = [root]
    while len(q) > 0:
        n = heapq.heappop(q)
        label = "ORD: " + str(n.order) + "\nP: " + str(n.priority) + "\nUB: " + str(n.upper_bound) + "\nYV_FIXED: " + str(len(n.yv_fixed)) + "\nYE_FIXED: " +  str(len(n.ye_fixed))
        if len(n.prune_reason): label += "\nPruned: " + n.prune_reason
        elif np.all([c.order == -1 for c in n.children]): label += "\nChildren >= than UB"

        color = 'black'
        if np.isclose(n.value, optval): color = 'green'
        elif n.prune_reason == "int": color = 'orange'
        elif n.prune_reason != "" or np.all([c.order == -1 for c in n.children]): color = 'red'

        dot.node(str(n.order), label=label, color=color)

        for c in n.children:
            if c.order == -1: continue #not visited/pruned
            dot.edge(str(n.order), str(c.order), c.branch_info)
            heapq.heappush(q, c)

    return dot


class Node:
    def __init__(self, val=None, yv=[], ye=[], yv_fixed={}, ye_fixed={}, infeasible=False):
        self.value = val
        self.yv = copy.deepcopy(yv)
        self.yv_fixed = copy.deepcopy(yv_fixed)
        self.ye = copy.deepcopy(ye)
        self.ye_fixed = copy.deepcopy(ye_fixed)
        self.infeasible = infeasible

        self.children = []

        # for logging/debugging
        self.order = -1
        self.priority = -1
        self.upper_bound = -1
        self.branch_info = ""
        self.prune_reason = ""


    def set_solution(self, val, yv, ye, infeasible):
        self.value = val
        self.yv = copy.deepcopy(yv)
        self.ye = copy.deepcopy(ye)
        self.infeasible = infeasible

    def __lt__(self, other): return self 


# Utils
def make_child(n):
    return Node(val=n.value, yv_fixed=copy.deepcopy(n.yv_fixed), ye_fixed=copy.deepcopy(n.ye_fixed))

def solve_relaxation(n, conic_graph, conic_source, conic_target):
    bb_node_shortest_path(conic_graph, conic_source, conic_target, n.yv_fixed, n.ye_fixed)
    n.set_solution(conic_graph.value, 
                    [v.binary_variable for v in conic_graph.vertices], 
                    [e.binary_variable for e in conic_graph.edges],
                    conic_graph.status == cp.INFEASIBLE)



'''
Different node priority options
'''
def p_obj(n, conic_graph, conic_source, conic_target): # priority is n's objective value
    solve_relaxation(n, conic_graph, conic_source, conic_target)
    return n.value # want to explore most promising (lowest objective value)

def p_est(n, conic_graph, conic_source, conic_target): # priority is estimated based on rounded LP 
    temp = Node()
    
    solve_relaxation(n, conic_graph, conic_source, conic_target) # starting point
    if conic_graph.status == 'infeasible': return np.inf # low priority, this node will be discarded upon exploration

    for i, v in enumerate(n.yv):
        temp.yv_fixed[i] = round(v.value)
    for i, e in enumerate(n.ye):
        temp.ye_fixed[i] = round(e.value)

    solve_relaxation(temp, conic_graph, conic_source, conic_target) # replace with pseudocost for efficiency# replace with pseudocost for efficiency
    return temp.value

def p_Astar(n, conic_graph, conic_source, conic_target): # use A* cost
    solve_relaxation(n, conic_graph, conic_source, conic_target)
    h = n.value # "current" path cost
    # need ADMISSIBLE H

    return g + h

'''
Different branching options
returns list of children
'''
# branch on variables closest to value
def b_value(n, value, num_children=2, **kwargs):
    iv, best_v = -1, np.inf
    for i, v in enumerate(n.yv):
        if v.value is not None and i not in n.yv_fixed.keys() and abs(v.value-value) <= best_v:
            iv = i
            best_v = abs(v.value-value)

    ie, best_e = -1, np.inf
    for i, e in enumerate(n.ye):
        if e.value is not None and i not in n.ye_fixed.keys() and abs(e.value-value) <= best_e:
            ie = i
            best_e = abs(e.value-value)

    if (iv == -1 and ie == -1): return []
    
    children = []
    for i in range(num_children):
        children.append(make_child(n))

    #TODO how does this work with more than 2 children? aren't the only options 1/0?
    if best_v < best_e:
        children[0].branch_info = "yv[" + str(iv) + "]: " + str(n.yv[iv].value)[:7] + " --> 0"
        children[1].branch_info = "yv[" + str(iv) + "]: " + str(n.yv[iv].value)[:7] + " --> 1"
        print("best vertex", iv, "is this close to:", value, best_v)

        children[0].yv_fixed[iv] = 0
        children[1].yv_fixed[iv] = 1
    else:
        children[0].branch_info = "ye[" + str(ie) + "]: " + str(n.ye[ie].value)[:7] + " --> 0"
        children[1].branch_info = "ye[" + str(ie) + "]: " + str(n.ye[ie].value)[:7] + " --> 1"
        print("best edge", ie, "is this close to:", value, best_e)
        
        children[0].ye_fixed[ie] = 0
        children[1].ye_fixed[ie] = 1
    
    return children


# branch on variables closest to integer
def b_int(n, num_children=2, **kwargs):
    iv, best_v = -1, np.inf
    for i, v in enumerate(n.yv):
        if i in n.yv_fixed: continue
        if v.value is not None and i not in n.yv_fixed.keys() and min(v.value, 1-v.value) <= best_v:
            iv = i
            best_v = min(v.value, 1-v.value)

    ie, best_e = -1, np.inf
    for i, e in enumerate(n.ye):
        if i in n.ye_fixed: continue
        if e.value is not None and i not in n.ye_fixed.keys() and min(e.value, 1-e.value) <= best_e:
            ie = i
            best_e = min(e.value, 1-e.value) 

    if (iv == -1 and ie == -1): return []
    
    children = []
    for i in range(num_children):
        children.append(make_child(n))

    #TODO how does this work with more than 2 children? aren't the only options 1/0?
    if best_v < best_e:
        children[0].branch_info = "yv[" + str(iv) + "]: " + str(n.yv[iv].value)[:7] + " --> 0"
        children[1].branch_info = "yv[" + str(iv) + "]: " + str(n.yv[iv].value)[:7] + " --> 1"
      #  print("best vertex", iv, "with value", str(n.yv[iv].value)[:7], "is this close to integer:", best_v)

        children[0].yv_fixed[iv] = 0
        children[1].yv_fixed[iv] = 1
    else:
        children[0].branch_info = "ye[" + str(ie) + "]: " + str(n.ye[ie].value)[:7] + " --> 0"
        children[1].branch_info = "ye[" + str(ie) + "]: " + str(n.ye[ie].value)[:7] + " --> 1"
      #  print("best edge", ie, "with value", str(n.ye[ie].value)[:7], "is this close to integer:", best_e)
        
        children[0].ye_fixed[ie] = 0
        children[1].ye_fixed[ie] = 1
    
    return children


# combine branching on 0.5 with edge length information
def b_length(n, edge_lengths):
    best_score, ie= -np.inf, -1,
    for i, e in enumerate(n.ye):
        if i in n.ye_fixed: continue

        fractionality = min(n.ye[i].value, 1-n.ye[i].value)
        if fractionality < 1e-4: continue
        score = fractionality - 0.01*edge_lengths[i]

        if (score > best_score):
            #if (i in n.ye_fixed or np.isclose(n.ye[i].value, np.rint(n.ye[i].value), atol=0.1)): continue # don't bother
            best_score = score
            ie = i

    # hopefully constraints fix case where edge fixing and vertices don't line up
    children = [make_child(n), make_child(n)]
    children[0].ye_fixed[ie] = 0
    children[0].branch_info = "ye[" + str(ie) + "] (score " + str(best_score)[:7] + "): " + str(n.ye[ie].value)[:7]  + " --> 0"
    children[1].ye_fixed[ie] = 1
    children[1].branch_info = "ye[" + str(ie) + "] (score " + str(best_score)[:7] + "): " + str(n.ye[ie].value)[:7]  + " --> 1"
    print("best edge", ie, "has length", best_score, "unfixed value was", n.ye[ie].value)  

    return children


# strong branching
def b_strong(n, conic_graph, conic_source, conic_target):
    iv, best_v = -1, 0
    for i, v in enumerate(n.yv):
        if i in n.yv_fixed: continue # don't fix again
        # evaluate potential children
        c0 = make_child(n)
        c0.yv_fixed[i] = 0
        solve_relaxation(c0, conic_graph, conic_source, conic_target)
        if conic_graph.status == cp.INFEASIBLE: continue

        c1 = make_child(n)
        c1.yv_fixed[i] = 1
        solve_relaxation(c1, conic_graph, conic_source, conic_target)
        if conic_graph.status == cp.INFEASIBLE: continue

        if (n.value - c0.value)*(n.value - c1.value) > best_v:
            iv = i
            best_v = (n.value - c0.value)*(n.value - c1.value)

    ie, best_e = -1, 0
    for i, e in enumerate(n.ye):
        if i in n.ye_fixed: continue # don't fix again
        # evaluate potential children
        c0 = make_child(n)
        c0.ye_fixed[i] = 0
        solve_relaxation(c0, conic_graph, conic_source, conic_target)
        if conic_graph.status == cp.INFEASIBLE: continue # do not pick this child

        c1 = make_child(n)
        c1.ye_fixed[i] = 1
        solve_relaxation(c1, conic_graph, conic_source, conic_target)
        if conic_graph.status == cp.INFEASIBLE: continue

        if (n.value - c0.value)*(n.value - c1.value) > best_e:
            ie = i
            best_e = (n.value - c0.value)*(n.value - c1.value)

    if (iv == -1 and ie == -1): 
        return []
    
    children = []
    children.append(make_child(n))
    children.append(make_child(n))

    #TODO how does this work with more than 2 children? aren't the only options 1/0?
    if best_v > best_e:
        children[0].branch_info = "yv[" + str(iv) + "]: " + str(n.yv[iv].value)[:7] + " --> 0"
        children[1].branch_info = "yv[" + str(iv) + "]: " + str(n.yv[iv].value)[:7] + " --> 1"
        print("best vertex", iv, "has this product:", best_v)

        children[0].yv_fixed[iv] = 0
        children[1].yv_fixed[iv] = 1
    else:
        children[0].branch_info = "ye[" + str(ie) + "]: " + str(n.ye[ie].value)[:7] + " --> 0"
        children[1].branch_info = "ye[" + str(ie) + "]: " + str(n.ye[ie].value)[:7] + " --> 1"
        print("best edge", ie, "has this product:", best_e)
        
        children[0].ye_fixed[ie] = 0
        children[1].ye_fixed[ie] = 1
    
    return children


calc_priority = p_obj
branch_children = b_int


def shortest_path_conic(conic_graph, conic_source, conic_target, seed, heuristic_info, tol=1e-4):
    print("seed:", seed)
    global DEBUG
    global calc_priority # node with lowest priority will be selected at each iteration
    global branch_children

    upper_bound = np.inf 
    upper_bound_node = None

    root = None # in case this is useful

    # maintain tree data structure
    leaf_nodes = [] # not pruned or branched

    #0. Initialize: create initial node with no fixed variables
    bb_node_shortest_path(conic_graph, conic_source, conic_target, {}, {})
    root = Node(val=conic_graph.value,
            yv=[v.binary_variable for v in conic_graph.vertices], 
            ye=[e.binary_variable for e in conic_graph.edges],
            infeasible=conic_graph.status == cp.INFEASIBLE)


    heapq.heappush(leaf_nodes, (calc_priority(root, conic_graph, conic_source, conic_target), root))

    #1. termination check
    iter = 0
    while (iter < 200 and len(leaf_nodes) > 0):
        iter+=1
        #print("upper_bound =", upper_bound)
        #print("queue length:", len(leaf_nodes))

        #2. Choose next node, pop it from leaf_nodes
        priority, n = heapq.heappop(leaf_nodes)
        n.order = iter
        n.upper_bound = upper_bound
        n.priority = priority
       # print("priority:", priority)

     #   print(f"Node has {len(n.yv_fixed)} fixed vertices, {len(n.ye_fixed)} fixed edges")
      #  print(f"Fixed edges: {n.ye_fixed}")

        #3. solve the LP at this node, conditioned on fixed variables. 
            # unbounded --> stop, OG is unbounded
        if len(n.yv) == 0: # otherwise LP was already solved as part of node selection or branching
            solve_relaxation(n, conic_graph, conic_source, conic_target)
            
        yv_values = [v.value for v in n.yv]
        ye_values = [e.value for e in n.ye]

        #4. Prune (GOTO Step 1) IF:
            # LP is infeasible
        if (n.infeasible): 
        #    print("infeasible, continuing")
            n.prune_reason = "infeasible"
            continue
            # upper bound <= objective val
        if (upper_bound <= n.value):
         #   print("node's value worse than upper bound, continuing")
            n.prune_reason = ">= UB"
            continue
            # if all vars are integers and val <= upper bound, update & prune Nodes whose val >= UB
        if (np.allclose(np.array(yv_values), np.rint(np.array(yv_values)), atol=tol) 
                and np.allclose(np.array(ye_values), np.rint(np.array(ye_values)), atol=tol)):
          #  print("all vars are integers, continuing")
            n.prune_reason = "int"
            upper_bound = n.value
            upper_bound_node = n
            leaf_nodes = [leaf for leaf in leaf_nodes if not (leaf[1].value is not None and leaf[1].value >= upper_bound)]
          #  print("len leaf nodes after prune:", len(leaf_nodes))
            n.upper_bound = upper_bound
            continue
        

        #6. Branch. decrease feasible region by fixing >=1 addtl. variables, add >=1 new nodes
        n.children = branch_children(n)#, conic_graph, conic_source, conic_target)
        for child in n.children:
            heapq.heappush(leaf_nodes, (calc_priority(child, conic_graph, conic_source, conic_target), child))

      #  print()
    
    # set conic_graph variables to those of BB's solution
    if upper_bound_node is not None: 
        solve_relaxation(upper_bound_node, conic_graph, conic_source, conic_target)
        print("took", iter, "iters")
    else: print("didn't converge")
    if DEBUG: makedot(root, conic_graph.value).render('footstep_obj_int_' + str(seed))


def shortest_path(graph, source, target, seed, heuristic_info={}):
    conic_graph = graph.to_conic()
    conic_source = conic_graph.get_vertex(source.name)
    conic_target = conic_graph.get_vertex(target.name)
    shortest_path_conic(conic_graph, conic_source, conic_target, seed, heuristic_info)
    graph._set_solution(conic_graph)

