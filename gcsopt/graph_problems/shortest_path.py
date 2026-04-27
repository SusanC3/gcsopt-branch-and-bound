import cvxpy as cp
from gcsopt.graph_problems.utils import (define_variables,
    enforce_edge_programs, set_solution)

def bb_node_shortest_path(conic_graph, source, target, yv_fixed, ye_fixed, tol=1e-4, **kwargs):

    # Define variables.
    yv, zv, ye, ze, ze_tail, ze_head = define_variables(conic_graph, binary=False)

    # TODO could edge values be inferred from verties ?

    # Enforce edge costs and constraints.
    cost, constraints = enforce_edge_programs(conic_graph, ye, ze, ze_tail, ze_head)

    for i, edge in enumerate(conic_graph.edges):
        if i in ye_fixed:
            constraints += [ye[i] == ye_fixed[i]]

    # Enforce vertex costs and constraints.
    for i, vertex in enumerate(conic_graph.vertices):
        inc = conic_graph.incoming_edge_indices(vertex)
        out = conic_graph.outgoing_edge_indices(vertex)

        # Source vertex.
        if vertex.name == source.name:
            cost += vertex.cost_homogenization(zv[i], 1)
            constraints += [yv[i] == 1, 1 == sum(ye[out]), zv[i] == sum(ze_tail[out])]
            for k in inc:
                constraints += [ye[k] == 0, ze_head[k] == 0]

        # Target vertex.
        elif vertex.name == target.name:
            cost += vertex.cost_homogenization(zv[i], 1)
            constraints += [yv[i] == 1, 1 == sum(ye[inc]), zv[i] == sum(ze_head[inc])]
            for k in out:
                constraints += [ye[k] == 0, ze_tail[k] == 0]

        # All other vertices.
        else:
            cost += vertex.cost_homogenization(zv[i], yv[i])
            constraints += [ 
                yv[i] <= 1,
                yv[i] == sum(ye[inc]),
                yv[i] == sum(ye[out]),
                zv[i] == sum(ze_head[inc]),
                zv[i] == sum(ze_tail[out])]
            
        # branch and bound fixed variables
        if i in yv_fixed:
            constraints += [yv[i] == yv_fixed[i]]
           
    # Solve problem and set solution.
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(**kwargs)
    set_solution(conic_graph, prob, ye, ze, yv, zv, tol)




def shortest_path(conic_graph, source, target, binary, tol, **kwargs):

    # Define variables.
    yv, zv, ye, ze, ze_tail, ze_head = define_variables(conic_graph, binary)

    # Enforce edge costs and constraints.
    cost, constraints = enforce_edge_programs(conic_graph, ye, ze, ze_tail, ze_head)

    # Enforce vertex costs and constraints.
    # TODO COULD ENFORCE BINARY VAL FOR B&B
    for i, vertex in enumerate(conic_graph.vertices):
        inc = conic_graph.incoming_edge_indices(vertex)
        out = conic_graph.outgoing_edge_indices(vertex)

        # Source vertex.
        if vertex.name == source.name:
            cost += vertex.cost_homogenization(zv[i], 1)
            constraints += [yv[i] == 1, 1 == sum(ye[out]), zv[i] == sum(ze_tail[out])]
            for k in inc:
                constraints += [ye[k] == 0, ze_head[k] == 0]

        # Target vertex.
        elif vertex.name == target.name:
            cost += vertex.cost_homogenization(zv[i], 1)
            constraints += [yv[i] == 1, 1 == sum(ye[inc]), zv[i] == sum(ze_head[inc])]
            for k in out:
                constraints += [ye[k] == 0, ze_tail[k] == 0]

        # All other vertices.
        else:
            cost += vertex.cost_homogenization(zv[i], yv[i])
            constraints += [ # TODO ADD CONSTRAINTS HERE TO FIX VARIABLES
                yv[i] <= 1,
                yv[i] == sum(ye[inc]),
                yv[i] == sum(ye[out]),
                zv[i] == sum(ze_head[inc]),
                zv[i] == sum(ze_tail[out])]
           
    # Solve problem and set solution.
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(**kwargs)
    set_solution(conic_graph, prob, ye, ze, yv, zv, tol)
    # TODO SEND TO CONVEX RELAXATION SOVLER