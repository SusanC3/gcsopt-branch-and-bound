import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from gcsopt import GraphOfConvexSets

from gcsopt.branch_bound import shortest_path
import mosek

# feasible large gap: 15, 20, 50, 60
# feasible small gap: 0
# lots of edges: 25

# https://github.com/TobiaMarcucci/olrc-code/blob/main/examples/chapter5/footstep_planning.ipynb
# Generate random non-overlapping rectangles in the plane.
for s in range(14, 15):
    np.random.seed(s) # fixed seed for random number generator
    m = 22 # number of rectangles
    L = np.zeros((m, 2)) # lower left corners of rectangles
    U = np.zeros((m, 2)) # upper right corners of rectangles
    for i in range(m):
        while True: # loop until random box does not overlap with previous
            ci = np.random.uniform(0, 1, 2) # center
            d = np.abs(np.random.randn(2) / 10) # half diagonal
            L[i] = ci - d
            U[i] = ci + d
            overlap = False
            for j in range(i): # check if random box overlaps with previous
                lij = np.maximum(L[i], L[j]) # lower left corner of intersection
                uij = np.minimum(U[i], U[j]) # upper right corner of intersection
                if np.all(lij <= uij): # checks nonemptiness of intersection
                    overlap = True
                    break
            if not overlap:
                break

    # Start point is center of bottom left box.
    i0 = np.argmin(np.sum(L, axis=1))
    x0 = (L[i0] + U[i0]) / 2
    print("x0:", x0)

    # Goal point is center of top right box.
    iK = np.argmax(np.sum(U, axis=1))
    xK = (L[iK] + U[iK]) / 2
    print("xK:", xK)


    # Initialize graph.
    graph = GraphOfConvexSets()

    # Add vertices.
    for i in range(m):
        vertex = graph.add_vertex(i)

        # footstep location (x, y)
        r = vertex.add_variable(2)

        # Has to be within rectangle m
        vertex.add_constraints([r >= L[i], r <= U[i]])

        # Fix start and goal points.
        if (i == i0):
            vertex.add_constraint(r == x0)
        elif (i == iK):
            vertex.add_constraint(r == xK)



    # Add edges between connected rectangles.
    # OG problem doesn't have max step size, so in theory all rectangles connect to each other
    edge_cutoff = 0.4
    num_edges = 0
    edge_lengths = []
    for i in range(m):
        for j in range(m):
            if (i == j): continue # don't step twice in same rectangle
            v1 = graph.get_vertex(i)
            v2 = graph.get_vertex(j)

            ci = (L[i] + U[i]) / 2
            cj = (L[j] + U[j]) / 2
            length = np.linalg.norm(cj - ci)
            # if length > edge_cutoff: continue
            edge_lengths.append(length)

            # Impose max step size to limit edges
            # this doesn't really limit the step size in practice it just reduced the number of edges
            e = graph.add_edge(v1, v2)
            num_edges += 1

            # Cost is squared distance between r1 and r2
            e.add_cost(cp.sum_squares(v2.variables[0] - v1.variables[0]))
            
    print("edges:", num_edges)
    # Select source and target vertices.
    source = graph.get_vertex(i0)
    target = graph.get_vertex(iK)

    # Run followin code only if this file is executed directly, and not when it is
    # imported by other files.
    if __name__ == "__main__":
        print("seed:", s)

        # Solve problem.
        graph.solve_shortest_path(source, target, binary=False)
        print("Problem status:", graph.status)
        print("Optimal value:", graph.value)

        v_ints, e_ints = 0, 0
        for v in graph.vertices:
            if np.isclose(v.binary_variable.value, np.rint(v.binary_variable.value), atol=0.1): v_ints+=1
        for e in graph.edges:
            if np.isclose(e.binary_variable.value, np.rint(e.binary_variable.value), atol=0.1): e_ints+=1

        print("vertex percentage:", v_ints / len(graph.vertices) * 100)
        print("edge percentage:", e_ints / len(graph.edges) * 100)

        print()

        # graph.solve_shortest_path(source, target, binary=True)
        # print("Problem status:", graph.status)
        # print("Optimal value:", graph.value)

        # from gcsopt.branch_bound import shortest_path
        # shortest_path(graph, source, target, s)
        # print("Problem status:", graph.status)
        # print("Optimal value:", graph.value)

        # Plot solution.
        
        # plt.figure()
        # plt.gca().set_aspect('equal')
        # # for li, ui in zip(L, U):
        # #     rect = Rectangle(li, *(ui - li), ec='black', fc='mintcream')
        # #     plt.gca().add_patch(rect)
        # # plt.plot(*x.value.T, marker='o', ls='--')
        # # plt.xlim(-.2, 1.2)
        # # plt.ylim(-.2, 1.2)
        # graph.plot_2d()
        # graph.plot_2d_solution()
        # plt.show()