import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from gcsopt import GraphOfConvexSets

from gcsopt.branch_bound import shortest_path

# https://github.com/TobiaMarcucci/olrc-code/blob/main/examples/chapter5/footstep_planning.ipynb
# Generate random non-overlapping rectangles in the plane.
np.random.seed(0) # fixed seed for random number generator
m = 30 # number of rectangles
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
for i in range(m):
    for j in range(m):
        if (i == j): continue # don't step twice in same rectangle
        v1 = graph.get_vertex(i)
        v2 = graph.get_vertex(j)

        e = graph.add_edge(v1, v2)

        # Cost is squared distance between r1 and r2
        e.add_cost(cp.sum_squares(v2.variables[0] - v1.variables[0]))
        

# Select source and target vertices.
source = graph.get_vertex(i0)
target = graph.get_vertex(iK)

# Run followin code only if this file is executed directly, and not when it is
# imported by other files.
if __name__ == "__main__":

    # Solve problem.
    graph.solve_shortest_path(source, target, binary=False)
    print("Problem status:", graph.status)
    print("Optimal value:", graph.value)

    graph.solve_shortest_path(source, target, binary=True)
    print("Problem status:", graph.status)
    print("Optimal value:", graph.value)

    # from gcsopt.branch_bound import shortest_path
    # shortest_path(graph, source, target)

    # Plot solution.
    
    plt.figure()
    plt.gca().set_aspect('equal')
    # for li, ui in zip(L, U):
    #     rect = Rectangle(li, *(ui - li), ec='black', fc='mintcream')
    #     plt.gca().add_patch(rect)
    # plt.plot(*x.value.T, marker='o', ls='--')
    # plt.xlim(-.2, 1.2)
    # plt.ylim(-.2, 1.2)
    graph.plot_2d()
    graph.plot_2d_solution()
    plt.show()