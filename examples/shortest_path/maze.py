import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from maze_utils import Maze, make_gap_maze
from gcsopt import GraphOfConvexSets

from gcsopt.branch_bound import shortest_path

# Create maze.
maze_side = 10
knock_downs = 2

for random_seed in range(4, 5):
    maze = Maze(maze_side, maze_side, random_seed)
    maze.knock_down_walls(knock_downs)

    # Start and goal points.
    start = np.array([0.5, 0])
    goal = np.array([maze_side - 0.5, maze_side])

    # Initialize graph.
    graph = GraphOfConvexSets()

    # Add vertices.
    for i in range(maze_side):
        for j in range(maze_side):
            vertex = graph.add_vertex((i, j))

            # Trajectory start and end point within cell.
            x = vertex.add_variable((2, 2))

            # Minimize distance traveled within cell.
            vertex.add_cost(cp.norm2(x[1] - x[0]))

            # Constrain trajectory segment in cell.
            l = np.array([i, j])
            u = l + 1
            vertex.add_constraints([x[0] >= l, x[0] <= u])
            vertex.add_constraints([x[1] >= l, x[1] <= u])

            # Fix start and goal points.
            if all(l == 0):
                vertex.add_constraint(x[0] == start)
            elif all(u == maze_side):
                vertex.add_constraint(x[1] == goal)

    # Add edges between communicating cells.
    for i in range(maze_side):
        for j in range(maze_side):
            cell = maze.get_cell(i, j)
            tail = graph.get_vertex((i, j))
            for direction, d in maze.directions.items():
                if not cell.walls[direction]:
                    head = graph.get_vertex((i + d[0], j + d[1]))
                    edge = graph.add_edge(tail, head)

                    # Enforce trajectory continuity.
                    end_tail = tail.variables[0][1]
                    start_head = head.variables[0][0]
                    edge.add_constraint(end_tail == start_head) 

    # Select source and target vertices.
    source = graph.get_vertex((0, 0))
    target = graph.get_vertex((maze_side - 1, maze_side - 1))

    # Run followin code only if this file is executed directly, and not when it is
    # imported by other files.
    if __name__ == "__main__":
        print("Seed:", random_seed)

        # Solve problem.
        graph.solve_shortest_path(source, target, binary=False)
        print("Problem status:", graph.status)
        print("Optimal value:", graph.value)

        # v_ints, e_ints = 0, 0
        # for v in graph.vertices:
        #     if np.isclose(v.binary_variable.value, np.rint(v.binary_variable.value), atol=0.1): v_ints+=1
        # for e in graph.edges:
        #     if np.isclose(e.binary_variable.value, np.rint(e.binary_variable.value), atol=0.1): e_ints+=1

        # print("vertex percentage:", v_ints / len(graph.vertices) * 100)
        # print("edge percentage:", e_ints / len(graph.edges) * 100)

        # plt.figure()
        # maze.plot()
        # for edge in graph.edges:
        #     head, tail = edge.head, edge.tail

        #     zh = head.binary_variable.value
        #     if zh > 1e-3:
        #         pt = head.variables[0].value[0]
        #         plt.scatter(pt[0], pt[1], c=[[zh]], cmap='hot_r', vmin=0, vmax=1,
        #                     s=30, edgecolors='none', zorder=3)
        #         plt.plot(*vertex.variables[0].value.T, 'b-', c='blue')

        #     zt = tail.binary_variable.value
        #     if zt > 1e-3:
        #         pt = head.variables[0].value[0]
        #         plt.scatter(pt[0], pt[1], c=[[zt]], cmap='hot_r', vmin=0, vmax=1,
        #                     s=30, edgecolors='none', zorder=3)
                
        #     e = edge.binary_variable.value
        #     cmap = plt.colormaps['hot_r']
        #     if e > 1e-3: 
        #         head_pt = head.variables[0].value[0]
        #         tail_pt = tail.variables[0].value[0]
        #         plt.plot([head_pt[0], tail_pt[0]], [head_pt[1], tail_pt[1]], 'b--', c=cmap(e))

    

        # plt.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0,1), cmap=plt.cm.hot_r),
        #             ax=plt.gca(), label='Relaxed binary value')
        # plt.show()
        # plt.clf()


        graph.solve_shortest_path(source, target, binary=True)
        print("Problem status:", graph.status)
        print("Optimal value:", graph.value)

        print()

        from gcsopt.branch_bound import shortest_path
        shortest_path(graph, source, target, random_seed)
        print("Problem status:", graph.status)
        print("Optimal value:", graph.value)


        # plt.figure()
        # maze.plot()
        # for vertex in graph.vertices:
        #     if np.isclose(vertex.binary_variable.value, 1):
        #         plt.plot(*vertex.variables[0].value.T, 'b--')
        # plt.show()


