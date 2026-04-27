import numpy as np
import matplotlib.pyplot as plt

def plot_bounds(bounds, fig_name):
    bounds[1] = np.where(bounds[1] >= 0, bounds[1], np.nan)
    plt.figure(figsize=(8, 2))
    plt.plot(bounds[0], bounds[2], lw=2, ls='-', label="best upper bound")
    plt.plot(
        [bounds[0, 0], bounds[0, -1]],
        [bounds[2, -1], bounds[2, -1]],
        lw=2, ls='--', label="optimal value")
    plt.plot(bounds[0], bounds[1], lw=2, ls=':', label="best lower bound")
    plt.xlim([bounds[0, 0], bounds[0, -1]])
    plt.xlabel("solver time (s)")
    plt.ylabel("objective value")
    plt.grid()
    plt.legend()
    plt.xlim(xmin=0)
    plt.savefig(fig_name + ".pdf", bbox_inches="tight")

# flight = np.load("flight_bounds.npy")
# plot_bounds(flight, "flight")
# print("Flight solve time:", flight[0, -1])

# bus = np.load("bus_bounds.npy")
# plot_bounds(bus, "bus")
# print("Bus solve time:", bus[0, -1])

# surveillance = np.load("surveillance_bounds.npy")
# plot_bounds(surveillance, "surveillance")
# print("Surveillance solve time:", surveillance[0, -1])

# cover = np.load("cover_bounds.npy")
# plot_bounds(cover, "cover")
# print("Cover solve time:", cover[0, -1])