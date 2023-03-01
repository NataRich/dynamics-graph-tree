import numpy as np
from algo import bruteforce_search
from tree import Interval, Linear, PiecewiseLinear, Branch, Tree, Plotter


def parse_itinerary(itinerary: list[Branch]):
    y_labels = [b.label() for b in itinerary]
    y_values = [b.order() for b in itinerary]
    x_values = np.arange(1, len(itinerary) + 1, 1)
    return x_values, y_values, y_labels


##############################
# Invariants
# - Branch a is mapped to branch b by the identity function.
# - Branch c is divided into two subintervals, c1 and c2.
##############################
interval_a = Interval(0, 10)  # [0, 10)
interval_b = Interval(10, 20)  # [10, 20)
interval_c1 = Interval(20, 25)  # [20, 25)
interval_c2 = Interval(25, 30)  # [25, 30)

branch_a = Branch('a', interval_a, 0)
branch_b = Branch('b', interval_b, 1)
branch_c1 = Branch('c1', interval_c1, 2)
branch_c2 = Branch('c2', interval_c2, 3)

f1 = Linear(1, 10, interval_a)  # f(x) = x + 10
f2 = Linear(1, 10, interval_b)  # f(x) = x + 10

plt = Plotter('Itinerary', 'Branches', 'Steps')
##############################


##############################
# Variants
# - Number of iterations.
# - How branch c1 is mapped to branch a.
# - How branch c2 is mapped to branch b.
##############################
def run_one(x: float, plot=True):
    return run_batch(x, x + 1, 1, plot=plot)


def run_batch(start: float, stop: float, step: float, plot=False):
    N_ITER = 150

    f3_1 = Linear(2, -40, interval_c1)  # f(x) = 2x - 40
    f3_2 = Linear(2, -40, interval_c2)  # f(x) = 2x - 40
    f = PiecewiseLinear([f1, f2, f3_1, f3_2])

    t = Tree(f, [branch_a, branch_b, branch_c1, branch_c2])

    n_aperiodic = 0  # number of aperiodic itineraries
    aperiodic_pts = []  # the starting points of the aperiodic itinerary

    for s in np.arange(start, stop, step):
        t.iter(s, N_ITER)
        x_values, y_values, y_labels = parse_itinerary(t.itinerary())
        prefix, cycle = bruteforce_search(''.join(y_labels))
        print(f'Itinerary from {format(s, ".1f")}: ({prefix}, {cycle})')
        if len(cycle) == 0:
            n_aperiodic += 1
            aperiodic_pts.append(s)
        if plot:
            plt.plot(x_values, y_values, ticks=t.orders(), labels=t.labels(), title=f'Itinerary from {s}')
            plt.plot(x_values, t.values(), title=f'Valued itinerary from {s}')

    print(f'Number of aperiodic itineraries: {n_aperiodic}')
    print(f'Aperiodic starting points: {aperiodic_pts}')
##############################


if __name__ == '__main__':
    # To enable plotting in batch mode, do `run_batch(x, y, z, plot=True)`
    # To disable plotting in single mode, do `run_batch(x, plot=False)`

    run_batch(0.0, 30.0, 0.1)
    # run_batch(0.0, 10, 0.5, plot=True)
    # run_batch(0.0, 10, 0.25)
    # run_batch(0.0, 10.0, 0.2)
    # run_batch(0.0, 10, 0.3)
    # run_one(4.8, plot=False)
