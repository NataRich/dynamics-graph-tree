from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt


def mutually_disjoint(intervals: list[Interval]):
    new_intervals = sorted(intervals, key=lambda x: x[0])
    for i in range(len(new_intervals) - 1):
        if new_intervals[i].overlaps(new_intervals[i + 1]):
            return False
    return True


class Interval:
    """
    A class that represents a conventional real-valued interval.

    Examples
    ----------
    (0, 1) <=> Interval(0, 1, incl_l=False)
    [0, 10] <=> Interval(0, 10, incl_r=True)
    [0, inf) <=> Interval(0, np.inf) <=> Interval(0, np.infty)
    3 <=> Interval(3, 3, incl_r=True)
    """
    def __init__(self, start: float, stop: float, incl_l: bool = True, incl_r: bool = False):
        """
        Creates a real-valued interval.

        :param start: any real number, np.inf, or np.infty
        :param stop: any real number, np.inf, or np.infty
        :param incl_l: True if the left bound should be inclusive; False, otherwise (True by default)
        :param incl_r: True if the right bound should be inclusive; False, otherwise (False by default)
        """
        if start is None or stop is None or incl_l is None or incl_r is None:
            raise ValueError('Interval parameters must not be None')
        if start > stop:
            raise ValueError(f'Invalid interval from {start} to {stop}')
        if start == stop and not (incl_l and incl_r):
            raise ValueError(f'Invalid interval from {start} to {stop} without inclusive bounds')
        if (start == np.infty and incl_l) or (stop == np.infty and incl_r):
            raise ValueError("Infinity cannot be included")

        self._int = [start, stop]
        self._l = incl_l
        self._r = incl_r

    def l_inclusive(self):
        return self._l

    def r_inclusive(self):
        return self._r

    def overlaps(self, other: Interval):
        if other[1] < self[0] or self[1] < other[0]:
            return False
        if other[1] == self[0] and not (other._r and self._l):
            return False
        if self[1] == other[0] and not (self._r and other._l):
            return False
        return True

    def __contains__(self, x: float):
        return self.overlaps(Interval(x, x, incl_r=True))

    def __eq__(self, other: Interval):
        return self[0] == other[0] and self[1] == other[1] and self._l == other._l and self._r == other._r

    def __len__(self):
        return self._int[1] - self._int[0]

    def __getitem__(self, item):
        return self._int[item]

    def __repr__(self):
        return str(self)

    def __str__(self):
        left = '[' if self._l else '('
        right = ']' if self._r else ')'
        return f'{left}{self._int[0]}, {self._int[1]}{right}'


class Linear:
    """
    A class that represents a simple linear function.

    Examples
    ----------
    f(x) = 3x + 3 on R <=> Linear(3, 3, Interval(-np.inf, np.inf, incl_l=False))
    f(x) = -x + 1 on [0, 1] <=> Linear(0, 1, Interval(0, 1, incl_r=True))
    """
    def __init__(self, slope: float, intercept: float, domain: Interval):
        if slope is None:
            raise ValueError('Slope must not be None')
        if intercept is None:
            raise ValueError('Intercept must not be None')
        if domain is None:
            raise ValueError('Domain must not be None')

        self._a = slope
        self._b = intercept
        self._d = domain

    def slope(self):
        return self._a

    def intercept(self):
        return self._b

    def domain(self):
        return self._d

    def __call__(self, x: float) -> float:
        if x not in self._d:
            raise ValueError(f'Value {x} is not in the domain {self._d}')
        return self._a * x + self._b

    def __str__(self):
        return f'f(x) = {self._a}x+{self._b} for x in {self._d}'


class PiecewiseLinear:
    """
    A class that represents a piecewise function.

    Examples
    ----------
    f_1(x) = 2x + 1 on (-inf, 0)
    f_2(x) = 3x + 5 on [0, inf)
    <=>
    f1 = Linear(2, 1, Interval(-np.inf, 0, incl_l=False))
    f2 = Linear(3, 5, Interval(0, np.inf))
    pl = PiecewiseLinear([f1, f2])
    """
    def __init__(self, funcs: list[Linear]):
        if funcs is None:
            raise ValueError('Function definitions must not be None')
        if not mutually_disjoint([f.domain() for f in funcs]):
            raise ValueError('Function domains must be mutually disjoint')

        self._funcs = sorted(funcs, key=lambda x: x.domain()[0])
        self._num_funcs = len(funcs)

    def __len__(self):
        return self._num_funcs

    def __call__(self, x: float):
        for f in self._funcs:
            if x in f.domain():
                return f(x)
        raise ValueError(f'Value {x} does not lie any defined domain')

    def __repr__(self):
        return str(self)

    def __str__(self):
        ss = [f'Piecewise linear function definitions ({self._num_funcs}):']
        ss = ss + [str(f) for f in self._funcs]
        return '\n'.join(ss)


class Branch:
    """
    A class that represents a branch of a tree with domain and label spceified.

    Examples
    ----------
    A branch labelled 'a' with domain [0, 1] <=> Branch('a', Interval(0, 1, incl_r=True))
    """
    def __init__(self, label: str, domain: Interval, order: int):
        if label is None or domain is None or order is None:
            raise ValueError('Branch parameters must not be None')

        self._label = str(label)
        self._domain = domain
        self._order = order

    def label(self):
        return self._label

    def order(self):
        return self._order

    def domain(self):
        return self._domain

    def __contains__(self, x: float):
        return x in self._domain

    def __eq__(self, other: Branch):
        return self._label == other._label and self._domain == other._domain

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f'''label='{self._label}', domain={self._domain}'''


class Tree:
    """
    A class that represents a simple tree.

    This tree has one fixed point that joins an arbitrary number of branches.
    """
    def __init__(self, f: PiecewiseLinear, bs: list[Branch]):
        if f is None or bs is None:
            raise ValueError('Tree parameters must not be None')
        # if len(f) != len(bs):  # TODO: Not necessarily have to equal
        #     raise ValueError('The number of functions does not match the number of branches')

        self._f = f
        self._bs = bs
        self._num_branches = len(bs)
        self._values = []
        self._itinerary = []
        self._meta = sorted([(b.label(), b.order()) for b in bs])

    def iter(self, start: float, num_it: int):
        self._values = []
        self._itinerary = []

        s = start
        for _ in range(num_it):
            self._values.append(s)
            self._itinerary.append(self.which_branch(s))
            s = self._f(s)

    def which_branch(self, x: float) -> Branch:
        for b in self._bs:
            if x in b:
                return b
        raise ValueError(f'Value {x} does not lie in any branch')  # reached if f is not surjective

    def values(self):
        return self._values

    def labels(self):
        return [m[0] for m in self._meta]

    def orders(self):
        return [m[1] for m in self._meta]

    def itinerary(self):
        return self._itinerary

    def __contains__(self, other: Branch):
        for b in self._bs:
            if b == other:
                return True
        return False

    def __repr__(self):
        return str(self)

    def __str__(self):
        ss = ['Tree definition:']
        bd = [f'Branch definitions ({self._num_branches}):']
        bd = bd + [str(b) for b in self._bs]
        ss = ss + bd + [str(self._f)]
        return '\n'.join(ss)


class Plotter:
    def __init__(self, title: str, ylabel: str, xlabel: str):
        self._title = title
        self._ylabel = ylabel
        self._xlabel = xlabel

    def plot(self, x, y, ticks=None, labels=None, title=None):
        plt.title(self._title if title is None else title)
        plt.ylabel(self._ylabel)
        if ticks is not None and labels is not None:
            plt.yticks(ticks, labels)
        plt.xlabel(self._xlabel)
        plt.plot(x, y, marker='o')
        plt.show()
