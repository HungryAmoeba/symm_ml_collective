import itertools
import numpy as np
from typing import FrozenSet, Set

# ==========
# PSET 1
# ==========


def permutation_matrices(n: int) -> np.ndarray:
    """Generates all permutation matrices of n elements
    Input:
        n: int
    Output:
        matrices: np.array of shape [n!, n, n]
    """
    matrices = np.array(list(itertools.permutations(np.eye(n), n)))
    return matrices


def generate_group(matrices: np.ndarray, decimals=4) -> np.ndarray:
    """Generate new group elements from matrices (group representations)
    Input:
        matrices: np.array of shape [n, d, d] of known elements
        decimals: int number of decimals to round to when comparing matrices
    Output:
        group: np.array of shape [m, d, d], where m is the size of the resultant group
    """
    (_, d, d2) = matrices.shape
    assert d == d2
    while True:
        trials = np.einsum("aij,bjk->abik", matrices, matrices).reshape((-1), d, d)
        trials = np.concatenate([matrices, trials], axis=0)
        (_, i) = np.unique(
            np.round(trials, decimals=decimals), axis=0, return_index=True
        )
        if len(i) == len(matrices):
            return matrices
        matrices = trials[i]


def cyclic_matrices(n: int) -> np.ndarray:
    """Generates all cyclic matrices of n elements
    Input:
        n: int
    Output:
        matrices: np.array of shape [n, n, n]
    """
    # Identity matrix
    generator = np.eye(n)
    # Roll
    generator = np.roll(generator, -1, axis=0)
    # Generate the corresponding group
    return generate_group(generator[None, ...])


def make_multiplication_table(
    matrices: np.ndarray, *, tol: float = 1e-08
) -> np.ndarray:
    """Makes multiplication table for group.
    Input:
        matrices: np.array of shape [n, d, d], n matrices of dimension d that form a group under matrix multiplication.
        tol: float numberical tolerance
    Output:
        Group multiplication table.
        np.array of shape [n, n] where entries correspond to indices of first dim of matrices.
    """
    (n, d, d2) = matrices.shape
    assert d == d2
    mtables = np.einsum("nij,mjk->nmik", matrices, matrices)
    result = mtables.reshape(1, n, n, d, d) - matrices.reshape(n, 1, 1, d, d)
    indices = np.nonzero(np.all(np.all((np.abs(result) < tol), axis=(-1)), axis=(-1)))
    indices = np.stack(indices)
    table = np.zeros([n, n], dtype=np.int32)
    table[indices[1], indices[2]] = indices[0]
    return table


def factors(n):
    """
    Returns the positive factors of the given nonzero integer n as a NumPy array.

    Parameters:
    n (int): A nonzero integer whose factors are to be determined.

    Returns:
    np.ndarray: Sorted array of factors.

    Raises:
    ValueError: If n is 0.
    """
    if n == 0:
        raise ValueError("Zero has infinitely many factors.")

    # Work with the absolute value to handle negative inputs.
    factors_set = set()

    # Loop from 1 to the square root of n
    for i in range(1, int(np.sqrt(n)) + 1):
        if n % i == 0:
            factors_set.add(i)
            factors_set.add(n // i)

    # Return the sorted factors as a NumPy array.
    return np.array(sorted(factors_set))


def subgroups(table: np.ndarray) -> Set[FrozenSet[int]]:
    """Find all subgroups of group.
    Input:
        table: np.array of shape [n, n] where the entry at [i, j] is the index of the product of the ith and jth elements in the group.
    Output:
        Yields tuples of elements that form subgroup.
    """
    n = table.shape[0]
    subgroups = set()
    for f in np.factors(n):
        for c in itertools.combinations(range(n), f):
            subtable = table[np.array(c)][:, np.array(c)]
            if set(c) == set(subtable.reshape((-1))):
                subgroups.add(frozenset(c))
    return subgroups


def conjugacy_classes(table: np.ndarray) -> Set[FrozenSet[int]]:
    """Returns the conjugacy classes of the group.
    Input:
        table: np.array of shape [n, n] where the entry at [i, j] is the index of the product of the ith and jth elements in the group.
    Output:
        Set of conjugacy classes. Each conjugacy class is a set of integers.
    """
    n = table.shape[0]
    g_inv = inverses(table)
    g = np.arange(n)
    return {frozenset(table[g, table[i, g_inv]]) for i in range(n)}
