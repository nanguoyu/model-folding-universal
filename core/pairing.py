import numpy as np
import torch
import numpy as np
import math
from scipy.optimize import linear_sum_assignment

def count_unmatched_elements(array):

  if array.ndim != 1:
    raise ValueError(f"array should be 1-d array")

  length = array.numel()

  unmatched_array = torch.zeros(length, dtype=torch.bool)

  for i in range(length):
    if array[i] == i:
      unmatched_array[i] = True

  return unmatched_array.sum()


def greedy_channel_pairing(correlation_matrix, pruned_channels=None, pairing_rate=1.0):
    """
    Pair channels using a greedy algorithm based on their absolute correlation, 
    considering both positive and negative correlations, and pair only a certain
    percentage of channels based on the pairing rate.

    Parameters:
    correlation_matrix (np.ndarray): A 2D array representing the correlation 
                                     between each pair of channels.
    pruned_channels (list of int, optional): List of pruned channel indices. If provided,
                                             the algorithm pairs pruned channels with 
                                             unpruned channels. Default is None.
    pairing_rate (float): The proportion of total channel pairs to be paired,
                          ranging from 0 (no pairing) to 1 (full pairing).

    Returns:
    list of tuples: A list where each tuple contains a pair of indices representing 
                    the paired channels.
    """
    # Make a copy of the correlation matrix to avoid modifying the original matrix
    corr_matrix = np.copy(correlation_matrix.cpu())

    n_channels = corr_matrix.shape[0]
    if pruned_channels is None:
        # Calculate the total number of pairs to be formed
        total_pairs = int(pairing_rate * (n_channels / 2))
        if pairing_rate>0 and total_pairs==0:
            total_pairs=1
        # total_pairs = int(math.ceil(pairing_rate * (n_channels / 2)))
        # Initialize an empty list to store the pairs
        pairs = []

        # Set diagonal elements to a very low value to avoid self-pairing
        np.fill_diagonal(corr_matrix, -np.inf)

        # Initialize a set to keep track of paired channels
        paired_channels = set()

        # Greedy pairing
        while len(pairs) < total_pairs and len(paired_channels) < n_channels:
            i, j = np.unravel_index(np.argmax(corr_matrix), corr_matrix.shape)
            # Add the pair (i, j) to the list of pairs
            pairs.append((i, j))
            # Add channels to the set of paired channels
            paired_channels.add(i)
            paired_channels.add(j)

            # Set the corresponding rows and columns to a very low value to avoid re-pairing
            corr_matrix[i, :] = -np.inf
            corr_matrix[:, i] = -np.inf
            corr_matrix[j, :] = -np.inf
            corr_matrix[:, j] = -np.inf


        reverse_pairs = []
        for i,j in pairs:
            reverse_pairs.append((j,i))

        pairs.extend(reverse_pairs)
        return pairs
    else:
        # print(f"corr_matrix: {corr_matrix}")
        print("[greedy_channel_pairing]: pairing channels of a pruned model")
        # Initialize an empty list to store the pairs
        pairs = []

        # Initialize a set to keep track of paired channels
        unpruned_channels = set(range(n_channels)) - set(pruned_channels)

        # Greedy pairing for pruned channels with unpruned channels
        for pruned_channel in pruned_channels:
            max_corr = -np.inf
            best_match = None
            # print(f"Try to find matched unpruned channel for pruned channel:{pruned_channel}")
            for unpruned_channel in unpruned_channels:
                corr = corr_matrix[pruned_channel, unpruned_channel]
                # print(f"corr_matrix[{pruned_channel}, {unpruned_channel}] = {corr_matrix[pruned_channel, unpruned_channel]}")
                if corr > max_corr:
                    # print(f"which is > {max_corr}")
                    max_corr = corr
                    best_match = unpruned_channel
                    # print(f"best match of channel {pruned_channel} is channel {best_match}")
            if best_match is not None:
                pairs.append((pruned_channel, best_match))
        # print(f"Pairs: {pairs}")
        # Add unpruned channels with themselves
        for unpruned_channel in unpruned_channels:
            pairs.append((unpruned_channel, unpruned_channel))
        # print(f"Pairs: {pairs}")
        return pairs

def convert_to_tuple_array(matrix):
  if not np.all((matrix == 0) | (matrix == 1)):
    raise ValueError("matrix should only contain 0 or 1")

  (rows, cols) = matrix.shape

  tuple_array = []

  for i in range(rows):
    for j in range(cols):
      if matrix[i, j] == 1:
        tuple_array.append((i, j))

  return tuple_array


def k_cardinality_linear_sum_assignment(correlation_matrix, k):
    """ solve the k-cardinality LAP by converting it to a standard one """
    # from:  https://github.com/feiran-l/Generalized-Shuffled-Linear-Regression/blob/a62e3830bdb104bcb7cdbe9104d33a8544618865/gslr.py#L29

    cost = np.copy(correlation_matrix.cpu())

    # add penalty to the diagonal to avoid self matching in model folding.
    np.fill_diagonal(cost, -99999)
    # 

    m, n = cost.shape
    k = int(k)
    res = np.zeros((m, m + n - k))
    if min(m, n) == k:
        r, c = linear_sum_assignment(cost, maximize=True)
        res[r, c] = 1
        # print(f"r: {r}")
        # print(f"c: {c}")
        return convert_to_tuple_array(res)
        # return c
    else:
        if m > n:
            cost, m, n, trans_flag = cost.T, n, m, True
        else:
            trans_flag = False
        cost -= np.min(cost) * np.ones(cost.shape)
        # transform the kLAP to an standard LAP
        diag_vec = np.min(cost, axis=1)[:m - k] - np.ones(m - k)
        right_up = m * n * np.max(cost) * np.ones((m - k, m - k))
        np.fill_diagonal(right_up, diag_vec)
        right_down = np.tile(diag_vec, (k, 1))
        dummy = np.concatenate((right_up, right_down), axis=0)
        cost = np.concatenate((cost, dummy), axis=1)
        # solve the transformed LAP
        r, c = linear_sum_assignment(cost, maximize=True)
        res[r, c] = 1
        return convert_to_tuple_array(res[:m, :n].T) if trans_flag else convert_to_tuple_array(res[:m, :n])
        # return c[:n] if not trans_flag else c[:n]

def find_non_twoway_matched_tuples(tuples):
# todo: correct this fun
    twoway_matched = []
    self_matched = []

    for x, y in tuples:
      if x==y:
        self_matched.append((x,y))
      else:
          if (y, x) not in twoway_matched:
              twoway_matched.append((x, y))

    return twoway_matched, self_matched

def find_non_twoway_paired(tuples):
    # Convert list of tuples to a set for faster lookup
    tuples_set = set(tuples)
    non_twoway_paired = set()

    for x, y in tuples:
        # Check if it's not a self match and if its reverse pair is not in the set
        if x != y and (y, x) not in tuples_set:
            non_twoway_paired.add((x, y))

    return list(non_twoway_paired)

def find_unmatched_pairs(arr):
    unmatched_pairs = []

    for i, j in enumerate(arr):
        # Check if the value at the current index points to an index
        # that does not point back to the current index
        if not (0 <= j < len(arr) and arr[j] == i):
            unmatched_pairs.append((i, j))

    return unmatched_pairs


def generate_random_symmetric_matrix(size):
    """
    Generate a random symmetric matrix.

    Args:
    size (int): The size of the matrix (number of rows and columns).

    Returns:
    torch.Tensor: A symmetric matrix.
    """
    random_matrix = torch.rand(size, size)
    symmetric_matrix = (random_matrix + random_matrix.t()) / 2
    return symmetric_matrix

def test_greed_pairing():
    for _ in range(1):
        symmetric_matrix = generate_random_symmetric_matrix(5)
        print(symmetric_matrix)
        
        pairs = greedy_channel_pairing(symmetric_matrix)
        print(f'pairs generated by greedy methods:\n{pairs}')
        self_matched_channels = find_non_twoway_matched_tuples(pairs)
        if len(self_matched_channels)>0:
            print("Exists self-matched pairs")
            raise ValueError(f'The following channels are self matched :{self_matched_channels}')

        perm_map = torch.arange(symmetric_matrix.shape[0])
        for i,j in pairs:
            perm_map[i]=j
            perm_map[j]=i
        # print(perm_map)


if __name__ == '__main__':
    test_greed_pairing()

