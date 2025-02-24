import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def cosine_comp(val1:np.ndarray, val2:np.ndarray):
    """Calculate the cosine similarity between two vectors.

     Returns a value between 0.0 and 1.0.
  """
    try:
        return cosine_similarity(val1.reshape(1, -1), val2.reshape(1, -1))[0][0]
    except ValueError:
        return 0