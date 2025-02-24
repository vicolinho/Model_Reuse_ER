import math

import numpy as np
from networkx import Graph
from sklearn.metrics.pairwise import cosine_similarity


def graipher(pts, K):
    farthest_pts_index = np.zeros((K), dtype=int)
    farthest_pts_index[0] = np.random.randint(len(pts))
    sims_sum = cosine_similarity(pts[farthest_pts_index[0]].reshape((1, -1)), pts)
    sims_avg = sims_sum / 1.0
    for i in range(1, K):
        # print(avg_sims.shape)
        farthest_pts_index[i] = np.argmin(sims_avg)
        sims_sum = sims_sum + cosine_similarity(pts[farthest_pts_index[i]].reshape((1, -1)), pts)
        # sims = np.maximum(sims, cosine_similarity(pts[farthest_pts_index[i]].reshape((1, -1)), pts))
        sims_avg = sims_sum / float(i)
        sims_avg[0][farthest_pts_index] = math.inf
    return farthest_pts_index

def graipher_with_initial_points(pts, K, initial_points):

    farthest_pts_index = np.zeros((K), dtype=int)
    sims_sum = cosine_similarity(initial_points[0].reshape((1, -1)), pts)
    sims_avg = sims_sum / 1.0
    for i in range(1, initial_points.shape[0]):
        sims_sum = sims_sum + cosine_similarity(initial_points[i].reshape((1, -1)), pts)
        sims_avg = sims_sum / float(i)
    print(sims_avg.shape)
    for i in range(0, K):
        # print(avg_sims.shape)
        farthest_pts_index[i] = np.argmin(sims_avg)
        sims_sum = sims_sum + cosine_similarity(pts[farthest_pts_index[i]].reshape((1, -1)), pts)
        # sims = np.maximum(sims, cosine_similarity(pts[farthest_pts_index[i]].reshape((1, -1)), pts))
        sims_avg = sims_sum / float(i+initial_points.shape[0])
        sims_avg[0][farthest_pts_index] = math.inf
    return farthest_pts_index
