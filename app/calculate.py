#!/usr/bin/python
# -*- coding: UTF-8 -*-
import numpy as np
import faiss


def knn(x, y, k, use_gpu):
    """ Wrapper for knn on CPU/GPU

    :param x:
    :param y:
    :param k:
    :param use_gpu:
    :return:
    """
    return knn_gpu(x, y, k) if use_gpu else knn_cpu(x, y, k)


def knn_gpu(x, y, k, mem=5 * 1024 * 1024 * 1024):
    """
        Perform knn on GPU
    """
    dim = x.shape[1]
    batch_size = mem // (dim * 4)
    sim = np.zeros((x.shape[0], k), dtype=np.float32)
    ind = np.zeros((x.shape[0], k), dtype=np.int64)
    for x_from in range(0, x.shape[0], batch_size):
        xto = min(x_from + batch_size, x.shape[0])
        bsims, binds = [], []
        for y_from in range(0, y.shape[0], batch_size):
            yto = min(y_from + batch_size, y.shape[0])
            idx = faiss.IndexFlatIP(dim)
            idx = faiss.index_cpu_to_all_gpus(idx)
            idx.add(y[y_from:yto])
            bsim, bind = idx.search(x[x_from:xto], min(k, yto - y_from))
            bsims.append(bsim)
            binds.append(bind + y_from)
            del idx
        bsims = np.concatenate(bsims, axis=1)
        binds = np.concatenate(binds, axis=1)
        aux = np.argsort(-bsims, axis=1)
        for i in range(x_from, xto):
            for j in range(k):
                sim[i, j] = bsims[i - x_from, aux[i - x_from, j]]
                ind[i, j] = binds[i - x_from, aux[i - x_from, j]]
    return sim, ind


def knn_cpu(x, y, k):
    """
        Perform knn on CPU
    """
    dim = x.shape[1]
    idx = faiss.IndexFlatIP(dim)
    idx.add(y)
    sim, ind = idx.search(x, k)
    return sim, ind


def score(x, y, fwd_mean, bwd_mean, margin):
    """
        score
    """
    return margin(x.dot(y), (fwd_mean + bwd_mean) / 2)


def score_candidates(x, y, candidate_inds, fwd_mean, bwd_mean, margin):
    """
        score score_candidates
    """
    scores = np.zeros(candidate_inds.shape)
    for i in range(scores.shape[0]):
        for j in range(scores.shape[1]):
            k = candidate_inds[i, j]
            scores[i, j] = score(x[i], y[k], fwd_mean[i], bwd_mean[k], margin)
    return scores
