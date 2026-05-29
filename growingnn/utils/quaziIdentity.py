"""Cached quasi-identity rescaling matrices for weight re-projection."""

import numpy as np
import cv2 as cv
import torch

import growingnn.core.config as config


class LRUCache:
    """LRU Cache with memory monitoring."""

    def __init__(self, max_size=10, max_memory_mb=100, enable_monitoring=True):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.enable_monitoring = enable_monitoring
        self.cache = {}
        self.current_memory_usage = 0

    def _get_memory_usage(self, array):
        if array is None:
            return 0
        return array.nbytes

    def is_memory_limit_reached(self, memory_usage):
        if len(self.cache) < 0.2 * self.max_size:
            return False
        return (len(self.cache) + 1 >= self.max_size or
                (self.enable_monitoring and self.current_memory_usage + memory_usage >= self.max_memory_bytes))

    def get(self, key):
        if key in self.cache:
            return self.cache[key]
        return None

    def put(self, key, value):
        if key in self.cache:
            return
        if self.is_memory_limit_reached(self._get_memory_usage(value)):
            self.clear()
        self.cache[key] = value
        if self.enable_monitoring:
            self.current_memory_usage += self._get_memory_usage(value)

    def clear(self):
        self.cache.clear()
        self.current_memory_usage = 0


RESHEPERS = LRUCache(
    max_size=config.RESHEPERS_CACHE_MAX_SIZE,
    max_memory_mb=config.RESHEPERS_CACHE_MAX_MEMORY_MB,
    enable_monitoring=config.RESHEPERS_CACHE_ENABLE_MONITORING
)


def eye_stretch(a, b):
    if a == b:
        return np.eye(a)
    A = np.eye(max(a, b))
    return cv.resize(A, (a, b)).T


def get_reshsper(size_from, size_to, *, dtype=None, device=None):
    """Return a rescaling matrix as a torch.Tensor (cached on CPU, then cast)."""
    key = (size_from, size_to)
    cached_value = RESHEPERS.get(key)
    if cached_value is None:
        np_matrix = eye_stretch(size_from, size_to)
        cached_value = torch.from_numpy(np.ascontiguousarray(np_matrix, dtype=config.FLOAT_TYPE))
        RESHEPERS.put(key, cached_value)
    if dtype is not None or device is not None:
        return cached_value.to(dtype=dtype, device=device)
    return cached_value


def clear_reshepers_cache():
    """Clear the RESHEPERS cache to free memory."""
    RESHEPERS.clear()
