import numpy as np
import cv2 as cv

import growingnn.config as config


class LRUCache:
    """LRU Cache implementation with memory monitoring for RESHEPERS"""
    
    def __init__(self, max_size=10, max_memory_mb=100, enable_monitoring=True):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024  # Convert MB to bytes
        self.enable_monitoring = enable_monitoring
        self.cache = {}
        self.current_memory_usage = 0
        
    def _get_memory_usage(self, array):
        """Calculate memory usage of a numpy array in bytes"""
        if array is None:
            return 0
        return array.nbytes
    
    def is_memory_limit_reached(self, memory_usage):
        """Check if we need to evict items based on size or memory limits"""
        if len(self.cache) < 0.2 * self.max_size:
            return False
        return (len(self.cache) + 1 >= self.max_size or 
                (self.enable_monitoring and self.current_memory_usage + memory_usage >= self.max_memory_bytes))
    
    def get(self, key):
        """Get item from cache and update its position"""
        if key in self.cache:
            return self.cache[key]
        return None
    
    def put(self, key, value):
        """Put item in cache with LRU eviction if needed"""
        # Remove if already exists
        if key in self.cache:
            return

        if self.is_memory_limit_reached(self._get_memory_usage(value)):
            self.clear()

        # Add new item
        self.cache[key] = value
        if self.enable_monitoring:
            self.current_memory_usage += self._get_memory_usage(value)
    
    def clear(self):
        """Clear the cache"""
        self.cache.clear()
        self.current_memory_usage = 0

# Initialize the LRU cache with config settings
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


def get_reshsper(size_from, size_to):
    key = (size_from, size_to)
    
    # Try to get from cache first
    cached_value = RESHEPERS.get(key)
    if cached_value is not None:
        return cached_value
    
    # Create new resheper if not in cache
    new_resheper = np.ascontiguousarray(eye_stretch(size_from, size_to), dtype=config.FLOAT_TYPE)
    RESHEPERS.put(key, new_resheper)
    return new_resheper

def clear_reshepers_cache():
    """Clear the RESHEPERS cache to free memory"""
    RESHEPERS.clear()