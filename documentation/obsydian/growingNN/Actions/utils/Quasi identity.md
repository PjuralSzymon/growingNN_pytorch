This page is about `growingnn/actions/utils/quaziIdentity.py`. The file name uses the spelling `quazi` in the repo.

### `eye_stretch(a, b)`

Lines 62 to 66. If `a == b`, returns `numpy.eye(a)`. Else builds a max-sized identity then resizes with OpenCV `cv2.resize` to shape `(a, b)` and transposes. Used by [[Layer Factory]] `LinearFactory.create_eye_linear` to get a rectangular weight matrix between two widths.

### `RESHEPERS` cache

Class `LRUCache` at lines 7 to 53. Global instance `RESHEPERS` at lines 56 to 59 reads `RESHEPERS_CACHE_MAX_SIZE`, `RESHEPERS_CACHE_MAX_MEMORY_MB`, and `RESHEPERS_CACHE_ENABLE_MONITORING` from [[Config]].

`get_reshsper(size_from, size_to)` at lines 69 to 80 returns a cached `numpy` array or builds via `eye_stretch` and stores it. Note the spelling `reshsper` in the public function name.

`clear_reshepers_cache` at lines 82 to 84 clears the global cache.

### Dependency

The file imports `cv2` as `cv`. The environment must have OpenCV Python bindings installed for import to succeed.

### Comparison with the original growingNN paper

[[Part 1]] entry 15.03.2026 says residual init moved away from heavy quasi-identity toward zero or small random for some paths. `eye_stretch` remains for `Layer_Type.EYE` linear factory.

### Known limitations

Cache eviction calls `clear()` on the whole cache when memory pressure hits (line 43), not true LRU per key. Typo names (`RESHEPERS`, `get_reshsper`) are public API now.
