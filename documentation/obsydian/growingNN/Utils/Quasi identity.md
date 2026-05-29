File: `growingnn/utils/quaziIdentity.py`. Cached quasi-identity matrices for weight re-projection when layer widths change. Used by [[Layer Factory]].

---

### `eye_stretch(a, b)`

If `a == b`, returns `numpy.eye(a)`. Else builds a max-sized identity, resizes with OpenCV `cv2.resize` to `(a, b)`, transposes. `LinearFactory.create_eye_linear` transposes again to match `nn.Linear` layout `(out_features, in_features)`.

---

### `RESHEPERS` cache

Class `LRUCache` (lines 10 to 47). Global `RESHEPERS` (lines 50 to 54) reads `RESHEPERS_CACHE_MAX_SIZE`, `RESHEPERS_CACHE_MAX_MEMORY_MB`, `RESHEPERS_CACHE_ENABLE_MONITORING` from [[Config]].

`get_reshsper(size_from, size_to)` returns a cached `numpy` array or builds via `eye_stretch`. Public spelling `reshsper` is intentional in code.

`clear_reshepers_cache()` clears the global cache.

---

### Dependency

Imports `cv2`. OpenCV Python bindings must be installed.

---

### Known limitations

Cache eviction clears the whole cache on memory pressure, not per-key LRU. Typo names (`RESHEPERS`, `get_reshsper`) are public API.
