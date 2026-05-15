This page lists defaults in `growingnn/core/config.py`. Other modules import this file as `growingnn.core.config` or `import growingnn.core.config as config`.

### Why a central config

One file keeps numbers for init ranges, cache limits, which module types count as editable, and logging flags. [[Logger]] reads the log-related keys here. [[Layer Factory]] reads `ADDING_RES_LAYERS_WEIGHT_INITIALIZATION_RANGE` and `RES_CONV_TO_LINEAR_GLOBAL_POOL_TYPE`. [[Quasi identity]] reads `FLOAT_TYPE`, `RESHEPERS_CACHE_*`, and uses the same float type for cached arrays.

### Main symbols

`FLOAT_TYPE` is `numpy.float32` at line 6.

`RESHEPERS_CACHE_MAX_SIZE` is `10` at line 8. `RESHEPERS_CACHE_MAX_MEMORY_MB` is `100` at line 9. `RESHEPERS_CACHE_ENABLE_MONITORING` is `True` at line 10. These feed the `LRUCache` in `growingnn/actions/utils/quaziIdentity.py` lines 56 to 59.

`ADDING_RES_LAYERS_WEIGHT_INITIALIZATION_RANGE` is `(0.0, 0.01)` at lines 12 and 13. Used in `LinearFactory.create_random_linear` in `growingnn/actions/utils/layer_Factory.py` lines 43 to 46.

`RES_CONV_TO_LINEAR_GLOBAL_POOL_TYPE` is the string `"max"` at line 13. Allowed values in code are `"avg"` or `"max"`. Used in `ConvFactory.create_zero_conv_before_linear` in `layer_Factory.py` lines 103 to 110.

`EDITABLE_MODULES` is a list `[nn.Linear, nn.Conv2d, nn.Conv1d, nn.Conv3d]` at line 15. [[Model Analyser]] uses it in `_is_editable_module` to decide which `call_module` nodes count as editable.

Logging block: `ENABLE_LOGGING` `True` line 18. `LOG_LEVEL` `"DEBUG"` line 19. `LOG_TO_FILE` `True` line 20. `LOG_FILE_NAME` `"growingnn.log"` line 21. `LOG_FILE_MAX_BYTES` `100 * 1024 * 1024` line 22. `LOG_FILE_BACKUP_COUNT` `9` line 23. Rough cap near 1 GB total with one active file plus backups.

### Known limitations

`EDITABLE_MODULES` is a list, not a tuple. Do not pass it as the second argument to `isinstance` in other code; action classes use their own `SUPPORTED_MODULES` tuples. See [[Residual Linear Actions]] for the `isinstance` tuple rule.

Changing `LOG_TO_FILE` to `False` still keeps the stream handler when `ENABLE_LOGGING` is `True`.

### Related

[[Logger]], [[Model Analyser]], [[Layer Factory]], [[Quasi identity]], [[Index]].
