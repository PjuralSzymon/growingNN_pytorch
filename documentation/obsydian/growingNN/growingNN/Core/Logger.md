This page is about `growingnn/core/logger.py`. It configures the standard library `logging` object named `"growingnn"`.

### What it does

It attaches a `StreamHandler` so messages go to the terminal. If `LOG_TO_FILE` is true in [[Config]], it also attaches `RotatingFileHandler` from `logging.handlers` with `maxBytes` and `backupCount` from config. The format string is `"%(asctime)s | %(levelname)-8s | %(message)s"` at lines 21 to 22 in `logger.py`.

### Why `logging.raiseExceptions = False`

Line 13 sets `logging.raiseExceptions = False`. On Windows, pytest can replace `sys.stderr` during capture. A handler that still points at an old stream can raise `OSError`. With this flag, logging errors do not print long tracebacks inside the logging system itself.

### Technicalities

If `ENABLE_LOGGING` is false, the logger gets `NullHandler` only (lines 38 to 39). Handlers are added only when the handler list is empty (line 20), so repeated imports do not duplicate handlers.

### Known limitations

Silencing `raiseExceptions` hides all handler bugs, not only the pytest case. Use `LOG_LEVEL` to reduce noise if debug volume is high.
