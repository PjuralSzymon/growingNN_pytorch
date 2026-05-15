This page is about `tests/regression/regression_utils.py`.

### `FOLDER_NAME`

String constant `"testResults/regression"` at line 10. Output PDFs and plots use this folder under the working directory.

### `clear_regression_folder`

Lines 13 to 17. Logs at debug level, deletes the folder if it exists with `shutil.rmtree`, then recreates it with `os.makedirs`.

### `plot_norms_and_parameter_count`

Lines 20 to 38. Builds a twin-axis matplotlib figure: left axis plots `norms` sequence, right axis plots `parameter_amounts` slice aligned to steps. Calls `plt.show()` at line 38. Non-headless runs may open a window unless the caller sets a backend such as `Agg` (see [[Test runners]]).

### `parse_regression_cli`

Lines 41 to 52. `argparse` parser with `--save-output` or `--save_output` taking `true` or `false`. Default is `false`. Sets `ns.save_output` as a real boolean.

### Known limitations

Importing this module pulls in `matplotlib`. Unit tests that import action modules which transitively import regression helpers can accidentally parse pytest argv if `parse_regression_cli()` runs at import time without passing `argv=[]`; keep CLI parsing only inside `if __name__ == "__main__"` blocks.

### Related

[[ResNet18 regression script]], [[Utils testing script]], [[Test runners]], [[Index]].
