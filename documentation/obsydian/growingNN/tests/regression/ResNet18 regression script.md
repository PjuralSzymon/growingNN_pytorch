This note is about `tests/regression/resnet_regression_test.py`.

What it does. It loads a real pretrained ResNet-18 from `torchvision.models.resnet18` with `ResNet18_Weights.DEFAULT`, sets `eval()`, traces with `torch.fx.symbolic_trace` into `gm`, then loops up to `ITERATIONS = 50` steps. Each step builds a list of actions from `AddResLinearLayer.generate_all_actions`, `AddResConvLayer.generate_all_actions`, and optional seq or delete flags, picks one action at random, calls `execute`, runs `gm(x)` with `BATCH_SIZE = 2` and `INPUT_SHAPE = (3, 64, 64)`, logs norms, draws FX graphs into `testResults/regression/` via FX graph drawer `draw_filtered_fx_graph` and `draw_torch_fx_graph`.

Why. It stress-tests growth on a large real graph with dotted submodule names. Where. Run as a script from the `tests` folder; CLI uses `parse_regression_cli` from Regression utils.

---

### Constants of interest

Lines 34 to 42: `USE_ADD_RES_LAYER`, `USE_ADD_RES_CONV_LAYER`, `USE_ADD_SEQ_LAYER`, `USE_ADD_SEQ_CONV_LAYER`, `USE_DEL_LAYER`. Line 41 sets input shape tuple `(3, 64, 64)`. Line 40 sets batch size `2`.

### Links to new safety logic

`AddResConvLayer.generate_all_actions` uses `LayerShapeAnalyser` / `LayerBridgeFinder` from `growingnn.utils.fx` so conv residual candidates that would break `torch.add` on different spatial sizes (for example a pair from `layer3` to `layer4` on ResNet-18) are skipped when shape metadata is present.