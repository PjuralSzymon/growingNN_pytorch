The training loop is standard SGD, like in most PyTorch projects. It runs in the training stage of each generation described in [[General]].

When you set the number of training epochs, remember two things. First, that count applies to one generation only, not to the full run. Second, the same loop runs again in every generation. The main difference from a plain training script is the learning rate. It is controlled by [[Learning Rate Scheduler]], which ramps up slowly at the start of a generation and down at the end. That keeps training stable while the model structure changes.
