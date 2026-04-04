In the original paper for growingNN, adding sequential layers is very simple because a layer is always connected with an activation function and nodes are pretty standalone tools, but in PyTorch everything can be a module and it is hard to predict what it will be in every use case. So even if a function that goes over layers is called:
module_sequential_pairs(...)
it will only return pairs of layers placed sequentially, it is possible that between those layers there will be some activation function or something even different and custom. To work around this, we are adding the following logic:

To place a layer between l1 -> l2, we start with l2 and go backward until we find l1, and then add the new layer on a path from l2 to l1 but just before l2. This way we can handle whatever will be placed between l1 and l2, with the idea that everything between those two layers will be a part of l1, just like activation functions, not a part of l2.
