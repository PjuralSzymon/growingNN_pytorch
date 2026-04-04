(03.2026 - ...)
1. During testing it seemed like a waste of resources to set residual layers to quasi-identity weight initialization. That is why it was changed, and the weight initialization for residual layers was only set to zero or random initialization close to zero, whereas in the original paper there were three types: zero, quasi-identity, and random.
