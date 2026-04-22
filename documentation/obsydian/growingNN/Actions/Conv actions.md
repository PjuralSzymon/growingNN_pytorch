Adding conv actions was very problematic because not all properties were accessible which were in growingNN


Something is still wrong with those conv 

Especially the seq version weight initialization 

Also checking the layer type should support other versions compatible with not only conv but also Avg pool and max pool 



We are missing adding conv between conv and dense layer 



## Changes from original growingNN paper

The original paper doesn't focus on conv layer initialization; it is using global config mode, which can be uniform/normal distribution and so on, which was probably a reason for the common data loss described in the paper or some instabilities seen in the training history. In this implementation, we are using special initialization that should limit memory loss during training 

## Residual connections

Those layers are generated with zero weight initialization 


## Sequential connections

Those layers are generated with zero weight initialization but with single 1 in the middle so we are not using the quasi-identity here because conv layers are not using matrix multiplication; those are using convolution 

## Connections between conv and linear

It was much harder to implement this operation in PyTorch than in growingNN. PyTorch doesn't require those specific shapes in the layers that we can get to know the exact values, so we need to go over it, and one of the ways was the implementation of create_zero_conv_before_linear which after the conv layer adds adaptive max or average pool and then flatten. It is not clear to me why only flatten did not work without adaptive pooling, apparently
