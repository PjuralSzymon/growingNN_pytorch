This module is tested by: [[Adding residual layers]]

1. Graph how output of model czhanged copared with how many parameter have the model during adding linear res layer
![[Pasted image 20260510215912.png]]


## Changes from original growingNN paper

The original paper doesn't focus on conv layer initialization; it is using global config mode, which can be uniform/normal distribution and so on, which was probably a reason for the common data loss described in the paper or some instabilities seen in the training history. In this implementation, we are using special initialization that should limit memory loss during training 
