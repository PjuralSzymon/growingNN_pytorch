To allow more possibilities to manually test the behaviour of the network after adding residual layers, I added a test which randomly adds 30 different layers residually of a given type of weight initialization. The main idea is that the network will not change even after many of those additions. Depending on different types of weight initialization, the results were the following:

For random weight initialization where weights were initialized with a (0.0, 0.01) range:
![[Pasted image 20260404172754.png]]
Here is a difference between what the model was returning before and what it is returning now; we can see that the more layers we add, the bigger the difference.

The graph was growing from such a small one:
![[Pasted image 20260404173518.png]]
To this big version:
![[Pasted image 20260404173508.png]] Those images are simplified versions of the torch.fx graph output.


If the range for random initialization is set to 0, then the output has no change. But if we set it to (0, 0.1), the change between the previous output and the current output will grow significantly: the bigger the range of random initialization, the bigger the difference in output and the bigger the chance of "forgetting".

![[Pasted image 20260404173955.png]]

When I set the value for the residual layers to EYE mode, which is residual layers where the initialization is set by quasi-identity, it did not make any sense, and the loss of learned information was huge.
[[Part 1]]
![[Pasted image 20260404174521.png]]
