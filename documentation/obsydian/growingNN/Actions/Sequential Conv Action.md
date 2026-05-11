
## Sequential connections

Those layers are generated with zero weight initialization but with single 1 in the middle so we are not using the quasi-identity here because conv layers are not using matrix multiplication; those are using convolution 

## Diffrences from GrowingNN package:
I removed possiblity that can  palce a layer conv beetwen conv and linear it was very problematic and it was removed from this library 


## Results
Results on big conv model: 
![[Pasted image 20260511225255.png]]