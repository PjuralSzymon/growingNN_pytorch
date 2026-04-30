(03.2026 - ...)
1. (15.03.2026) During testing it seemed like a waste of resources to set residual layers to quasi-identity weight initialization. That is why it was changed, and the weight initialization for residual layers was only set to zero or random initialization close to zero, whereas in the original paper there were three types: zero, quasi-identity, and random.
2. (29.04.2026) Almost all add layer modueles were implemnted I added a new class delete layer it is not finsihed I htink module analyser will require a lot of upgrade to make it work we need functions thatw ill get all layers before a given layer and all layers after, also some funciton of required type, I'm starting to wonder if this is not too limited maybe that what layers are consider "adjustable" should be written from configuration layer for example don't work on nn.Linear work only on FeedForward which can be a custom module (common in LLMs)
3. (30.04.2026) Delete layer action is almost ready but regression test seems to fail in 1 particulas scenario and be a bit unstable, 
	1. It fails when we add few (<10) layers and then remove those 10 for some reason 1 is allways left Maybe it don't handle the case when all the inputs come form add modules ?
	2. If fails when we add and then remove a lot of layers maybe the same reason as 1.
	3. 
