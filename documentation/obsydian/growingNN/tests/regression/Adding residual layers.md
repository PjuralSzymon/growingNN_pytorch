To give more possibility to manually tests the behaviour of netowrk after adding resifdual layets I added a test which randmly adds 30 diffrent layers residualy of given type of weight initializaiton, The main idea is that the netowrk won't change after even a lot of those additions, depending on diffrent types of weight initialization the results were teh following: 

For random weight initalization where weights were initalized with (0.0, 0.01) range:
![[Pasted image 20260404172754.png]]
Here is a diffrence from what model was returning before and what is returning now we can see taht the more layers we add the bigger the difrence 

The graph was growing from such a small one: 
![[Pasted image 20260404173518.png]]
To this big version: 
![[Pasted image 20260404173508.png]]Those images are simplyfited version of torch.fx graph output


If the range for random initalization is set to 0 then the Output have no change. But if we will set it to (0,0.1) the change beetwen previous output and current output will gorw significantly: THe bigger range of random initalization the bigger diffrence of output the bigger chance of "forgetting"

![[Pasted image 20260404173955.png]]

When i set the value for the residual layers to EYE mode which is residual layers where the initalization is set by quazi identiti it didn't made any sense, and the loss over learned infomration was huge
[[Part 1]]
![[Pasted image 20260404174521.png]]