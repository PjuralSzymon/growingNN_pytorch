We implemented this diffrent that oringal so in the orignal paper we conenct every layer to every else sepertaly here since in pytorch is not taht possible we first get teh sum of all input and then pass it to all output which may be much more effitient 

To find out what layers we can removed we used [[Model Analyser#^7a8eff]]
To run the delete funciton so the main part we use [[Model Transformer#^f4531d]]

Unforutantelly currently deleting layers have 2 problems:
1. It leaves intermidate stages which are hard to remove conceptally ![[Pasted image 20260506222841.png]]
2. Second issue is that those give a huge data loss: ![[Pasted image 20260506223240.png]] In the below graph from around 25 we started to remove layers 