In orignal paper for groiwngNN adding sequentail layers is very simple becoause layer is allways conencted wiht a activation funciton and nodes are pretty stanalone toolls but in pytorch everything can me a module and it's hard to rpedict what it iwll be in a every use case so even if a fucntion that going over layers called: 
module_sequential_pairs(...)
Will only return pair of layers palced sequentailly it is a posisble that beetwen thsoe layers there will be some activation function or something even diffrent and custom so to go around this, we are adding the following logic

To palced a laer beetwen l1 ->l2 wirh start with l2 and go backward until you find l1 
and then add the new layer on apath to l1 form l2 but jsut before l2 this way we can handle what ever will be palced beetwen l1 and l2 with an idea that everyhint beetwen those 2 layers will be a part of l1 just like act funcitons not a part of l2 