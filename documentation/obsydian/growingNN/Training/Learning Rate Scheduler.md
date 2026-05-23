The learning rate plays a crucial role when the network structure changes often. Each action chosen in simulation can make the next training stage unstable. That is why we use a custom scheduler in the [[Training loop]].

At the start of a generation, the scheduler raises the learning rate slowly up to a maximum. At the end, it lowers the rate again. That gives a smooth transition before and after an architecture change.

In the first paper we compared several schedulers. We kept the best one from that study for a long time. During later work we saw that this scheduler can still cause instability in some cases and definitly needs more reaserch.

![[Pasted image 20260523092815.png]]
