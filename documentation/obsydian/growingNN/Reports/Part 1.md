(03.2026 - ...)
1. (15.03.2026) During testing it seemed like a waste of resources to set residual layers to quasi-identity weight initialization. That is why it was changed, and the weight initialization for residual layers was only set to zero or random initialization close to zero, whereas in the original paper there were three types: zero, quasi-identity, and random.
2. (29.04.2026) Almost all add layer modules were implemented I added a new class delete layer it is not finished I think module analyser will require a lot of upgrade to make it work we need functions that will get all layers before a given layer and all layers after, also some function of required type, I'm starting to wonder if this is not too limited maybe that what layers are considered "adjustable" should be written from configuration layer for example don't work on nn.Linear work only on FeedForward which can be a custom module (common in LLMs)
3. (30.04.2026) Delete layer action is almost ready but regression test seems to fail in 1 particular scenario and be a bit unstable, 
	1. It fails when we add few (<10) layers and then remove those 10 for some reason 1 is always left Maybe it doesn't handle the case when all the inputs come from add modules ?
	2. It fails when we add and then remove a lot of layers maybe the same reason as 1.
4. (05.05.2026) I found a fix delete layer was not working due to wrong detection of hidden layers the fix works better but still it doesn't get what is input layer only knows output layer I think I fixed it but function for detecting that needs some optimization also I discovered an issue with adding layers it looks like we can add a layer which is after the output layer The error was that gm.graph.inserting_after was used during adding res layer but instead we should use: insert before
5. (06.05.2026) A lot of problems were fixed but there is a new more general one after removing layers we can be left with many intermediate modules that don't do anything like: ![[Pasted image 20260506222448.png]] Those were not added during the process of removing it is an effect of removing everything else 
6. (09.05.2026) I planned how to remove neurons but to test it i used bigger module and then i had problems with types so I added "EDITABLE_MODULES" to config and I start to think this is going into wrong design way we need a better pattern there but I continue to see if it will work 
7. (10.05.2026) Po najnowszych testach na modelach które mają dużo różnych ilości neuronów zauważyłem że potrzebny jest Q identity module do łączenia warstw o innych rozmiarach przy usuwaniu warstw Jednak nie skorzystałem z Q identity jest to coś do zbadania. Drugi problem na którym skończyłem to że AddSeqConvLayer nie działa dodawanie warstw pomiędzy nie działa (Dodałem też logowanie trochę drobnych poprawek)
8. (16.05.2026) Cursor created some kind of module that use fx to mock the output of the layer and that way get true in and out shape, I didn't know it can be used cursor use it in some weird illogical part but when i started to read about it that can be solution to generalization and types limitation with this i need to refactor modules that generate actions but it seems to be much better general way to not care about layer type during adding a layer only what shapes it returns ! This way we can even add linear layers between something unpredictable like batch norm and 1d conv if situation will choose that way ! After those fixes the resnet model was changing successfully: ![[Pasted image 20260516173559.png|669]] Progress of learning looked good:![[resnetallactions.png|428]]
   But the action counter shows some actions have problem with generating: 
   2026-05-16 17:35:03,926 | INFO     | action          | count
   2026-05-16 17:35:03,926 | INFO     | AddResConvLayer | 34
   2026-05-16 17:35:03,926 | INFO     | AddSeqConvLayer | 14
   2026-05-16 17:35:03,926 | INFO     | DelLayer        | 2
   Also I think the: Dotted Module Names should be removed that idea didn't work 
9. (18.05.2026) I fixed unit tests and when i switched the conv actions off the seq were back again: 
   action      | count
   2026-05-18 08:36:19,706 | INFO     | ------------+------
   2026-05-18 08:36:19,706 | INFO     | AddResLayer | 30
   2026-05-18 08:36:19,706 | INFO     | AddSeqLayer | 12
   2026-05-18 08:36:19,706 | INFO     | DelLayer    | 8
   Because we got so many conv actions the seq actions are hard to use which seems like a new problem for research After rerunning it 200 times: ![[resnethistory200iters.png|248]]and we hit all of the actions: 
   2026-05-18 08:57:44,266 | INFO     | action                    | count
   2026-05-18 08:57:44,266 | INFO     | ---------------------+------
   2026-05-18 08:57:44,267 | INFO     | AddResConvLayer   | 143
   2026-05-18 08:57:44,267 | INFO     | AddResLayer           | 1
   2026-05-18 08:57:44,267 | INFO     | AddSeqConvLayer  | 39
   2026-05-18 08:57:44,267 | INFO     | AddSeqLayer          | 3
   2026-05-18 08:57:44,267 | INFO     | DelLayer                 | 14


