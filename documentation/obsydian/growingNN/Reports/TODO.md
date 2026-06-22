1. [  ] Generators of actions are missing implementation with safeguards against creating layers that are too large.
2.  [  ] Residual and sequential actions currently do not support convolutional layers as input layers.
3.  [  ] Adding linear layers have no check to edit only LInear layers it only cahnges not convolutional but there can be others in bigger modules 
4.  [  ] Action generation is not general enought we are specifying very specific types of conenction and cahnges beetwen very specific types what about more general types or very specific types like dropout , norms and so on ( Remove EDITABLE_MODULES)
5. [  ] Detecting if a layer is hidden is badly written
6. [  ] Generating seimualiton dataset is depracated and need more reaserch 





# Nice to haves/Ideas
1. [ ] It would be nice to also add posibility to change the default type on which we are working for example somone created a block called FeedForward and wants to operate on this as a deafult layer used to grow not nn.Linear for example in LLMs it can ba a single transormer block 
2. [ ] Actions need deeper research in terms of what configuration can be best for global use, for example how to configure weight initialization range for non-zero residual layers.
3. [ ] I didn't use the growingNN orginal appraoch to delete layers I didn't use the Q identity because I thought it will be more benefitial to not use it so when we delet layer we delete only those for which the shapes will much so no additional reshaping is needed thsoe two aprpaoch with QI and more deletion can be reaserched some day. 
4. [ ] Analyze the Learning Rate Scheduler there is a risk that the current config is not stable 
5. [ ] Add supportr for adding droput / avg pool maybe generalization of seq and res actions to be parameterized with layer type 
6. [ ] 
