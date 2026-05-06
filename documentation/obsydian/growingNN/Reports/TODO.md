1. [  ] Generators of actions are missing implementation with safeguards against creating layers that are too large.
2.  [  ] Residual and sequential actions currently do not support convolutional layers as input layers.
3.  [  ] Adding linear layers have no check to edit only LInear layers it only cahnges not convolutional but there can be others in bigger modules 
4.  [  ] Action generation is not general enought we are specifying very specific types of conenction and cahnges beetwen very specific types what about more general types or very specific types like dropout , norms and so on 
5. [  ] Detecting if a layer is hidden is badly written





# Nice to haves/Ideas
1. [  ] It would be nice to also add posibility to change the default type on which we are working for example somone created a block called FeedForward and wants to operate on this as a deafult layer used to grow not nn.Linear for example in LLMs it can ba a single transormer block 
2. Actions need deeper research in terms of what configuration can be best for global use, for example how to configure weight initialization range for non-zero residual layers.
