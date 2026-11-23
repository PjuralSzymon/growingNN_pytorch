1. [  ] Generators of actions are missing implementation with safeguards against creating layers that are too large.
2.  [  ] Residual and sequential actions currently do not support convolutional layers as input layers.
3.  [  ] Adding linear layers have no check to edit only LInear layers it only cahnges not convolutional but there can be others in bigger modules 
4.  [  ] Action generation is not general enought we are specifying very specific types of conenction and cahnges beetwen very specific types what about more general types or very specific types like dropout , norms and so on ( Remove EDITABLE_MODULES)
5. [  ] Detecting if a layer is hidden is badly written
6. [  ] Generating seimualiton dataset is depracated and need more reaserch 
7. [  ] Monte Carlo 2: UCB1 cannot tell candidates apart. Scores are accuracies about 0.50-0.58 that differ by 0.01-0.03, while the legacy exploration term C*log(N)/n is about 3.9 at 50 candidates, so visits are handed out in turn. Normalize the reward at each node (gain over the parent, divided by measurement noise, or min-max among siblings), use the sqrt form (MCTS_UCB1_USE_SQRT=True) and tune C on that scale. Check offline first on the Experiment 018 Phase 15a measurements (16 arms x 3 measurements x 5 settings, real result known).
8. [  ] Monte Carlo 2: the search budget is too small for UCB1 to exploit: 24 s per candidate at 16-25 s per measurement is about 1 visit each. UCB1 needs many more visits than candidates. Give about 75-100 s per candidate (3-4 measurements) and drop Dropout and shrinking actions from the candidates.
9. [  ] Monte Carlo 2 depth is unfair to shallow nodes: a grandchild continues from the trained child model, so it is measured after twice the simulation epochs and its raw accuracy is not comparable with the child's own score. Its mean also mixes good and bad random follow-ups. Score every node by its gain over its parent and back up the best continuation (max) instead of the mean of all. Verify the training-length bias on Phase 12/14 simulation files first.
10. [  ] Scoring setting from Experiment 018 Phase 15a: 10 scoring epochs on 10% of the training data ranks candidates best (Spearman 0.77-0.79 against the mean of the last 5 real epochs); 15 epochs is worst. Use it in the next full-growth run.





# Nice to haves/Ideas
1. [ ] It would be nice to also add posibility to change the default type on which we are working for example somone created a block called FeedForward and wants to operate on this as a deafult layer used to grow not nn.Linear for example in LLMs it can ba a single transormer block 
2. [ ] Actions need deeper research in terms of what configuration can be best for global use, for example how to configure weight initialization range for non-zero residual layers.
3. [ ] I didn't use the growingNN orginal appraoch to delete layers I didn't use the Q identity because I thought it will be more benefitial to not use it so when we delet layer we delete only those for which the shapes will much so no additional reshaping is needed thsoe two aprpaoch with QI and more deletion can be reaserched some day. 
4. [ ] Analyze the Learning Rate Scheduler there is a risk that the current config is not stable 
5. [ ] Add supportr for adding droput / avg pool maybe generalization of seq and res actions to be parameterized with layer type 
6. [ ] 
