influence of each training point -> for each val point

not feasible -> for pretraining data

so go for finetuning unlearning

or if we have the pretraining data .. find the influence for training data for that specific test_point



influence part

find alternative to datainf

==not hessian vector== ,  use first order derivatives or alternative to hvp

use inverse log of this for ==faster computation==


find some method to do all the ==calculation on gpu== instead of memory - quantize the lora grads to 1.38 bits

instead of unlearning do  -> after finetuning of all data .. use influence to find most influential then + deep unlearning to get the forget dataset 
->try to do adverserial attack on that datapoints ..like  put noise and overfit a lot  , finetune the lora adapter  weights 

hyperparameter tuning -> take 2 at a time , 4 , then 6 ...

train a lora model to not answer the ones form inf_unlearn dataset -> inference level unlearning , 



