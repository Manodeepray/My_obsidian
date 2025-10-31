

just ... lora on base model -> long f.t. run on adv dataset -> get logprobs -> detach lora --> new lora params -->long f.t. run safe dataset --> get logprobs --detach

then dual distillation etc .. and  have some param efficient techniques .. like lora , reft , hyper network etc

