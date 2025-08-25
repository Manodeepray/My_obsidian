---

excalidraw-plugin: parsed
tags: [excalidraw]

---
==⚠  Switch to EXCALIDRAW VIEW in the MORE OPTIONS menu of this document. ⚠== You can decompress Drawing data with the command palette: 'Decompress current Excalidraw file'. For more info check in plugin settings under 'Saving'


# Excalidraw Data

## Text Elements
TODO

--> DATASET Cereation
    --> harmful
    --> harmless

--> selection on teachers and student

    --> break teacher to make a super bad llm
    --> have an already broken teacher
    --> get good teacher from deepIgnorance


--> eval of teachers on dataset
    --> BAD T --> prompt start with "sure i can ......" or "No i refuse..."
    --> GOOD T

--> why did cuda run out during inference ... long prompt????

--> ALGORITHM implementation: 
    
    --> CB (canary params)

    --> LAT (gradient perturbation)

    -->Core algorithm
        --> SELECT THE LOOPS TYPES
        --> ADD RANDOMNESS
        --> add annealing parameter gamma
        --> GRAD ACCUM IF ANY
        --> 0PTIM, LR , BATCH 
        --> EPOCHS ,  
        --> SEQUENCE 
            --> SEQUENTIAL
            --> CYCLIC
            --> MIXED
        -->collat fn
    
    --> The teacher should provide bad response to good prompts too and 
        good teacher provide refusal to harmful prompts 
    
    --> LOSSES
        --> DPO
        ==> NPO
        ==> KL
        ==> KL rev

--> FULL EVAL
    --> BASED ON FA & HS ANTIDOTE
    --> create the hs and fa bench
    --> refusal
    --> in context
    --> adv finetuning

-->make it memory efficient

--> param offloading

--> efficient training 
    --> batch or something ^S3FXeKvC

to notice
    --> instead on only penalizing o harmful response and rewarding on harmless
    also reward on refusal??

    --> collect/gather canary()canaries for each batch i or for get canary entire epoch/phase

    --> we will be doing
        epoch
            |
            | LEVEL     ---------> SO  FOR EACH LEVEL DO WE GO THROUGH ENTIRE 
            |     |                     BENIGN DATASET OR DIVIDE IT FOR 5 LEVELS
            |     |   BATCH
                                
            

    --> add canary ^beEzYlg9

%%
## Drawing
```compressed-json
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
```
%%