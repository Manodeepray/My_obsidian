[[tamper_res_algo.excalidraw]]



imeta training.. where i harden a model against harmful  ft .. then use hyper network to generate lora weights that will train to try to get the model to give harmful information....that will attacked model will then make logprobs to do npo/reverse dpo to harden the model even more.. will this work .. be critical



meta training.. where i harden a model against harmful  ft using distillation... then use hyper network to for teacher that will be somehow trained against the hardened student model an try to break the student model and harmful information.. how will this work so that the studne doesnt get weak but teacher gets strong.... then that teacher will  make logprobs to do npo/reverse dpo to harden the student model even more.. will this work .. be critical