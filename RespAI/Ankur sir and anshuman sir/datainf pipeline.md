setup the repo

```
python3 src/generate_sentence-math_datasets.py
```


```
python /YOUR-DATAINF-PATH/DataInf/src/sft_trainer.py \
    --model_name /YOUR-LLAMA-PATH/llama/models_hf/llama-2-13b-chat \
    --dataset_name /YOUR-DATAINF-PATH/DataInf/datasets/math_with_reason_train.hf \
    --output_dir /YOUR-DATAINF-PATH/DataInf/models/math_with_reason_13bf \
    --dataset_text_field text \
    --load_in_8bit \
    --use_peft
```

then go  to the 

