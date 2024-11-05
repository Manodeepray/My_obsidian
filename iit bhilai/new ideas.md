- use similarity instead of RAG for symptom analysis
- quantize models YOLO , use vision transformer for obj det - setup the transformer similar to the YOLO model ingestion structure for training
- lnn
- use vit


prompting techniques for converting natural language tp proper symptom acc to the data 

``` python
symptom_imput = ""
prompt = f"from the given {symptom_imput} convert it into : max 10 year "
```
make sure it goes through the symptoms iteratively



o/p : proper data processing and chunking , reranking , use streamlit

