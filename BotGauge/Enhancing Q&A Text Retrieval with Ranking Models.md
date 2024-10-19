

arXiv:2409.07691v1 [cs.IR] 12 Sep 2024

# Enhancing Q&A Text Retrieval with Ranking Models: Benchmarking, fine-tuning and deploying Rerankers for RAG

Gabriel de Souza P. MoreiraNVIDIASão PauloBrazil[gmoreira@nvidia.com](mailto:gmoreira@nvidia.com)Ronay AkNVIDIASarasotaUSA[ronaya@nvidia.com](mailto:ronaya@nvidia.com)Benedikt SchiffererNVIDIABerlinGermany[bschifferer@nvidia.com](mailto:bschifferer@nvidia.com)Mengyao XuNVIDIASanta ClaraUSA[mengyaox@nvidia.com](mailto:mengyaox@nvidia.com)Radek OsmulskiNVIDIABrisbaneAustralia[rosmulski@nvidia.com](mailto:rosmulski@nvidia.com)Even OldridgeNVIDIAVancouverCanada[eoldridge@nvidia.com](mailto:eoldridge@nvidia.com)

(2018)

###### Abstract.

Ranking models play a crucial role in enhancing overall accuracy of text retrieval systems. These multi-stage systems typically utilize either dense embedding models or sparse lexical indices to retrieve relevant passages based on a given query, followed by ranking models that refine the ordering of the candidate passages by its relevance to the query.

This paper benchmarks various publicly available ranking models and examines their impact on ranking accuracy. We focus on text retrieval for question-answering tasks, a common use case for Retrieval-Augmented Generation systems. Our evaluation benchmarks include models some of which are commercially viable for industrial applications.

We introduce a state-of-the-art ranking model, NV-RerankQA-Mistral-4B-v3, which achieves a significant accuracy increase of  14% compared to pipelines with other rerankers. We also provide an ablation study comparing the fine-tuning of ranking models with different sizes, losses and self-attention mechanisms.

Finally, we discuss challenges of text retrieval pipelines with ranking models in real-world industry applications, in particular the trade-offs among model size, ranking accuracy and system requirements like indexing and serving latency / throughput.

Text retrieval, ranking models, embedding models, retrieval-augmented generation, rag pipelines, model deployment, transformers.


## 1.Introduction

Text retrieval is a core component for many information retrieval applications like search, Question-Answering (Q&A) and recommender systems. More recently, text retrieval has been leveraged by Retrieval-Augmented Generation (RAG)(Lewis et al., [2020](https://arxiv.org/html/2409.07691v1#bib.bib16); Ram et al., [2023](https://arxiv.org/html/2409.07691v1#bib.bib28)) systems, empowering Large Language Models (LLM) with external and up-to-date context.

Text embedding models represent variable-length text as a fixed dimension vector that can be used for downstream tasks. They are key for effective text retrieval, as they can semantically match pieces of textual content that can be symmetric (e.g. similar sentences or documents) or asymmetric (question and passages that might containing its answer).

Embedding models are based on the Transformer architecture. Some examples of seminal works are Sentence-BERT (Reimers and Gurevych, [2019](https://arxiv.org/html/2409.07691v1#bib.bib29)), DPR (Karpukhin et al., [2020](https://arxiv.org/html/2409.07691v1#bib.bib13)), E5 (Wang et al., [2022b](https://arxiv.org/html/2409.07691v1#bib.bib33)) and E5-Mistral (Wang et al., [2023](https://arxiv.org/html/2409.07691v1#bib.bib34)). They are typically trained with Constrastive Learning (Chen et al., [2020](https://arxiv.org/html/2409.07691v1#bib.bib4); Karpukhin et al., [2020](https://arxiv.org/html/2409.07691v1#bib.bib13)) as a bi-encoder or late combination model (Zamani et al., [2018](https://arxiv.org/html/2409.07691v1#bib.bib39)), i.e. query and passage are embedded separately and the model is optimized to maximize the similarity between query and relevant (positive) passages and minimize the similarity between query and non-relevant (negative) passages.

Retrieval systems that leverage text embedding models typically split the corpus into small chunks or passages (e.g. sentences or paragraphs), embed those passages and index corresponding embeddings into a vector database. This setup allows efficiently retrieving relevant passages from the embedded query by using Maximum Inner Product Search (MIPS) (Lewis et al., [2020](https://arxiv.org/html/2409.07691v1#bib.bib16)) or another Approximate Nearest-Neighbor (ANN) algorithm.

The MTEB (Muennighoff et al., [2022](https://arxiv.org/html/2409.07691v1#bib.bib23)) is a popular benchmark of text embedding models for different tasks like retrieval, classification, clustering, semantic textual similarity, among others. We can notice from MTEB leaderboard1 that in general the larger the embedding model in terms of parameters the higher the accuracy it can achieve. However, that brings engineering challenges for companies in deploying such systems, as large embedding models can be prohibitively costly or slow to index very large textual corpus / knowledge bases.

For that reason, multi-stage text retrieval pipelines have been proposed to increase indexing and serving throughput, as well as improving the retrieval accuracy. In those pipelines, a sparse and/or dense embedding model are first used to retrieve top-k candidate passages, followed by a ranking model that refines the final ranking of those passages, as illustrated in Figure [1](https://arxiv.org/html/2409.07691v1#S1.F1 "Figure 1 ‣ 1. Introduction ‣ Enhancing Q&A Text Retrieval with Ranking Models: Benchmarking, fine-tuning and deploying Rerankers for RAG").

![Refer to caption](https://arxiv.org/html/2409.07691v1/x1.png)

Figure 1.Illustration of the typical indexing and querying pipelines for multi-stage text retrieval

Ranking models are typically Transformer models that operate as a cross-encoder or early combination model (Zamani et al., [2018](https://arxiv.org/html/2409.07691v1#bib.bib39)) that takes as input both the query and passage pair concatenated and uses the self-attention mechanism to interact more deeply with the query and passage pair, and model their semantic relationship. Ranking models are used to provide relevance predictions only for the top-k candidates retrieved by the retrieval model. They can increase retrieval accuracy and make it possible using smaller embedding model, considerably reducing the indexing time and cost.

In this paper, we present a benchmark of publicly available ranking models for text retrieval and discuss how they affect the ranking accuracy compared to the original ordering provided by different embedding models. To ensure the usefulness of this benchmark for companies, in our experimentation we include certain embedding and ranking models that can be used commercially, i.e. that have a proper license and were not trained on research-only public data such as the popular MS-MARCO (Bajaj et al., [2016](https://arxiv.org/html/2409.07691v1#bib.bib2)) dataset.

The main contributions of this paper are:

- • 
    
    We provide a comprehensive accuracy evaluation of publicly available ranking models with different commercially usable embedding models for Q&A Text Retrieval;
    
- • 
    
    We introduce a state-of-the-art ranking model: NV-RerankQA-Mistral-4B-v3, and describe its architecture, pruning from Mistral 7B and fine-tuning techniques for leading ranking accuracy;
    
- • 
    
    We present an ablation study fine-tuning ranking models on top of different base model sizes. Then for the Mistral base model, we experiment with both point-wise and pair-wise losses and different attention/pooling mechanisms and discuss how these combinations affect the accuracy;
    
- • 
    
    Finally, we discuss trade-off aspects related to deploying text retrieval pipelines with or without a ranking model, like inference latency and embeddings indexing throughput.
    

## 2.Background and Related work

In earlier days of information retrieval, a common approach to refine search results based on sparse retrieval models (e.g. BM25) was to use feature-based learning-to-rank models with point-wise, pair-wise or list-wise losses (Liu et al., [2009](https://arxiv.org/html/2409.07691v1#bib.bib17)).

Neural Ranking Models (NRM) (Zamani et al., [2018](https://arxiv.org/html/2409.07691v1#bib.bib39)), typically feed-forward networks, were proposed to capture interactions between query and document. DeepMatch (Lu and Li, [2013](https://arxiv.org/html/2409.07691v1#bib.bib18)) modeled each text as a sequence of terms to generate a matching score. DRMM (Guo et al., [2016](https://arxiv.org/html/2409.07691v1#bib.bib9)) represents the input texts as histogram-based features. Duet (Mitra et al., [2017](https://arxiv.org/html/2409.07691v1#bib.bib21)) proposed a ranking model composed of two separate neural networks trained jointly, one that matches query and document using local representation and another using learned distributed representation. In (Dehghani et al., [2017](https://arxiv.org/html/2409.07691v1#bib.bib7)) input text was represented with bag-of-words and averaged bag-of-embeddings, and it was proposed three point-wise and pair-wise models, trained using weekly supervised data.

Transformers (Vaswani et al., [2017](https://arxiv.org/html/2409.07691v1#bib.bib31)) have moved natural language processing (NLP) field from manually handcrafting features to learning semantic and deeper text representations. The Deep Learning Track at TREC 2019 (Craswell et al., [2020](https://arxiv.org/html/2409.07691v1#bib.bib6)) hosted an extensive assessment of retrieval models after BERT (Devlin et al., [2018](https://arxiv.org/html/2409.07691v1#bib.bib8)) was introduced by Google, and demonstrated the effectiveness of leveraging pre-trained transformers as ranking models (Hambarde and Proença, [2023](https://arxiv.org/html/2409.07691v1#bib.bib10)).

In (Nogueira and Cho, [2019](https://arxiv.org/html/2409.07691v1#bib.bib24)) BERT was adapted as a passage re-ranker. They proposed leveraging the next sentence prediction training task to feed as input the concatenated query and passage separated by a special token and adding on top of the [CLS] output vector a single layer binary classification model. The authors later proposed in (Nogueira et al., [2019](https://arxiv.org/html/2409.07691v1#bib.bib25)) a multi-stage retrieval pipeline composed by BM25 and two BERT ranking models: one trained with point-wise loss (monoBERT) and the other with a pair-wise loss (duoBERT).

To deal with BERT limitation of 512 tokens, (Yang et al., [1903](https://arxiv.org/html/2409.07691v1#bib.bib37)) proposed splitting documents into sentences for BERT usage. In (Qiao et al., [2019](https://arxiv.org/html/2409.07691v1#bib.bib27)) they experiment with bi-encoder and cross-encoder ranking models based on BERT, the latter being more effective due to the deeper interaction it provides. They also noticed BERT is more effective when working with semantically close query-document pairs compared to using click data for training.

Since then, a number of cross-encoder models based on Transformers have been released from the community 2 3. We discuss and compare some of the most accurate ranking models available publicly in the next section.

## 3.Benchmarking ranking models for Q&A text retrieval

Benchmarking models in terms of accuracy is important to support decision on which models should be used in production pipelines or fine-tuned for domain adaptation.

In this section, we evaluate text retrieval pipelines composed by different embedding and ranking models. To ensure the usefulness of this benchmark for companies, we evaluate the pipelines using three commercially usable embedding models and top-performing ranking models.

We emphasize our evaluation on Question-Answering (Q&A) datasets, as that is a popular application of RAG systems.

### 3.1.Retrieval models

There are many embedding models publicly available for the community, whose accuracy for multiple tasks can be found at the MTEB leaderboard4. Most of those models have being trained on research-only datasets like MS-MARCO and cannot be used commercially.

We evaluate embedding models that can be used for industry text retrieval applications. We select for experiments three embedding models for candidate retrieval, as the emphasis of our experiments is evaluating ranking models:

- • 
    
    Snowflake/snowflake-arctic-embed-l (Merrick et al., [2024](https://arxiv.org/html/2409.07691v1#bib.bib20)) 5 6 (335M params) - Embedding model based on BERT-large, trained with contrastive learning for two training rounds, with in-batch negative samples and hard negatives.
    
- • 
    
    nvidia/nv-embedqa-e5-v5 7 (335M params) - Embedding model based on e5-large-unsupervised trained for multiple rounds on supervised data with contrastive learning, and both in-batch negatives and hard-negatives.
    
- • 
    
    nvidia/nv-embedqa-mistral-7b-v2 8 (7.24B params)- Large embedding model based on Mistral 7B v0.1 (Jiang et al., [2023](https://arxiv.org/html/2409.07691v1#bib.bib12)) 9, modified to use bi-directional attention and average pooling as done in NV-Embed (Lee et al., [2024](https://arxiv.org/html/2409.07691v1#bib.bib15)) and NV-Retriever (Moreira et al., [2024](https://arxiv.org/html/2409.07691v1#bib.bib22)).
    

### 3.2.Ranking models

We describe here the ranking models we evaluated in this investigation, including recent ranking models that perform high on retrieval benchmarks according to their reports. From those, the only reranking models that can be used for commercial purposes from their licences and train data are NV-RerankQA-Mistral-4B-v3, which we introduce in this paper, and mixedbread-ai/mxbai-rerank-large-v1.

- • 
    
    ms-marco-MiniLM-L-12-v2 10 (33M params) - fine-tuned on top of the MiniLMv2 model (Wang et al., [2020a](https://arxiv.org/html/2409.07691v1#bib.bib35)) with SentenceTransformers package11 on the MS MARCO passage ranking dataset (Bajaj et al., [2016](https://arxiv.org/html/2409.07691v1#bib.bib2)).
    
- • 
    
    jina-reranker-v2-base-multilingual 12 (278M params) - A multi-lingual ranking model finetuned from XLM-RoBERTa (Conneau et al., [2019](https://arxiv.org/html/2409.07691v1#bib.bib5))
    
- • 
    
    mixedbread-ai/mxbai-rerank-large-v1 13 (435M params) - Largest re-ranker model provided by Mixedbread
    
- • 
    
    bge-reranker-v2-m3 14 (568M params) - A multi-lingual ranking model fine-tuned from BGE M3-Embedding (Chen et al., [2024](https://arxiv.org/html/2409.07691v1#bib.bib3)) with FlagEmbedding package15
    
- • 
    
    NV-RerankQA-Mistral-4B-v3 16 (4B params) - Large and powerful re-ranker that takes as base model a pruned version of Mistral 7B and is fine-tuned with a blend of supervised data with contrastive learning. It is fully described in Section [4](https://arxiv.org/html/2409.07691v1#S4 "4. Fine-tuning a state-of-the-art ranking model: NV-RerankQA-Mistral-4B-v3 ‣ Enhancing Q&A Text Retrieval with Ranking Models: Benchmarking, fine-tuning and deploying Rerankers for RAG").
    

### 3.3.Evaluation setup

The evaluation setup mimics the typical text retrieval indexing and querying pipelines, as previously illustrated in Figure [1](https://arxiv.org/html/2409.07691v1#S1.F1 "Figure 1 ‣ 1. Introduction ‣ Enhancing Q&A Text Retrieval with Ranking Models: Benchmarking, fine-tuning and deploying Rerankers for RAG").

The indexing pipeline takes place first, where a text corpus is chunked into smaller passages. For our evaluation, we use datasets from BEIR (Thakur et al., [2021](https://arxiv.org/html/2409.07691v1#bib.bib30)) datasets, which are already chunked and truncated to max 512 tokens. The chunked passages are embedded using an embedding model and stored in a vector index / database. The querying pipeline then takes place for providing for each query a list with ranked passages for retrieval metrics computation (NDCG@10). In detail, the question is embedded and it is performed a vector search (e.g. using exact or Approximate Nearest Neighbour (ANN) algorithm) on the vector index, returning the top-k most relevant passages for the question. Finally, the top-k (set to 100 in our evaluation experiments) passages are re-ranked with a ranking model to generate the final ordered list.

We perform the evaluation on the three Question-Answering datasets from BEIR (Thakur et al., [2021](https://arxiv.org/html/2409.07691v1#bib.bib30)) retrieval benchmark: Natural Questions (NQ) (Kwiatkowski et al., [2019](https://arxiv.org/html/2409.07691v1#bib.bib14)), HotpotQA (Yang et al., [2018](https://arxiv.org/html/2409.07691v1#bib.bib38)) and FiQA (Maia et al., [2018](https://arxiv.org/html/2409.07691v1#bib.bib19)).

### 3.4.Benchmark results

In this section, we provide the benchmark results of text retrieval pipelines with different embedding and ranking models.

In Table [1](https://arxiv.org/html/2409.07691v1#S3.T1 "Table 1 ‣ 3.4. Benchmark results ‣ 3. Benchmarking ranking models for Q&A text retrieval ‣ Enhancing Q&A Text Retrieval with Ranking Models: Benchmarking, fine-tuning and deploying Rerankers for RAG"), we compare pipelines with three commercially usable embedding models (Section [3.1](https://arxiv.org/html/2409.07691v1#S3.SS1 "3.1. Retrieval models ‣ 3. Benchmarking ranking models for Q&A text retrieval ‣ Enhancing Q&A Text Retrieval with Ranking Models: Benchmarking, fine-tuning and deploying Rerankers for RAG")) and their combination with a number of ranking models (Section [3.2](https://arxiv.org/html/2409.07691v1#S3.SS2 "3.2. Ranking models ‣ 3. Benchmarking ranking models for Q&A text retrieval ‣ Enhancing Q&A Text Retrieval with Ranking Models: Benchmarking, fine-tuning and deploying Rerankers for RAG")). Retrieval accuracy is measured with NDCG@10 for Q&A BEIR datasets.

Table 1.Evaluation (NDCG@10) of multi-stage text retrieval pipelines with different embedding and ranking models on text Q&A datasets from BEIR

|   |   |   |   |   |
|---|---|---|---|---|
|Reranker model|Avg.|NQ|HotpotQA|FiQA|
|Embedding: snowflake-arctic-embed-l|0.6100|0.6311|0.7518|0.4471|
|+ ms-marco-MiniLM-L-12-v2|0.5771|0.5876|0.7586|0.3850|
|+ mxbai-rerank-large-v1|0.6077|0.6433|0.7401|0.4396|
|+ jina-reranker-v2-base-multilingual|0.6481|0.6768|0.8165|0.4511|
|+ bge-reranker-v2-m3|0.6585|0.6965|0.8458|0.4332|
|+ NV-RerankQA-Mistral-4B-v3|0.7529|0.7788|0.8726|0.6073|
|Embedding: NV-EmbedQA-e5-v5|0.6083|0.6380|0.7160|0.4710|
|+ ms-marco-MiniLM-L-12-v2|0.5785|0.5909|0.7458|0.3988|
|+ mxbai-rerank-large-v1|0.6077|0.6450|0.7279|0.4502|
|+ jina-reranker-v2-base-multilingual|0.6454|0.6780|0.7996|0.4585|
|+ bge-reranker-v2-m3|0.6584|0.6974|0.8272|0.4506|
|+ NV-RerankQA-Mistral-4B-v3|0.7486|0.7785|0.8470|0.6203|
|Embedding: NV-EmbedQA-Mistral7B-v2|0.7173|0.7216|0.8109|0.6194|
|+ ms-marco-MiniLM-L-12-v2|0.5875|0.5945|0.7641|0.4039|
|+ mxbai-rerank-large-v1|0.6133|0.6439|0.7436|0.4523|
|+ jina-reranker-v2-base-multilingual|0.6590|0.6819|0.8262|0.4689|
|+ bge-reranker-v2-m3|0.6734|0.7028|0.8635|0.4539|
|+ NV-RerankQA-Mistral-4B-v3|0.7694|0.7830|0.8904|0.6350|

We can clearly observe that for smaller embedding models like snowflake-arctic-embed-l and NV-EmbedQA-e5-v5 (335M params), all cross-encoders (except for the small ms-marco-MiniLM-L-12-v2) improve considerably the ranking accuracy compared to the retriever. On the other hand, for the larger NV-EmbedQA-Mistral7B-v2 embedding model, only the large NV-RerankQA-Mistral-4B-v3 reranker is able to improve its accuracy.

The NV-RerankQA-Mistral-4B-v3 reranker provides the highest ranking accuracy for all datasets by a large margin (+14% compared to the second best reranker: bge-reranker-v2-m3). That demonstrates the effectiveness of our adaptation of Mistral 7B as a cross-encoder.

### 3.5.A small note about model licensing

For training NV-RerankQA-Mistral-4B-v3 we have selected only public datasets whose license allows their usage for industry applications. Some other models are released with permissive licenses like Apache 2.0 or MIT, but we do not know which datasets they were trained on or whether they got a special license to use research-only datasets like MS-Marco, for example. Every company should check with its legal team on model licensing for commercial usage.

## 4.Fine-tuning a state-of-the-art ranking model: NV-RerankQA-Mistral-4B-v3

We introduce in this paper the state-of-the-art NV-RerankQA-Mistral-4B-v3, that performs best in our benchmark on text retrieval for Q&A (Section [3.4](https://arxiv.org/html/2409.07691v1#S3.SS4 "3.4. Benchmark results ‣ 3. Benchmarking ranking models for Q&A text retrieval ‣ Enhancing Q&A Text Retrieval with Ranking Models: Benchmarking, fine-tuning and deploying Rerankers for RAG")).

Mistral 7B (Jiang et al., [2023](https://arxiv.org/html/2409.07691v1#bib.bib12)) decoder model has been successfully adopted as embedding models for retrieval when repurposed and fine-tuned with contrastive learning(Wang et al., [2023](https://arxiv.org/html/2409.07691v1#bib.bib34); Lee et al., [2024](https://arxiv.org/html/2409.07691v1#bib.bib15); Moreira et al., [2024](https://arxiv.org/html/2409.07691v1#bib.bib22)).

In this work, we adapted Mistral 7B v0.1 (Jiang et al., [2023](https://arxiv.org/html/2409.07691v1#bib.bib12)) as a ranking model. In order to reduce the number of parameters from the base model, thus its inference compute and memory requirements, we prune it by keeping only the bottom 16 layers out of its 32 layers17. We also modify its self-attention mechanism from uni-directional (causal) to bi-directional, so that for each token it is possible to attend to other tokens in both right and left sides, as that has shown to improve accuracy for Mistral-based embedding models (Lee et al., [2024](https://arxiv.org/html/2409.07691v1#bib.bib15); Moreira et al., [2024](https://arxiv.org/html/2409.07691v1#bib.bib22)).

We feed as input to the model the tokenized question and candidate passage pair, concatenated and separated by a special token. We perform average pooling on the outputs of last Transformer layer and add a feed-forward layer on top that outputs a single-unit with the likelihood of a given passage being relevant to a question.

![Refer to caption](https://arxiv.org/html/2409.07691v1/x2.png)

Figure 2.Architecture of NV-RerankQA-Mistral-4B-v3 cross-encoder, pruned and adapted from Mistral 7B

Cross-encoder ranking models are binary classifiers that discriminate between positive and negative passages. They typically are trained with the binary cross-entropy loss as in Equation [1](https://arxiv.org/html/2409.07691v1#S4.E1 "In 4. Fine-tuning a state-of-the-art ranking model: NV-RerankQA-Mistral-4B-v3 ‣ Enhancing Q&A Text Retrieval with Ranking Models: Benchmarking, fine-tuning and deploying Rerankers for RAG"), where p=ϕ⁢(q,d) is the model predicted likelihood of the passage d being relevant to query q.

|   |   |   |   |
|---|---|---|---|
|(1)||L=−(y⁢log⁡(p)+(1−y)⁢log⁡(1−p))||

Instead, for NV-RerankQA-Mistral-4B-v3 we follow (Wang et al., [2022a](https://arxiv.org/html/2409.07691v1#bib.bib32)) and train the reranker with contrastive learning over the positive and its negative candidates scores using the list-wise InfoNCE loss (Oord et al., [2018](https://arxiv.org/html/2409.07691v1#bib.bib26)), shown in Equation [2](https://arxiv.org/html/2409.07691v1#S4.E2 "In 4. Fine-tuning a state-of-the-art ranking model: NV-RerankQA-Mistral-4B-v3 ‣ Enhancing Q&A Text Retrieval with Ranking Models: Benchmarking, fine-tuning and deploying Rerankers for RAG"), where d+ is a positive relevant passage, d− is one of the N negative passages and τ is the temperature parameter.

|   |   |   |   |
|---|---|---|---|
|(2)||L=−log⁢exp(ϕ(q,d+)/τ))exp(ϕ(q,d+)/τ))+∑i=1Nexp(ϕ(q,d−)/τ))||

The negative candidates used for contrastive learning are mined from the corpus in the data pre-processing stage by using a teacher embedding model. We use the TopK-PercPos hard-negative mining method introduced in (Moreira et al., [2024](https://arxiv.org/html/2409.07691v1#bib.bib22)), configured with maximum negative score threshold as 95% of the positive scores to remove potential false negatives.

We present in Section [5](https://arxiv.org/html/2409.07691v1#S5 "5. Ablation study on fine-tuning ranking models ‣ Enhancing Q&A Text Retrieval with Ranking Models: Benchmarking, fine-tuning and deploying Rerankers for RAG") an ablation study on fine-tuning ranking models, with some experiments focused on our choices for the loss and self-attention mechanism for NV-RerankQA-Mistral-4B-v3.

## 5.Ablation study on fine-tuning ranking models

In this section, we present an ablation study on fine-tuning and comparing different base models as rerankers. For Mistral base model, we also compare choices of self-attention mechanism (unidirectional vs bi-directional) and training losses (binary vs categorical cross-entropy).

For broader comparison, we evaluate the ranking models with the three different embedding models described in Section [3.1](https://arxiv.org/html/2409.07691v1#S3.SS1 "3.1. Retrieval models ‣ 3. Benchmarking ranking models for Q&A text retrieval ‣ Enhancing Q&A Text Retrieval with Ranking Models: Benchmarking, fine-tuning and deploying Rerankers for RAG").

### 5.1.Model size matters

Model size is an important aspect for trading-off model accuracy and inference throughput. For this section we fine-tune and compare three base models with different sizes as ranking models: MiniLM-L12-H384-uncased (Wang et al., [2020b](https://arxiv.org/html/2409.07691v1#bib.bib36)) (33M params) 18, deberta-v3-large (He et al., [2021](https://arxiv.org/html/2409.07691v1#bib.bib11)) 19 (435M) and Mistral 4B (4B params), the latter pruned and modified from Mistral 7B v0.1 (Jiang et al., [2023](https://arxiv.org/html/2409.07691v1#bib.bib12)) as described in Section [4](https://arxiv.org/html/2409.07691v1#S4 "4. Fine-tuning a state-of-the-art ranking model: NV-RerankQA-Mistral-4B-v3 ‣ Enhancing Q&A Text Retrieval with Ranking Models: Benchmarking, fine-tuning and deploying Rerankers for RAG").

Those ranking models are all fine-tuned with the same compute budget (max 4 hours of training in a single server with 8x A100 GPUs) and same train set.

In Table [2](https://arxiv.org/html/2409.07691v1#S5.T2 "Table 2 ‣ 5.1. Model size matters ‣ 5. Ablation study on fine-tuning ranking models ‣ Enhancing Q&A Text Retrieval with Ranking Models: Benchmarking, fine-tuning and deploying Rerankers for RAG"), we present the comparison of fine-tuning those different ranking models, with the same train set and compute budget. Although the pre-training of those base models is different, it is possible to observe a pattern where larger ranking models provide higher retrieval accuracy.

The most accurate model is based on Mistral 4B, but deberta-v3-large is surprisingly accurate for its smaller number of parameters, and is a good candidate architecture for deploying as a cross-encoder, as we discuss in Section [6](https://arxiv.org/html/2409.07691v1#S6 "6. Deployment trade-off considerations for text retrieval pipelines with ranking models ‣ Enhancing Q&A Text Retrieval with Ranking Models: Benchmarking, fine-tuning and deploying Rerankers for RAG").

Table 2.Comparing multi-stage text retrieval pipelines with different-sized ranking models, fine-tuned with the same train set and compute budget. Metric is NDCG@10.

|   |   |   |   |   |
|---|---|---|---|---|
|Reranker model|Avg.|NQ|HotpotQA|FiQA|
|Embedding: snowflake-arctic-embed-l|0.6100|0.6311|0.7518|0.4471|
|+ MiniLM-L12-H384-uncased|0.6227|0.6436|0.8128|0.4118|
|+ deberta-v3-large|0.7277|0.7452|0.8548|0.5832|
|+ Mistral 4B|0.7414|0.7690|0.8681|0.5872|
|Embedding: NV-EmbedQA-e5-v5|0.6083|0.6380|0.7160|0.4710|
|+ MiniLM-L12-H384-uncased|0.6213|0.6438|0.7954|0.4248|
|+ deberta-v3-large|0.7150|0.7441|0.8319|0.5689|
|+ Mistral 4B|0.7366|0.7689|0.8423|0.5987|
|Embedding: NV-EmbedQA-Mistral7B-v2|0.7173|0.7216|0.8109|0.6194|
|+ MiniLM-L12-H384-uncased|0.6355|0.6484|0.8269|0.4312|
|+ deberta-v3-large|0.7413|0.7486|0.8700|0.6055|
|+ Mistral 4B|0.7575|0.7717|0.8857|0.6152|

### 5.2.Causal vs Bi-directional Attention mechanism

In Section [4](https://arxiv.org/html/2409.07691v1#S4 "4. Fine-tuning a state-of-the-art ranking model: NV-RerankQA-Mistral-4B-v3 ‣ Enhancing Q&A Text Retrieval with Ranking Models: Benchmarking, fine-tuning and deploying Rerankers for RAG") we describe that for our adapted Mistral 4B we modified the standard self-attention mechanism of Mistral from uni-directional (causal) to bi-directional attention.

We compare the accuracy with those two self-attention mechanisms in Table [3](https://arxiv.org/html/2409.07691v1#S5.T3 "Table 3 ‣ 5.2. Causal vs Bi-directional Attention mechanism ‣ 5. Ablation study on fine-tuning ranking models ‣ Enhancing Q&A Text Retrieval with Ranking Models: Benchmarking, fine-tuning and deploying Rerankers for RAG"), both using average pooling, and demonstrate the effectiveness of bi-directional attention for allowing deeper interaction among input query and passage tokens.

Table 3.Comparing NDCG@10 of Mistral 4B reranker fine-tuned with different attention mechanisms.

|   |   |   |   |   |
|---|---|---|---|---|
|Reranker model|Avg.|NQ|HotpotQA|FiQA|
|Embedding: snowflake-arctic-embed-l|0.6100|0.6311|0.7518|0.4471|
|+ Mistral 4B (unidirectional attention)|0.7312|0.7663|0.8612|0.5660|
|+ Mistral 4B (bidirectional attention)|0.7414|0.7690|0.8681|0.5872|
|Embedding: NV-EmbedQA-e5-v5|0.6083|0.6380|0.7160|0.4710|
|+ Mistral 4B (unidirectional attention)|0.7264|0.7655|0.8372|0.5766|
|+ Mistral 4B (bidirectional attention)|0.7366|0.7689|0.8423|0.5987|
|Embedding: NV-EmbedQA-Mistral7B-v2|0.7173|0.7216|0.8109|0.6194|
|+ Mistral 4B (unidirectional attention)|0.7464|0.7690|0.8781|0.5920|
|+ Mistral 4B (bidirectional attention)|0.7575|0.7717|0.8857|0.6152|

### 5.3.BCE vs InfoNCE Loss

Cross-encoders are typically trained with the point-wise Binary Cross-Entropy (BCE) loss (Equation [1](https://arxiv.org/html/2409.07691v1#S4.E1 "In 4. Fine-tuning a state-of-the-art ranking model: NV-RerankQA-Mistral-4B-v3 ‣ Enhancing Q&A Text Retrieval with Ranking Models: Benchmarking, fine-tuning and deploying Rerankers for RAG")), as we discussed in Section [4](https://arxiv.org/html/2409.07691v1#S4 "4. Fine-tuning a state-of-the-art ranking model: NV-RerankQA-Mistral-4B-v3 ‣ Enhancing Q&A Text Retrieval with Ranking Models: Benchmarking, fine-tuning and deploying Rerankers for RAG").

On the other hand, we fine-tune Mistral 4B with the list-wise InfoNCE loss(Oord et al., [2018](https://arxiv.org/html/2409.07691v1#bib.bib26)) (Equation [2](https://arxiv.org/html/2409.07691v1#S4.E2 "In 4. Fine-tuning a state-of-the-art ranking model: NV-RerankQA-Mistral-4B-v3 ‣ Enhancing Q&A Text Retrieval with Ranking Models: Benchmarking, fine-tuning and deploying Rerankers for RAG")) and contrastive learning.

We experiment with those two losses, both using the same sets of hard-negative passages mined from the corpus, as described in Section [4](https://arxiv.org/html/2409.07691v1#S4 "4. Fine-tuning a state-of-the-art ranking model: NV-RerankQA-Mistral-4B-v3 ‣ Enhancing Q&A Text Retrieval with Ranking Models: Benchmarking, fine-tuning and deploying Rerankers for RAG").

In Table [4](https://arxiv.org/html/2409.07691v1#S5.T4 "Table 4 ‣ 5.3. BCE vs InfoNCE Loss ‣ 5. Ablation study on fine-tuning ranking models ‣ Enhancing Q&A Text Retrieval with Ranking Models: Benchmarking, fine-tuning and deploying Rerankers for RAG"), we can clearly observe the higher retrieval accuracy obtained when using InfoNCE, a list-wise contrastive learning loss trained to maximize the relevance score of the question and positive passage pair, while minimizing the score for question and negative passage pairs.

Table 4.Comparing NDCG@10 of Mistral 4B reranker with bi-directional attention fine-tuned with different losses.

|   |   |   |   |   |
|---|---|---|---|---|
|Reranker model|Avg.|NQ|HotpotQA|FiQA|
|Embedding: snowflake-arctic-embed-l|0.6100|0.6311|0.7518|0.4471|
|+ Mistral 4B (BCE loss)|0.7230|0.7375|0.8609|0.5706|
|+ Mistral 4B (InfoNCE loss)|0.7414|0.7690|0.8681|0.5872|
|Embedding: NV-EmbedQA-e5-v5|0.6083|0.6380|0.7160|0.4710|
|+ Mistral 4B (BCE loss)|0.7171|0.7368|0.8357|0.5786|
|+ Mistral 4B (InfoNCE loss)|0.7366|0.7689|0.8423|0.5987|
|Embedding: NV-EmbedQA-Mistral7B-v2|0.7173|0.7216|0.8109|0.6194|
|+ Mistral 4B (BCE loss)|0.7373|0.7394|0.8774|0.5949|
|+ Mistral 4B (InfoNCE loss)|0.7575|0.7717|0.8857|0.6152|

This ablation study explains our choices of using bi-directional attention and InfoNCE loss for fine-tuning NV-RerankQA-Mistral-4B-v3.

## 6.Deployment trade-off considerations for text retrieval pipelines with ranking models

As we discussed before, the model sizes and usage of ranking models have implications on the performance of the deployed text retrieval pipelines and downstream systems that use it, such as RAG applications. Deploying the indexing pipeline to production using a large embedding model would be computationally expensive, especially if the document corpus is large. Furthermore, when we deploy query pipeline to production, it is critical that it can handle a large number of queries in a timely manner and scale on demand. In some cases, it might be possible to improve both the retrieval accuracy and indexing throughput by replacing a single-stage query pipeline of a large embedding model by a two-stage pipeline composed of a smaller embedding model and a ranking model.

We conduct performance experiments with our models optimized20 with TensorRT21 and deployed on Triton Inference Server22. In Table [5](https://arxiv.org/html/2409.07691v1#S6.T5 "Table 5 ‣ 6. Deployment trade-off considerations for text retrieval pipelines with ranking models ‣ Enhancing Q&A Text Retrieval with Ranking Models: Benchmarking, fine-tuning and deploying Rerankers for RAG"), we present the average query embedding latency and passages indexing throughput for two embedding models with different sizes. Although the time difference to embed a single query with those two models will not compromise the overall online retrieval latency, the indexing time of the chunked passages of the textual corpus will take 8.2x longer with the larger embedding model, which results in higher compute/cost requirements when re-indexing is needed (e.g. when corpus or embedding model changes).

Table 5.Average query embedding latency (batch size=1 question with 20 tokens) and passages indexing throughput (batch size=64 passages with 512 tokens), with models converted to FP-16 and TensorRT deployed on Triton Inference Server on a single H100-HBM3-80GB GPU24

. Embedding modelQuery embedding latencyPassages indexing throughputNV-EmbedQA-E5-v55.1 ms558.4 passages/secNV-EmbedQA-Mistral7B-v219.8 ms68.7 passages/sec

Thus, by using a two-stage pipeline with NV-EmbedQA-E5-v5 embedding and NV-RerankQA-Mistral-4B-v3 ranking models instead of a single-stage pipeline with NV-EmbedQA-Mistral7B-v2, in addition to achieving higher retrieval accuracy (see Table [1](https://arxiv.org/html/2409.07691v1#S3.T1 "Table 1 ‣ 3.4. Benchmark results ‣ 3. Benchmarking ranking models for Q&A text retrieval ‣ Enhancing Q&A Text Retrieval with Ranking Models: Benchmarking, fine-tuning and deploying Rerankers for RAG")), we will also reduce the indexing time by 8.2x.

We also have to consider the latency the ranking model adds to the query pipeline. In our example, after the query is embedded with NV-EmbedQA-E5-v5 (in 5.1 ms), candidate relevant passages are retrieved from a vector database using ANN, and the top-40 passages will be provided to a ranking model for final ranking. The 40 candidate passages would be scored on their relevancy with respect to the query by the NV-RerankQA-Mistral-4B-v3 model in total 266 ms on average, which might add a reasonable time to the overall query pipeline latency depending on the non-functional requirements for the system. That could be improved by distributing the ranking scoring requests of those 40 candidates among multiple GPUs / Triton Inference Server instances. Another option would be deploying a model with a good trade-off between size/latency and accuracy, like deberta-v3-large25, which is much smaller (435M) than NV-RerankQA-Mistral-4B-v3 (4B) and might provide a good ranking accuracy as discussed in the ablation study on Table [2](https://arxiv.org/html/2409.07691v1#S5.T2 "Table 2 ‣ 5.1. Model size matters ‣ 5. Ablation study on fine-tuning ranking models ‣ Enhancing Q&A Text Retrieval with Ranking Models: Benchmarking, fine-tuning and deploying Rerankers for RAG").

In summary, the decision of which models to include in a text retrieval pipeline should consider the business requirements for retrieval accuracy and system requirements on indexing throughput and serving latency. For example, if the text corpus to index is huge, probably indexing throughput will be the bottleneck and smaller embedding models should be used. On the other hand, if serving latency requirements are very strict, a fast query pipeline is more critical, and large ranking models should be avoided.

## 7.Conclusion

In this paper, we provide a comprehensive evaluation of multi-stage text retrieval pipelines for Question-Answering, a common use case for RAG applications. The evaluated pipelines composed of commercially viable embedding models and state-of-the-art ranking models. We introduce the NV-RerankQA-Mistral-4B-v3, that provides the best ranking accuracy by a large margin in our benchmark and is commercially usable for industrial applications.

We describe how we adapted the decoder-only Mistral 7B and fine-tuned it as a cross-encoder, pruning the base model and modifying its attention and pooling mechanism to build the NV-RerankQA-Mistral-4B-v3.

We also provide an ablation study comparing the fine-tuning of different-sized base models as cross-encoders, and highlight the relationship between their number of parameters and ranking accuracy. We also compare the benefits of leveraging bi-directional attention and InfoNCE loss for training a Mistral cross-encoder.

Finally, we discussed important deployment considerations for real-world text retrieval systems, with respect to trading-off model size, retrieval accuracy and systems requirements like serving latency and indexing throughput.

## References

- (1)↑
- Bajaj et al. (2016)↑Payal Bajaj, Daniel Campos, Nick Craswell, Li Deng, Jianfeng Gao, Xiaodong Liu, Rangan Majumder, Andrew McNamara, Bhaskar Mitra, Tri Nguyen, et al. 2016.Ms marco: A human generated machine reading comprehension dataset._arXiv preprint arXiv:1611.09268_ (2016).
- Chen et al. (2024)↑Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian, and Zheng Liu. 2024.Bge m3-embedding: Multi-lingual, multi-functionality, multi-granularity text embeddings through self-knowledge distillation._arXiv preprint arXiv:2402.03216_ (2024).
- Chen et al. (2020)↑Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. 2020.A simple framework for contrastive learning of visual representations. In _International conference on machine learning_. PMLR, 1597–1607.
- Conneau et al. (2019)↑Alexis Conneau, Kartikay Khandelwal, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek, Francisco Guzmán, Edouard Grave, Myle Ott, Luke Zettlemoyer, and Veselin Stoyanov. 2019.Unsupervised cross-lingual representation learning at scale._arXiv preprint arXiv:1911.02116_ (2019).
- Craswell et al. (2020)↑Nick Craswell, Bhaskar Mitra, Emine Yilmaz, Daniel Campos, and Ellen M Voorhees. 2020.Overview of the TREC 2019 deep learning track._arXiv preprint arXiv:2003.07820_ (2020).
- Dehghani et al. (2017)↑Mostafa Dehghani, Hamed Zamani, Aliaksei Severyn, Jaap Kamps, and W Bruce Croft. 2017.Neural ranking models with weak supervision. In _Proceedings of the 40th international ACM SIGIR conference on research and development in information retrieval_. 65–74.
- Devlin et al. (2018)↑Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2018.Bert: Pre-training of deep bidirectional transformers for language understanding._arXiv preprint arXiv:1810.04805_ (2018).
- Guo et al. (2016)↑Jiafeng Guo, Yixing Fan, Qingyao Ai, and W Bruce Croft. 2016.A deep relevance matching model for ad-hoc retrieval. In _Proceedings of the 25th ACM international on conference on information and knowledge management_. 55–64.
- Hambarde and Proença (2023)↑Kailash A. Hambarde and Hugo Proença. 2023.Information Retrieval: Recent Advances and Beyond._IEEE Access_ 11 (2023), 76581–76604.[https://doi.org/10.1109/ACCESS.2023.3295776](https://doi.org/10.1109/ACCESS.2023.3295776)
- He et al. (2021)↑Pengcheng He, Jianfeng Gao, and Weizhu Chen. 2021.Debertav3: Improving deberta using electra-style pre-training with gradient-disentangled embedding sharing._arXiv preprint arXiv:2111.09543_ (2021).
- Jiang et al. (2023)↑Albert Q Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, et al. 2023.Mistral 7B._arXiv preprint arXiv:2310.06825_ (2023).
- Karpukhin et al. (2020)↑Vladimir Karpukhin, Barlas Oğuz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. 2020.Dense passage retrieval for open-domain question answering._arXiv preprint arXiv:2004.04906_ (2020).
- Kwiatkowski et al. (2019)↑Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, et al. 2019.Natural questions: a benchmark for question answering research._Transactions of the Association for Computational Linguistics_ 7 (2019), 453–466.
- Lee et al. (2024)↑Chankyu Lee, Rajarshi Roy, Mengyao Xu, Jonathan Raiman, Mohammad Shoeybi, Bryan Catanzaro, and Wei Ping. 2024.NV-Embed: Improved Techniques for Training LLMs as Generalist Embedding Models._arXiv preprint arXiv:2405.17428_ (2024).
- Lewis et al. (2020)↑Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al. 2020.Retrieval-augmented generation for knowledge-intensive nlp tasks._Advances in Neural Information Processing Systems_ 33 (2020), 9459–9474.
- Liu et al. (2009)↑Tie-Yan Liu et al. 2009.Learning to rank for information retrieval._Foundations and Trends® in Information Retrieval_ 3, 3 (2009), 225–331.
- Lu and Li (2013)↑Zhengdong Lu and Hang Li. 2013.A deep architecture for matching short texts._Advances in neural information processing systems_ 26 (2013).
- Maia et al. (2018)↑Macedo Maia, Siegfried Handschuh, André Freitas, Brian Davis, Ross McDermott, Manel Zarrouk, and Alexandra Balahur. 2018.Www’18 open challenge: financial opinion mining and question answering. In _Companion proceedings of the the web conference 2018_. 1941–1942.
- Merrick et al. (2024)↑Luke Merrick, Danmei Xu, Gaurav Nuti, and Daniel Campos. 2024.Arctic-Embed: Scalable, Efficient, and Accurate Text Embedding Models._arXiv preprint arXiv:2405.05374_ (2024).
- Mitra et al. (2017)↑Bhaskar Mitra, Fernando Diaz, and Nick Craswell. 2017.Learning to match using local and distributed representations of text for web search. In _Proceedings of the 26th international conference on world wide web_. 1291–1299.
- Moreira et al. (2024)↑Gabriel de Souza P Moreira, Radek Osmulski, Mengyao Xu, Ronay Ak, Benedikt Schifferer, and Even Oldridge. 2024.NV-Retriever: Improving text embedding models with effective hard-negative mining._arXiv preprint arXiv:2407.15831_ (2024).
- Muennighoff et al. (2022)↑Niklas Muennighoff, Nouamane Tazi, Loïc Magne, and Nils Reimers. 2022.MTEB: Massive text embedding benchmark._arXiv preprint arXiv:2210.07316_ (2022).
- Nogueira and Cho (2019)↑Rodrigo Nogueira and Kyunghyun Cho. 2019.Passage Re-ranking with BERT._arXiv preprint arXiv:1901.04085_ (2019).
- Nogueira et al. (2019)↑Rodrigo Nogueira, Wei Yang, Kyunghyun Cho, and Jimmy Lin. 2019.Multi-stage document ranking with BERT._arXiv preprint arXiv:1910.14424_ (2019).
- Oord et al. (2018)↑Aaron van den Oord, Yazhe Li, and Oriol Vinyals. 2018.Representation learning with contrastive predictive coding._arXiv preprint arXiv:1807.03748_ (2018).
- Qiao et al. (2019)↑Yifan Qiao, Chenyan Xiong, Zhenghao Liu, and Zhiyuan Liu. 2019.Understanding the Behaviors of BERT in Ranking._arXiv preprint arXiv:1904.07531_ (2019).
- Ram et al. (2023)↑Ori Ram, Yoav Levine, Itay Dalmedigos, Dor Muhlgay, Amnon Shashua, Kevin Leyton-Brown, and Yoav Shoham. 2023.In-context retrieval-augmented language models._Transactions of the Association for Computational Linguistics_ 11 (2023), 1316–1331.
- Reimers and Gurevych (2019)↑Nils Reimers and Iryna Gurevych. 2019.Sentence-bert: Sentence embeddings using siamese bert-networks._arXiv preprint arXiv:1908.10084_ (2019).
- Thakur et al. (2021)↑Nandan Thakur, Nils Reimers, Andreas Rücklé, Abhishek Srivastava, and Iryna Gurevych. 2021.Beir: A heterogenous benchmark for zero-shot evaluation of information retrieval models._arXiv preprint arXiv:2104.08663_ (2021).
- Vaswani et al. (2017)↑Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017.Attention is all you need._Advances in neural information processing systems_ 30 (2017).
- Wang et al. (2022a)↑Liang Wang, Nan Yang, Xiaolong Huang, Binxing Jiao, Linjun Yang, Daxin Jiang, Rangan Majumder, and Furu Wei. 2022a.Simlm: Pre-training with representation bottleneck for dense passage retrieval._arXiv preprint arXiv:2207.02578_ (2022).
- Wang et al. (2022b)↑Liang Wang, Nan Yang, Xiaolong Huang, Binxing Jiao, Linjun Yang, Daxin Jiang, Rangan Majumder, and Furu Wei. 2022b.Text embeddings by weakly-supervised contrastive pre-training._arXiv preprint arXiv:2212.03533_ (2022).
- Wang et al. (2023)↑Liang Wang, Nan Yang, Xiaolong Huang, Linjun Yang, Rangan Majumder, and Furu Wei. 2023.Improving text embeddings with large language models._arXiv preprint arXiv:2401.00368_ (2023).
- Wang et al. (2020a)↑Wenhui Wang, Hangbo Bao, Shaohan Huang, Li Dong, and Furu Wei. 2020a.Minilmv2: Multi-head self-attention relation distillation for compressing pretrained transformers._arXiv preprint arXiv:2012.15828_ (2020).
- Wang et al. (2020b)↑Wenhui Wang, Furu Wei, Li Dong, Hangbo Bao, Nan Yang, and Ming Zhou. 2020b.Minilm: Deep self-attention distillation for task-agnostic compression of pre-trained transformers._Advances in Neural Information Processing Systems_ 33 (2020), 5776–5788.
- Yang et al. (1903)↑Wei Yang, Haotian Zhang, and Jimmy Lin. 1903.Simple applications of BERT for ad hoc document retrieval. CoRR abs/1903.10972 (2019)._URL: http://arxiv. org/abs/1903.10972_ (1903).
- Yang et al. (2018)↑Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W Cohen, Ruslan Salakhutdinov, and Christopher D Manning. 2018.HotpotQA: A dataset for diverse, explainable multi-hop question answering._arXiv preprint arXiv:1809.09600_ (2018).
- Zamani et al. (2018)↑Hamed Zamani, Mostafa Dehghani, W Bruce Croft, Erik Learned-Miller, and Jaap Kamps. 2018.From neural re-ranking to neural ranking: Learning a sparse representation for inverted indexing. In _Proceedings of the 27th ACM international conference on information and knowledge management_. 497–506.