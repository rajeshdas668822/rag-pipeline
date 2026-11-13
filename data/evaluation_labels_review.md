# Evaluation label review sheet (AI-drafted, pending your review)

> **These labels were drafted by an AI assistant from the chunk text and have NOT been human-verified.** Please accept or correct each proposed label.

- Chunk config these ids depend on: `chunk_size=512`, `chunk_overlap=64` (Config defaults).
- **Changing the chunk config invalidates every chunk id** below; regenerate with `python experiments/dump_chunks.py` and re-label.
- Inspect any chunk in `data/chunks_dump.jsonl`. Grades: 3 = directly and fully answers, 2 = substantially answers, 1 = partially/tangentially relevant.
- Chunk ids are shortened to `#N` below; full id is `Mastering RAG 2026_0_N`.

## q001: Why did larger context windows not make retrieval obsolete?
Category: multi_relevant

- `#59` grade **3**:  Augmented Generation RAG Chapter 01 Is RAG dead? Finally, we can say with confidence that large context windows have changed how teams use RAG, but they have not eliminated it. You'll use RAG when building high-volume production systems where cost and latency are most critical. The long context win
- `#14` grade **2**: ings, and stored in a vector database for lookup. A massive context window promised to eliminate this engineering work. Context windows are getting larger between 2024 and 2025Figure 1.2. 8 | Chapter 01 Is RAG dead? However, the enthusiasm for a RAG-less system quickly clashed with the realities of 
- `#19` grade **2**: most relevant passages, and the LLM then generates responses using this focused context. Long-context models, however, must load and process the entire context for each query, regardless of which portions contain relevant information. The cost differences are substantial. Vector retrieval queries ty

## q002: How does cost per query compare between long-context models and RAG?
Category: multi_relevant, same_topic_different_question

- `#19` grade **3**: most relevant passages, and the LLM then generates responses using this focused context. Long-context models, however, must load and process the entire context for each query, regardless of which portions contain relevant information. The cost differences are substantial. Vector retrieval queries ty
- `#21` grade **3**: 00,000-token document (at $1 per 1M tokens). Query 1: "Summarize this document." The model processes 500k tokens. Cost: $0.50. Query 2: "What does it say about Q3 revenue?" The model processes the entire 500k-token document again, plus the first query and answer. Cost: ~$0.50. Query 3: "Compare that
- `#22` grade **2**: te session just cost $1.50. If 1,000 users do this per day, that's $1,500. Compare that to RAG, where each query might cost $0.01 by retrieving only relevant chunks. 11 | Chapter 01 Is RAG dead? The Retrieval Precision Problem Beyond the economic constraints, long-context models face a significant p

## q003: What is the lost in the middle problem?
Category: same_topic_different_question

- `#24` grade **3**: ured effect: performance often drops as the "haystack" grows, especially if the "needle" is placed in the middle of the context, a well-documented "lost in the middle" problem. However, needle in a haystack only measures simple retrieval. The RULER benchmark, now used in evaluating models like Qwen2
- `#23` grade **2**: y and real-world results. Research on long-context performance reveals that models struggle to maintain accuracy as the context size increases. The "needle in a haystack" benchmark tests this directly. In this test, a single, specific fact (the "needle") is inserted at a random location within a lar

## q004: What does the needle in a haystack test measure?
Category: same_topic_different_question

- `#23` grade **3**: y and real-world results. Research on long-context performance reveals that models struggle to maintain accuracy as the context size increases. The "needle in a haystack" benchmark tests this directly. In this test, a single, specific fact (the "needle") is inserted at a random location within a lar
- `#24` grade **2**: ured effect: performance often drops as the "haystack" grows, especially if the "needle" is placed in the middle of the context, a well-documented "lost in the middle" problem. However, needle in a haystack only measures simple retrieval. The RULER benchmark, now used in evaluating models like Qwen2

## q005: When should I fine-tune a model rather than rely on retrieval?
Category: multi_relevant

- `#48` grade **3**: ates or misunderstands domain terminology, retrieval will not fix the issue. Fine-tuning is required in such a case. If a model produces confident answers about information that was updated yesterday, fine-tuning alone cannot help, and here, retrieval becomes necessary. 20 | Chapter 01 Is RAG dead? 
- `#47` grade **3**: uning is effective for correcting recurring errors, enforcing output structure, and adapting tone or style. RAG is necessary when answers depend on changing data, recent documents, or external systems. A model fine-tuned on last quarter’s documentation cannot answer questions about this quarter’s up
- `#46` grade **2**: remains unchanged, but it receives up-to-date context retrieved from external sources. This allows the system to answer questions using data that did not exist when the model was trained. Think of it this way: fine-tuning improves how a model behaves. RAG determines what information the model sees. 

## q006: What are the seven failure points in RAG deployments?
Category: multi_relevant

- `#70` grade **3**:  an Enterprise RAG System Refer to figure 2.2 to see where each failure mode occurs in the RAG pipeline. Seven Production Failure Modes Seven failure pointsFigure 2.2 1. Missing Content (FP1) occurs when the s ystem receives a query that cannot be answered with the available documents. For example, 
- `#71` grade **3**: s numbers. 2. Mi ssed Top-Ranked Documents (FP2) occurs when relevant documents exist but are ranked too low. A correct answer may appear at position 47, while your system only examines the top 10 results. 3. No t in Context (FP3) occurs when relevant documents are retrieved but do not reach the gen
- `#72` grade **3**: ccurs when the answer is present in the context, but the model does not use it. Noise may obscure the relevant passage, or the model may overlook it. 5. Wr ong Format (FP5) occurs when the model does not follow the required formatting. If a query requests a table and the model returns prose, format 
- `#73` grade **2**: stem 6. Incorrect Specificity (FP6) occurs when the level of detail does not m atch the request. A user may ask for an overview and receive a lengthy response, or ask for specifics and receive a high-level summary. 7. Inc omplete (FP7) occurs when the model returns only part of the required informat

## q007: What should the system do when a user asks about something the documents never covered?
Category: different_terminology, multi_relevant

- `#70` grade **3**:  an Enterprise RAG System Refer to figure 2.2 to see where each failure mode occurs in the RAG pipeline. Seven Production Failure Modes Seven failure pointsFigure 2.2 1. Missing Content (FP1) occurs when the s ystem receives a query that cannot be answered with the available documents. For example, 
- `#541` grade **3**: st Before Production RAG systems must know when they don’t know the answer. Negative rejection assesses whether the model will decline to answer a question when none of the contexts provide useful information, rather than giving an incorrect response. As you’ll see in figure 7.7, the model should re
- `#548` grade **2**: ction Test your RAG system with these three scenarios: → Noise filtering: Retrieve 5 documents where only 1 contains the answer. Ask a specific question. Does the model extract from the correct document or blend in irrelevant information? → Nega tive rejection: Ask a question your documents cannot a

## q008: How do input guardrails protect a RAG system?
Category: multi_relevant, same_topic_different_question

- `#78` grade **3**: ssing Content (FP1) by blocking queries that shouldn't receive responses. Input Guardrails → PII Detection: queries often contain na mes, email addresses, phone numbers, or other personal information. If this information passes through logs or monitoring systems, it can create compliance issues. PII
- `#79` grade **3**: mple, an email address is converted into a neutral token, allowing the system to continue processing without storing personal data. → Pr ompt Injection Prevention: some users attempt to override system instructions by adding directives such as “ignore previous instructions.” Prompt injection detecti
- `#80` grade **2**: instructions or restricted data paths. → Topic Filtering: a RAG system often serves a defined domain. Topic filtering ensures that the system only processes queries that fall within that domain. A classifier checks the query and rejects off-topic requests with a clear redirect. This reduces unnecess
- `#81` grade **2**: Enterprise RAG System Guardrail Purpose Method PII Detection Prevent data leakage Regex + NER → placeholders Injection Prevention Block manipulation Pattern matching Topic Filtering Enforce domain boundaries Classification + rejection Code Injection Stop malicious code Syntax detection Token Limits 

## q009: How can personal data in user queries be detected before it reaches the logs?
Category: same_topic_different_question

- `#78` grade **3**: ssing Content (FP1) by blocking queries that shouldn't receive responses. Input Guardrails → PII Detection: queries often contain na mes, email addresses, phone numbers, or other personal information. If this information passes through logs or monitoring systems, it can create compliance issues. PII
- `#81` grade **1**: Enterprise RAG System Guardrail Purpose Method PII Detection Prevent data leakage Regex + NER → placeholders Injection Prevention Block manipulation Pattern matching Topic Filtering Enforce domain boundaries Classification + rejection Code Injection Stop malicious code Syntax detection Token Limits 

## q010: How should the system handle a follow-up like 'What about the cost?' that lacks context?
Category: different_terminology, multi_relevant

- `#84` grade **3**: eo.aiMastering Retrieval Augmented Generation RAG Chapter 02 How to Architect an Enterprise RAG System Query processing converts the user’s input into a form that the retrieval system can handle. Many queries are incomplete or vague, so the system rewrites them to create clear and specific requests.
- `#85` grade **2**: acks context. "What about the cost?" doesn't specify what. Query rewriting turns these unclear inputs into precise requests and helps address Incorrect Specificity (FP6). Refer to figure 2.4, which illustrates three query rewriting techniques. Query rewriting techniquesFigure 2.4 35 | Chapter 02 How
- `#86` grade **3**: a query may depend on previous messages. For example, a user may ask about a product, then request a comparison, and later ask about fees. The final query cannot be processed correctly without the earlier context. History-based rewriting reconstructs a complete version of the query, enabling retriev

## q011: How can a request that asks about two things at once be split up before searching?
Category: different_terminology

- `#87` grade **3**:  features of platinum and gold credit cards" is essentially asking about two things: platinum features and gold features. Subquery decomposition breaks the query into separate parts, retrieves results for each part, and combines them before generation. Similar Query Generation Retrieval may fail whe
- `#86` grade **2**: a query may depend on previous messages. For example, a user may ask about a product, then request a comparison, and later ask about fees. The final query cannot be processed correctly without the earlier context. History-based rewriting reconstructs a complete version of the query, enabling retriev

## q012: What is HyDE and why does it help retrieval?
Category: single_topic

- `#109` grade **3**: tical Document Embeddings (HyDE) HyDE generates a concise synthetic answer to the query and embeds it instead of the original text. This helps retrieval when the query uses different wording from the documents and improves matching across varied terminology. Query Routing Large systems often maintai
- `#57` grade **2**: es information before responding, critical for legal research or medical information where accuracy matters → HyDE: Generates an ideal answer first, then searches for supporting evidence. Works well for complex research questions → Gra ph RAG: Maps relationships between entities for fraud detection 

## q013: How does maximal marginal relevance improve the diversity of retrieved results?
Category: same_topic_different_question

- `#118` grade **3**: may return many chunks from the same document, which limits coverage. Diversity mechanisms increase the range of information provided to the generator, which helps address the Incomplete (FP7) failure point. Maximal Marginal Relevance (MMR) MMR balances relevance and diversity. Instead of selecting 
- `#119` grade **2**: ure 2.8, which illustrates this process. For a query about database optimization, standard retrieval might return 10 chunks all about index tuning. MMR returns chunks about index tuning, query optimization, caching, schema design, and hardware considerations. Maximal marginal relevance (MMR)Figure 2

## q014: What is autocut and how does it filter weak results?
Category: same_topic_different_question

- `#120` grade **3**: milarity scores and identifies where the scores decline significantly. The system uses that point as a threshold and excludes results that fall below it. This prevents the return of documents with weak relevance. For instance, Weaviate's autocut analyzes similarity scores and identifies significant 
- `#119` grade **2**: ure 2.8, which illustrates this process. For a query about database optimization, standard retrieval might return 10 chunks all about index tuning. MMR returns chunks about index tuning, query optimization, caching, schema design, and hardware considerations. Maximal marginal relevance (MMR)Figure 2

## q015: How can I match on small pieces of text but hand the generator the larger enclosing section?
Category: different_terminology, multi_relevant

- `#240` grade **3**: ’s ParentDocumentRetriever strategy divides documents into small chunks for retrieval, but stores references to larger parent documents. Retrieval occurs in small chunks for precision, but the system returns the parent document to the LLM for context. For instance, a research paper might be chunked 
- `#121` grade **3**: thresholds. Recursive retrieval Recursive retrieval uses small chunks for matching to improve precision and then returns the larger sections that contain those chunks. This provides enough context for generation while maintaining accurate matching. (figure 2.9). Embedding lookup and expanded window 

## q016: How does a cross-encoder reranker score a query against a document?
Category: multi_relevant

- `#463` grade **3**: orporate into its response. 182 | Chapter 06 How to Select a Reranker Reranker architectures trade off accuracy, speed, and implementation complexity. Understanding these approaches helps you select the right one for your requirements. Cross-Encoder Reranking Types of Rerankers Cross-encoders proces
- `#464` grade **3**: model receives both inputs concatenated with a separator token, then produces a relevance score. This joint processing allows the model to learn how query terms relate to document terms. Cross-encoder architecture processes the query and document togetherFigure 6.2 183 | Chapter 06 How to Select a R
- `#453` grade **3**:  the most common reranker architecture, concatenate the query and document together, then analyze their interaction through transformer layers. This joint processing captures relevance signals that vector similarity misses. The reranker produces relevance scores for each candidate, then returns the 
- `#452` grade **2**: less useful results. Vector similarity doesn't always correlate perfectly with relevance to the user's specific query. A document about "quarterly revenue trends" might rank highly for a query about "Q3 revenue increase" due to shared vocabulary, even if it doesn't address the specific quarter. Seco

## q017: How much latency does reranking add to a RAG pipeline?
Category: multi_relevant, same_topic_different_question

- `#458` grade **3**: ith focused, relevant context than with noisy, scattered information. Irrelevant chunks introduce facts that the model might incorrectly incorporate into its responses, increasing the risk of hallucination. 180 | Chapter 06 How to Select a Reranker Reranking adds latency to your pipeline. Cross-enco
- `#461` grade **2**: Generation RAG Chapter 06 How to Select a Reranker Table 6.1 summarizes these impacts across retrieval quality, context quality, computational cost, and accuracy: Impact Area Without Reranker With Reranker Tradeoff Retrieval Precision Baseline NDCG +15-48% NDCG improvement Latency increase Context Q
- `#467` grade **2**:  Reranker Cross-encoder reranking works for most production RAG systems, where you can absorb 200-400ms latency and need to rerank 20- 100 candidates. The accuracy gains justify the computational cost for applications where response quality matters more than ultra-low latency. Diagnose Your Latency 
- `#522` grade **2**: > galileo.aiMastering Retrieval Augmented Generation RAG Chapter 06 How to Select a Reranker Table 6.11 maps reranker types to their impact on Galileo metrics: Reranker Type Chunk Attribution Context Adherence Completeness Added Latency No reranker (baseline) Lower Lower Lower 0ms Cross-encoder High

## q018: How many comparisons does pairwise reranking need for 50 documents?
Category: single_topic

- `#473` grade **3**: computationally expensive than pointwise methods. For 20 documents, you need 190 comparisons. For 50 documents, you need 1,225 comparisons. However, pairwise methods often produce more accurate rankings than pointwise approaches because they directly model relative relevance. Pairwise reranking work
- `#472` grade **2**: me (Refer to figure 6.3). Instead of scoring documents independently, the model decides which of two documents is more relevant to the query. This approach captures relative relevance better than absolute scoring. Pairwise reranking comparing documents in pairsFigure 6.3 187 | Chapter 06 How to Sele

## q019: When is LLM listwise reranking worth the several-second wait?
Category: multi_relevant

- `#478` grade **3**: eves higher NDCG scores than pointwise or pairwise methods on BEIR benchmarks. The performance comes at a cost: 4-6 seconds of added latency and higher API fees. Listwise reranking is effective for high-value queries where accuracy justifies 4-6 seconds of latency, offline batch processing where spe
- `#479` grade **3**: .5 guide when to apply this approach. Pros Cons Can reason about relative importance Context window limits candidate count Highest accuracy on benchmarks 4-6 seconds added latency Handles complex ranking scenarios Expensive per query Natural for LLMs Non-deterministic outputs Listwise reranking trad
- `#480` grade **2**:  a medical condition, or analyzing financial data. That's exactly where listwise LLM reranking makes sense. For everyday queries, the wait isn't worth it. 190 | Chapter 06 How to Select a Reranker ColBERT-Style Late Interaction ColBERT (Contextualized Late Interaction over BERT) represents documents

## q020: How does ColBERT late interaction use token embeddings?
Category: multi_relevant

- `#481` grade **3**: uring retrieval, the system compares query token embeddings against document token embeddings through maximum similarity pooling. ColBERT computing token-level interactionsFigure 6.5 Query tokens 191 | Chapter 06 How to Select a Reranker This multi-vector approach enables fine-grained matching at th
- `#482` grade **3**: kens, producing more precise relevance scores than single-vector similarity. For reranking, ColBERT processes all candidate documents through the encoder once, storing the token embeddings. At query time, it computes query token embeddings and performs efficient late interaction with stored document
- `#480` grade **2**:  a medical condition, or analyzing financial data. That's exactly where listwise LLM reranking makes sense. For everyday queries, the wait isn't worth it. 190 | Chapter 06 How to Select a Reranker ColBERT-Style Late Interaction ColBERT (Contextualized Late Interaction over BERT) represents documents
- `#317` grade **2**: gs once. At query time, the system compares query token embeddings against stored document embeddings through maximum similarity pooling. This proves to be faster than running the full model for each query-document pair, while providing more nuanced matching than single-vector approaches. ColBERT v2

## q021: What are phantom hits in reranking?
Category: single_topic

- `#510` grade **3**:  contexts. A query about "Python programming" might highly rank a document about "Python snakes in programming conservation efforts" because both terms appear, but the semantic relationship is wrong. Solution: Use hybrid reranking that combines cross-encoder scores with keyword matching. Require doc
- `#509` grade **2**: leo.aiMastering Retrieval Augmented Generation RAG Chapter 06 How to Select a Reranker Rerankers fail in predictable patterns. Recognizing these failures helps you diagnose and fix ranking problems before they reach production. Common Reranker Failures and Solutions Phantom Hits Context Truncation R

## q022: What happens if a document is longer than the reranker can read?
Category: different_terminology

- `#511` grade **3**: ct any document below 0.3, even if it ranks in the top results. Documents exceeding the reranker's context limit get truncated, losing important information. If your reranker supports 512 tokens but your average document contains 800 tokens, you're evaluating only the first 512 tokens of each docume
- `#512` grade **2**:  as a separate candidate. Rerank all chunks, then aggregate scores by parent document. Alternatively, switch to a longer-context reranker (BGE-M3 for 8K tokens, Voyage Rerank 2 for 16K tokens). 202 | Chapter 06 How to Select a Reranker Length Mismatch Short queries paired with long documents, or lon
- `#516` grade **1**:  top results for each interpretation, or use the retrieval patterns to infer the most likely interpretation. 203 | Chapter 06 How to Select a Reranker Table 6.10 summarizes these failure patterns and their solutions: Failure Mode Symptom Solution Phantom Hits High scores for wrong semantic matches H

## q023: Why might a reranker trained on Wikipedia fail on medical text?
Category: multi_relevant

- `#513` grade **3**: ed query-document pairs struggle when the length ratio deviates significantly from the training data. Solution: Expand short queries by adding terms from the initial retrieval results before reranking. Extract representative sentences from long documents rather than passing entire documents. Match t
- `#514` grade **3**: orly on specialized domains. A reranker trained on Wikipedia and news articles will struggle with medical terminology, legal citations, or technical specifications because it never learned these vocabulary patterns. Solution: Fine-tune the reranker on domain-specific query-document pairs. Even 1,000

## q024: Which reranker can I self-host on my own hardware with an open license?
Category: different_terminology, multi_relevant

- `#494` grade **3**: y). They work across multiple data types, including text, code, JSON, and function selection for LLM tool calls. Benchmarks show mxbai-rerank-large-v2 achieves 57.49 NDCG on BEIR, outperforming both open-source alternatives and closed- source options like Cohere and Voyage. The base model scores 55.
- `#493` grade **2**: Select a Reranker Mixedbread AI mxbai-rerank Mixedbread AI released mxbai-rerank-base-v2 (0.5B parameters) and mxbai- rerank-large-v2 (1.5B parameters). These models use the Qwen-2.5 architecture as their foundation and incorporate a three-stage reinforcement learning training process (GRPO, Contras
- `#499` grade **3**: el capabilities: If you need proven reliability at scale with strong multilingual support, Cohere Rerank 3.5 offers the most mature production option, providing comprehensive language coverage. If you require on-premise deployment, need complete control over your infrastructure, or want the highest 
- `#508` grade **1**: andidates to rerank ColBERT or hybrid Efficient token-level matching Multilingual support needed Cohere, Mixedbread, BGE-M3 100+ language support On-premise deployment required Mixedbread, BGE-M3 Open-source options Very long documents (10K+ tokens) Voyage Rerank 2 16K context window Budget-constrai

## q025: Which reranker supports the longest context window?
Category: single_topic

- `#496` grade **3**: ank 2.5 supports a 32,000 token context length, double that of Rerank 2 and 8x that of Cohere Rerank v3.5. This extended context enables reranking of much longer documents without truncation, handling technical documentation, legal contracts, and research papers more effectively. The model introduce
- `#495` grade **2**: quire on-premises deployment or full control over the reranking infrastructure can run these models on their own hardware. Feature Specification Languages 100+ Context Length 8,192 tokens Deployment Self-hosted (Apache 2.0) Key Strength Highest BEIR score, open-source Mixedbread AI mxbai-rerank spec
- `#498` grade **2**: atency while still outperforming Cohere Rerank v3.5 by 7.16% on standard benchmarks and 10.36% on MAIR. Feature Specification Languages Multilingual Context Length 32,000 tokens combined Deployment API-based Key Strength Longest context window Voyage AI Rerank 2 specificationsTable 6.8c 197 | Chapte

## q026: How should I build a test set to compare rerankers on my own data?
Category: multi_relevant

- `#504` grade **3**: mance 199 | Chapter 06 How to Select a Reranker Test Your Reranker on Real Queries Build a test set from your actual use case: 1. Ex tract 50-100 real user queries from logs 2. Fo r each query, manually identify 3-5 documents that should rank in the top 5 3. In clude hard negatives: similar document
- `#337` grade **2**: elevance quality Higher indicates better ranking of relevant docs Chunk Attribution Used chunks / retrieved chunks Higher means retrieved chunks are more useful Embedding evaluation metricsTable 4.14 Build a test set from your actual use case: 1. Ex tract 100-200 real user queries from logs 2. Fo r 
- `#505` grade **1**: u're evaluating This custom evaluation reveals which reranker works best for YOUR data, not generic benchmarks. Run this test before committing to a production reranker. EXERCISE BEIR scores provide useful comparisons but don't guarantee performance on your specific data. Domain shift affects result

## q027: What does NDCG measure?
Category: multi_relevant, same_topic_different_question

- `#335` grade **3**: an Reciprocal Rank) measures the average position of the first relevant result. MRR = 0.5 means the first relevant result appears at position 2 on average. → NDCG@k (N ormalized Discounted Cumulative Gain) weights relevant results by their position. A relevant result at position 1 contributes more t
- `#502` grade **3**:  (Normalized Discounted Cumulative Gain at 10), measures ranking quality by weighting relevant documents by their position. A relevant document at position 1 contributes more to the score than one at position 10. A perfect ranking (all relevant documents at the top) yields an NDCG@10 of 1.0. Additio
- `#501` grade **2**: yage's 16K context window eliminates chunking headaches. 198 | Chapter 06 How to Select a Reranker The BEIR (Benchmarking IR) benchmark evaluates rerankers across diverse retrieval tasks. BEIR includes 18 datasets spanning different domains: question answering (Natural Questions, HotpotQA), fact ver

## q028: How is mean reciprocal rank defined?
Category: multi_relevant, same_topic_different_question

- `#503` grade **3**: op K results. Recall@10 = 0.8 means 80% of relevant documents appear in the top 10 results. → MRR (M ean Reciprocal Rank) measures the average position of the first relevant result. MRR = 0.5 means the first relevant result appears at position two on average. → MA P@K (Mean Average Precision at K) c
- `#334` grade **2**: ects results dramatically. A model scoring 70 on MTEB might score 40 on your medical records or legal contracts. Your domain vocabulary, document structure, and query patterns differ from MTEB's test sets. Measuring Embedding Performance Key Evaluation Metrics → Recall@k m easures the percentage of 
- `#335` grade **2**: an Reciprocal Rank) measures the average position of the first relevant result. MRR = 0.5 means the first relevant result appears at position 2 on average. → NDCG@k (N ormalized Discounted Cumulative Gain) weights relevant results by their position. A relevant result at position 1 contributes more t

## q029: How do flat, HNSW, IVF and DiskANN indexes differ?
Category: multi_relevant

- `#387` grade **3**: grows, exhaustive distance computation becomes impractical, and approximate indexing methods become necessary. HNSW organizes vectors into a multi-layer graph optimized for fast approximate search. Queries traverse the graph from coarse to fine layers, quickly narrowing the search space. This design
- `#389` grade **3**: oring the vectors themselves. Systems with millions of vectors require substantial memory, and billion-vector deployments become extremely expensive. This memory constraint is why alternative indexes exist. IVF (Inverted File Index) DiskANN IVF partitions the vector space into clusters and assigns e
- `#392` grade **3**: y dimensions: Index Type Recall Search Latency Memory Usage Best For Flat 100% O(n) Baseline <100K vectors HNSW 95-99% O(log n) 2-3x baseline Production use IVF 85-95% O(log n) 0.2-0.5x baseline Memory-constrained deployments DiskANN 90-95% O(log n) 0.02x baseline + SSD Billion-scale corpora Vector 
- `#390` grade **2**: ements by keeping centroids in RAM and storing full vectors on disk. IVF supports much larger corpora than HNSW on the same hardware, often tens or hundreds of millions of vectors. Retrieval quality depends on clustering quality and the nprobe setting, which balances recall and latency. DiskANN exte
- `#385` grade **2**: er primarily in how they index vectors for similarity search. These architectural choices determine recall, latency, memory usage, and update behavior. In practice, four approaches dominate production systems. Flat Indexes HNSW (Hierarchical Navigable Small World) Flat indexes store vectors without 

## q030: How much memory does an HNSW index need compared with IVF for 500,000 vectors?
Category: single_topic

- `#393` grade **3**: verage chunks per document 2. Multiply chunk count by embedding dimension 3. Multiply by 4 bytes per float32 value 4. Add index overhead (2-3x for HNSW, 0.2-0.5x for IVF) A corpus of 100,000 documents averaging 5 chunks each yields 500,000 vectors. At 1,536 dimensions, each vector needs 6KB. That's 
- `#388` grade **2**: sage, since the full graph must remain in RAM. Parameters such as M and efConstruction control recall, build time, and memory overhead, as well as memory limits, eventually require disk-based alternatives. 156 | Chapter 05 How to Select a Vector Database HNSW's speed comes from keeping the entire gr
- `#392` grade **2**: y dimensions: Index Type Recall Search Latency Memory Usage Best For Flat 100% O(n) Baseline <100K vectors HNSW 95-99% O(log n) 2-3x baseline Production use IVF 85-95% O(log n) 0.2-0.5x baseline Memory-constrained deployments DiskANN 90-95% O(log n) 0.02x baseline + SSD Billion-scale corpora Vector 

## q031: How does scalar int8 quantization compare with binary quantization?
Category: multi_relevant

- `#398` grade **3**: ering Retrieval Augmented Generation RAG Chapter 05 How to Select a Vector Database Scalar Quantization Scalar quantization reduces precision by converting floating- point values into lower-precision integers, such as int8 or int4. This significantly lowers memory usage while preserving the structur
- `#399` grade **3**: s adopt it as a default compression strategy when moderate memory savings suffice. It fits interactive workloads that need to balance latency, memory usage, and retrieval quality. Table 5.3 compares quantization approaches: Technique Compression Recall Impact Speed Impact Use Case Binary Very high N
- `#395` grade **2**: recision. At scale, this becomes a practical requirement rather than an optimization. Binary Quantization Product Quantization Binary quantization converts each vector dimension into a single bit based on its sign, which helps in reducing storage requirements. Similarity comparisons then operate on 

## q032: What is the difference between pre-filtering and post-filtering in vector search?
Category: multi_relevant

- `#401` grade **3**: ystem. How the database applies these filters determines whether query performance stays stable as the corpus grows. Pre-filtering vs. Post-filtering Pre-filtering applies metadata conditions first, then runs vector search only on the matching subset. This works well when filters are selective and s
- `#402` grade **3**: xes such as HNSW. The index no longer reflects the true neighborhood structure, which can reduce recall and increase latency. Post-filtering reverses the order. The system performs a vector search on the full index, then removes results that do not meet the metadata conditions. This preserves index 
- `#403` grade **1**: ore vectors if only a small fraction satisfies the filter. Some databases reduce this tradeoff by integrating metadata directly into the index. Systems such as Qdrant and Weaviate incorporate filtering into their indexing and query execution paths, allowing complex Boolean conditions without the sha

## q033: What does the alpha parameter control in hybrid search?
Category: multi_relevant

- `#312` grade **3**: ks both result sets. See figure 4.4. 121 | Chapter 04 How to Select an Embedding Model Pinecone, Weaviate, Qdrant, and Milvus support hybrid search natively. You configure a weight parameter (often called alpha) that balances dense and sparse contributions: → al pha = 1.0: Pure dense retrieval (sema
- `#106` grade **3**: ies require both semantic similarity and exact term matching. Hybrid search implements a dense vector index for semantic similarity (figure 2.6), a sparse inverted index for specific terms, and reranking to combine and order results. The balance between dense and sparse is adjustable through an alph
- `#313` grade **2**: 0.5: Equal weighting (balanced hybrid) → al pha = 0.3: 30% dense, 70% sparse (keyword-focused hybrid) → al pha = 0.0: Pure sparse retrieval (keyword only) This hybrid approach addresses a common RAG failure mode. A query about "2024 Q3 revenue" might retrieve documents discussing revenue in general 
- `#405` grade **2**: valFigure 5.2 162 | Chapter 05 How to Select a Vector Database Weaviate and Elasticsearch execute both searches in parallel and merge results using rank fusion. An alpha parameter controls the balance. Higher values emphasize semantic similarity. Lower values emphasize keyword matching. Optimal valu

## q034: Which embedding approach works best for rare technical terms?
Category: single_topic

- `#308` grade **3**: are zero, with non-zero values marking only specific, important terms. This selectivity makes sparse embeddings efficient for keyword-heavy domains where precise term matching matters (figure 4.3). Models like SPLADE, SPLADE++, and BM25 variants learn which terms are most relevant for retrieval. The
- `#309` grade **3**:  "hypercholesterolemia," that appear infrequently but carry high semantic importance. Sparse embeddings assign these terms high weights, ensuring that queries that mention them retrieve relevant documents even when general semantic similarity is low. Consider sparse embeddings when your queries and 
- `#310` grade **1**:  4.8 for the complete tradeoff analysis. Pros Cons Precise term matching for rare vocabulary Misses synonyms and paraphrases Efficient storage (most values zero) Struggles with semantic similarity Explainable (can see which terms matched) Less mature tooling than dense Works well with specialized te

## q035: Can I shrink embedding vectors to save storage without retraining the model?
Category: different_terminology, multi_relevant

- `#326` grade **3**: aller dimensions are embedded within larger ones, much like Russian nesting dolls. The smallest dimension captures core semantic meaning, while each additional layer adds more nuanced information. During training, the model simultaneously optimizes all these sizes. This means you can truncate the ve
- `#327` grade **3**:  127 | Chapter 04 How to Select an Embedding Model OpenAI's text-embedding-3-small and text- embedding-3-large support this natively. Both models generate up to 3,072-dimensional embeddings, but you can request 256, 512, 768, or 1,536 dimensions at embedding time. Smaller dimensions reduce storage a
- `#325` grade **2**: ddings may produce better results. EXERCISE: Evaluate your document lengths 126 | Chapter 04 How to Select an Embedding Model Variable Dimension Embeddings Variable Dimension EmbeddingsFigure 4.7 Figure 4.7 illustrates how Matryoshka Representation Learning trains a single embedding model that opera

## q036: How does late chunking differ from the usual chunk-then-embed order?
Category: single_topic

- `#228` grade **3**: ontext-enriched chunking trade-offsTable 3.7 90 | Chapter 03 Advanced Chunking Techniques Late Chunking Late chunking reverses the traditional chunking order. Instead of chunking text first and then embedding each chunk, late chunking embeds the entire document and performs chunking after embedding,
- `#229` grade **3**: tion RAG Chapter 03 Advanced Chunking Techniques Introduced by Jina AI in 2024, this approach addresses the semantic gap problem in traditional chunking. When you chunk first, each chunk loses context from the surrounding text. Late chunking maintains the full document context during embedding, then
- `#230` grade **2**: rough the embedding model's transformer layers. Only at the final pooling step does the system apply chunk boundaries to extract separate embeddings for each chunk. Each chunk embedding contains information about its position and role within the larger document. Documents where chunks need awareness

## q037: What does context-enriched chunking do and how well does it work?
Category: multi_relevant

- `#224` grade **3**:  Which product line? Which document? This strategy prepends contextual information to each chunk before embedding. The original chunk remains intact, while the embedding captures both its content and its document-level context. Anthropic's research on contextual retrieval showed that this approach r
- `#225` grade **3**: neration RAG Chapter 03 Advanced Chunking Techniques The process works in two steps: → Fi rst, use an LLM to generate context for each chunk. The LLM receives the entire document and the specific chunk, then produces a brief contextual statement. → Se cond, prepend this context to the chunk before e
- `#226` grade **2**: ort, Financial Performance section: Revenue increased 15% quarter-over-quarter." The embedding now captures both the fact and its source context. Retrieval matches queries about Q3 2024 revenue to this chunk, even though the original chunk doesn't mention Q3 2024. This technique is particularly well
- `#223` grade **2**: nships, making retrieval actually work for structured content. 88 | Chapter 03 Advanced Chunking Techniques Let's go through a code snippet to look at how it works: Context-enriched chunking addresses a fundamental problem: isolated chunks lose their surrounding context. A chunk reading "Revenue inc

## q038: Which chunking strategy suits legal documents and FAQs?
Category: multi_relevant

- `#210` grade **3**: rity takes precedence over uniform chunk sizes. FAQ content, where each question-answer pair should stay intact, can benefit from sentence splitting. Legal documents requiring clause completeness also work well with this approach. See Table 3.4 to understand the trade-offs. Pros Cons Preserves compl
- `#209` grade **2**: hod analyzes text to find sentence endings based on punctuation and grammar rules. You control how many sentences per chunk and how many sentences overlap between chunks. A chunk might contain 5 sentences with a 1-sentence overlap, ensuring no sentence ever gets cut mid-way. 81 | Chapter 03 Advanced
- `#176` grade **2**: allucinations but increase storage costs and query latency. 66 | Chapter 03 Advanced Chunking Techniques Selecting the right chunking strategy requires understanding the variables that shape retrieval quality. These parameters determine which approach works best for your specific use case. Factors I
- `#245` grade **1**: se when Key consideration Character Splitter Prototyping quickly or processing unstructured text Ignores meaning; splits can break mid- sentence Recursive Character Splitter You need size control with some semantic preservation Better than character splitting but still breaks on dense paragraphs Sen

## q039: What are the drawbacks of semantic splitting?
Category: multi_relevant

- `#215` grade **3**: strategy. Table 3.5 shows that when you can afford the embedding overhead during indexing, semantic chunking delivers superior retrieval quality. Pros Cons Respects semantic boundaries and topic shifts Computationally expensive (embeds every sentence) Chunks contain related information Variable and 
- `#216` grade **3**: or very large chunks Semantic splitting trade-offsTable 3.5 If you're processing millions of documents, embedding every sentence for semantic chunking becomes expensive. Calculate the cost first. For a 10,000-document corpus averaging 100 sentences per document, that's 1 million embedding calls just

## q040: How do I pick a chunk size based on the type of question users ask?
Category: multi_relevant

- `#185` grade **3**:  reduce the number of retrievals needed. Embedding Model Constraints LLM Context Window 69 | Chapter 03 Advanced Chunking Techniques User query types shape optimal chunk size. Different questions need different information granularity. Factual queries like "What was our Q4 revenue?" need precise, fo
- `#186` grade **3**: urns the exact number without surrounding paragraphs about company strategy or market conditions. Analytical queries like "How did market conditions affect our Q4 performance?" need more context. Larger chunks preserve the relationships between revenue figures, market analysis, and strategic respons
- `#187` grade **2**: efit from smaller chunks. When you ask "Compare our customer retention rates to industry benchmarks over the past three years," the retriever must find retention data, industry data, and temporal comparisons. Smaller chunks increase the likelihood of retrieving all necessary pieces. Query Patterns C

## q041: How do rerankers reduce hallucinations?
Category: multi_relevant

- `#462` grade **3**: ribution ~0.65 ~0.82 Better grounding Hallucination Rate Higher Lower Quality improvement Reranker impact on system performance Table 6.1 Why does a better ranking reduce hallucinations? When you give an LLM 20 chunks where only three are relevant, the model sees 17 chunks of noise that might contai
- `#460` grade **3**:  grounds responses in provided information rather than generating plausible-sounding but incorrect details. Chunk attribution metrics measure this effect. Systems with reranking show approximately 10% higher attribution rates than systems without reranking, indicating the model receives more useful 
- `#459` grade **2**: re. LLM-based rerankers take significantly longer, with GPT-4 and Llama-70B requiring up to 1 minute to rerank 50 documents. This latency matters for real-time applications. A customer support chatbot with 500ms response time budgets can absorb cross-encoder latency. A code completion system requiri

## q042: Why retrieve many candidates first and then rerank only for ordering?
Category: different_terminology, multi_relevant

- `#451` grade **3**: y search across millions of documents, retrieving the top 50-100 candidates in under 100ms. This initial retrieval optimizes for speed and recall, casting a wide net to ensure that relevant documents appear somewhere in the results. 178 | Chapter 06 How to Select a Reranker The problem lies in the r
- `#453` grade **2**:  the most common reranker architecture, concatenate the query and document together, then analyze their interaction through transformer layers. This joint processing captures relevance signals that vector similarity misses. The reranker produces relevance scores for each candidate, then returns the 
- `#450` grade **2**: hing the most useful chunks to the top before generation. 06 Chapter 177 | Chapter 06 How to Select a Reranker Rerankers take the initial retrieval results and produce a refined ranking based on precise relevance scoring. The process happens in two stages. See figure 6.1. What are Rerankers? Two-sta

## q043: How can I stop one customer's data from showing up in another customer's answers?
Category: different_terminology, multi_relevant

- `#146` grade **3**: nancy Multi-tenant systems serve multiple organizations within a single deployment. Each organization has its own data, and the system must prevent data from one organization from appearing in responses to another. Metadata-Based Isolation One approach uses metadata to separate data by tenant. Each 
- `#147` grade **3**: e requesting organization. This method is efficient but depends on the correct implementation of filtering. A mistake in filter logic can expose data to the wrong tenant, so it requires careful testing. Separate Indexes Another approach is to maintain separate vector indexes for each tenant. A query
- `#148` grade **2**: t increases storage and operational overhead because each tenant needs its own index. You select an approach based on scale, data sensitivity, and operational constraints. Metadata filtering is efficient for large numbers of tenants, while separate indexes reduce risk in environments that require st

## q044: How does the system honor a user's request to delete their personal data?
Category: different_terminology, multi_relevant

- `#150` grade **3**:  remove unnecessary fields from embeddings and store sensitive information in a separate location when possible. Documents that contain personal data can reference identifiers rather than embedding full names or addresses directly. Right to Erasure GDPR requires that users can request the deletion o
- `#151` grade **2**: ved, the system removes those items from all storage layers. This includes raw documents, derived chunks, embeddings, and indexes. Automated workflows help ensure that all references are removed. Audit Trails Audit trails record access patterns and system activity. Each record includes the user iden
- `#149` grade **1**:  Compliance RAG systems that process personal or regulated data must meet the requirements of the applicable laws and standards. These include GDPR in Europe, CCPA in California, HIPAA for healthcare, and SOC 2 for security, all impose obligations. The architecture must support data protection, dele

## q045: How do I fine-tune a model with LoRA adapters?
Category: no_answer

- No relevant chunks (topic absent from the book; verified by keyword search).

## q046: How do I transcribe audio recordings before indexing them?
Category: no_answer

- No relevant chunks (topic absent from the book; verified by keyword search).

## q047: How do I tune the k1 and b parameters of the BM25 scoring formula?
Category: no_answer

- No relevant chunks (topic absent from the book; verified by keyword search).

## q048: How do I apply stemming and lemmatization before building a TF-IDF index?
Category: no_answer

- No relevant chunks (topic absent from the book; verified by keyword search).
