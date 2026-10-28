# Requirements Document

## Introduction

The RAG Retrieval and Reranking Evaluation Lab is a local-first, educational Python project whose primary objective is to understand and **measure** whether reranking improves a Retrieval-Augmented Generation (RAG) pipeline. The system is built directly in Python — without LangChain or LlamaIndex — so that the internal behaviour of each component (chunking, embedding, retrieval, reranking, generation, evaluation) is fully exposed and independently verifiable.

The lab runs entirely on the user's local machine: embeddings and answer generation use models served by Ollama, which runs locally either as a native installation or as a Docker container, with components connecting to it over a configurable local endpoint (base URL). Dense similarity is computed from scratch with NumPy, keyword retrieval uses BM25, and the primary reranker is a dedicated cross-encoder executed locally through Sentence Transformers / PyTorch. No paid APIs are used.

The project is organised into five sequential phases, each of which must produce a runnable, verifiable deliverable before the next phase begins:

- **Phase 1** — Build the retrieval engine from scratch (chunk, embed, index, retrieve).
- **Phase 2** — Implement a real cross-encoder reranker and observe how it reorders candidates.
- **Phase 3** — Evaluate retrieval and reranking scientifically with labelled data and ranking metrics.
- **Phase 4** — Compare open-source retrieval strategies (dense, BM25, hybrid) and two reranking approaches (cross-encoder vs LLM-based).
- **Phase 5** — Assemble a complete local RAG application and evaluate whether reranking improves the final generated answers.

The codebase must be modular so that any single component can be swapped without rewriting the pipeline.

## Glossary

- **RAG (Retrieval-Augmented Generation)**: A technique where documents relevant to a query are retrieved and supplied as context to a language model that generates an answer.
- **Ollama**: A local model runtime that serves embedding and chat/instruct models through a local HTTP API. Ollama may run either as a native local installation or as a Docker container; in both cases it is reachable at a configurable local endpoint (default `http://localhost:11434`).
- **Ollama Endpoint**: The configurable base URL (default `http://localhost:11434`) at which the locally-running Ollama instance — whether native or containerised — exposes its HTTP API; all components that need an Ollama model connect to this endpoint.
- **embeddinggemma**: The embedding model served by Ollama and used to produce vector embeddings for documents and queries.
- **Source Document**: An original input file (`.pdf`, `.md`, `.txt`, `.json`, or `.jsonl`) placed in the `data` directory from which document text is extracted before chunking.
- **Source Loader / Text Extraction**: The step that reads a source document file, detects its type by extension, and extracts plain text for chunking. PDF text is extracted with a local Python PDF library (e.g. pypdf); Markdown is read as UTF-8 text with its markup preserved or stripped to plain text.
- **Document**: A single stored unit of knowledge-base text, identified by a unique document ID, originating either from a structured JSON/JSONL record or from text extracted from a `.pdf`, `.md`, or `.txt` source file.
- **Chunk**: A fixed-size, optionally overlapping, segment of the plain text extracted from a source document used as the unit of retrieval.
- **Chunk Size**: The configurable length of a chunk, expressed in a defined unit (characters or tokens).
- **Chunk Overlap**: The configurable number of units shared between two consecutive chunks of the same source document.
- **Embedding**: A numeric vector representation of text produced by an embedding model.
- **Bi-encoder**: An encoder that embeds the query and each document independently, enabling precomputed document vectors and fast similarity search (used for dense retrieval).
- **Cross-encoder**: A model that receives a query and a document together as a single input and outputs a relevance score; it cannot precompute document representations and is used for reranking. The primary model is `cross-encoder/ms-marco-MiniLM-L6-v2`.
- **Cosine Similarity**: A similarity measure between two vectors equal to their dot product divided by the product of their magnitudes, implemented from scratch in NumPy.
- **Dense Retrieval**: Retrieval that ranks documents by embedding (cosine) similarity to the query.
- **BM25**: A keyword-based lexical ranking function, provided by the `rank-bm25` library.
- **Hybrid Retrieval**: Retrieval that combines dense and BM25 ranked lists.
- **RRF (Reciprocal Rank Fusion)**: A method of combining multiple ranked lists using the reciprocal of each item's rank, used to merge dense and BM25 results without adding incomparable raw scores.
- **Retriever**: The component that returns an initial ranked candidate set of documents for a query.
- **Candidate Set**: The set of documents returned by the retriever and passed to the reranker; the reranker can only reorder documents within this set.
- **Reranker**: The component that re-scores and reorders the candidate set to improve relevance ordering.
- **Cross-encoder Reranker**: A reranker using the dedicated cross-encoder model.
- **LLM Reranker**: A reranker that prompts an Ollama-hosted instruct model (such as `qwen2.5:3b`) to produce a structured relevance judgment per document.
- **Top-k**: The number of highest-ranked documents returned by a retrieval or reranking step.
- **Generator**: The component that calls an Ollama chat model to produce a final answer from the query and selected context.
- **Ground Truth / Relevance Label**: A manually assigned judgment of whether a document is relevant to a query; never derived from the reranker's own scores.
- **Graded Relevance**: A relevance label with more than two levels (for example 0, 1, 2) used by NDCG.
- **Recall@K**: The fraction of relevant documents that appear within the top K ranked results.
- **Precision@K**: The fraction of the top K ranked results that are relevant.
- **MRR (Mean Reciprocal Rank)**: The mean, over queries, of the reciprocal of the rank of the first relevant document, computed over a defined cutoff (MRR@10).
- **NDCG (Normalized Discounted Cumulative Gain)**: A ranking metric that rewards placing more-relevant documents higher, normalised against the ideal ordering (NDCG@5).
- **Latency**: The wall-clock time taken by a measured operation.
- **Baseline**: An experiment configuration that uses initial retrieval results without reranking.
- **Reranked**: An experiment configuration that applies a reranker to the candidate set.
- **Faithfulness**: The degree to which a generated answer is supported by the retrieved context.
- **Context Relevance**: The degree to which the retrieved context is relevant to the query.
- **Citation**: A reference in a generated answer to the source document ID(s) that support a claim.
- **Ragas**: An optional evaluation framework that may be introduced in later phases for answer evaluation.
- **FAISS**: An optional library for efficient vector search that may replace the NumPy similarity search later.

## Requirements

### Requirement 1: Project Structure and Modularity

**User Story:** As a RAG engineer, I want a modular project with components separated from phase experiment scripts, so that I can swap any single component without rewriting the pipeline.

#### Acceptance Criteria

1. THE System SHALL organise all reusable component modules (retriever, reranker, embedder, generator, and evaluation) under a top-level `src` directory and all runnable phase scripts under a top-level `experiments` directory, with no reusable component logic defined inside the `experiments` directory.
2. THE System SHALL store source documents and evaluation data under a top-level `data` directory using files with a `.pdf`, `.md`, `.txt`, `.json`, or `.jsonl` extension.
3. IF a file under the `data` directory cannot be parsed as valid JSON or JSONL, or a `.pdf`, `.md`, or `.txt` source file cannot have its text extracted, THEN THE System SHALL reject the file and produce an error indicating the offending file path and the parse or extraction failure cause, without loading partial data.
4. THE System SHALL implement the retrieval, reranking, generation, and evaluation components using only Python standard-library modules and the project's declared local dependencies, SHALL permit a local Python PDF-extraction library (such as `pypdf`) as a declared local dependency for text extraction, and SHALL NOT import or depend on the LangChain or LlamaIndex libraries.
5. WHERE any single component (retriever, reranker, embedder, or generator) is replaced with an alternative implementation that preserves the component's defined input and output interface, THE System SHALL execute the affected phase experiment scripts to completion with zero source-line modifications to those scripts.
6. THE System SHALL execute every component (retriever, reranker, embedder, generator, and evaluation) on the user's local machine using locally executed models and the locally-running Ollama instance reachable at the configured Ollama Endpoint, regardless of whether that instance is a native installation or a Docker container, and SHALL NOT issue any network request to an external or paid inference API.
7. IF a required local model or the locally-running Ollama instance at the configured Ollama Endpoint is unavailable when a component is invoked, whether that instance is a native installation or a Docker container, THEN THE System SHALL halt that component and produce an error indicating the missing model or unavailable Ollama Endpoint, without falling back to any external API.

### Requirement 2: Document Loading and Knowledge Base (Phase 1)

**User Story:** As a RAG engineer, I want to load a labelled knowledge base of document chunks, so that I have controlled material to retrieve against and known relevance for testing.

#### Acceptance Criteria

1. WHEN the Document_Loader is given a source document file, THE Document_Loader SHALL detect the file type by extension and support the `.pdf`, `.md`, `.txt`, `.json`, and `.jsonl` extensions.
2. WHEN the Document_Loader is given a `.pdf` file, THE Document_Loader SHALL extract the document text using a local Python PDF library and SHALL NOT call an external or paid API.
3. WHEN the Document_Loader is given a `.md` or `.txt` file, THE Document_Loader SHALL read the file as UTF-8 text.
4. WHEN the Document_Loader is given a `.json` or `.jsonl` file, THE Document_Loader SHALL load each structured document record as a record containing a unique document ID and its non-empty text content.
5. WHEN the Document_Loader produces documents or chunks from a `.pdf`, `.md`, or `.txt` source file, THE Document_Loader SHALL assign each produced document or chunk a unique document ID derived from the source file, such as the filename combined with a sequence number, because those files do not carry explicit document IDs.
6. IF a source file has an unsupported extension, THEN THE Document_Loader SHALL skip or reject the file and return an error indicating the unsupported file type.
7. IF a `.pdf` file cannot be opened or yields no extractable text, THEN THE Document_Loader SHALL return an error indicating the extraction failure for that file.
8. IF two or more documents resolve to the same document ID, THEN THE Document_Loader SHALL reject the load and return an error indicating a duplicate document ID.
9. WHEN document loading completes successfully, THE Knowledge_Base SHALL contain between 100 and 500 document chunks, each with a text length between 1 and 2000 characters, collectively covering all five of the following topics: RAG, embeddings, vector databases, Transformer architectures, and Python.
10. THE Knowledge_Base SHALL include at least one document in each of the following five categories: genuinely relevant to a test query, semantically similar but irrelevant, keyword-overlapping but answering a different question, containing the answer in different wording, and containing only part of the answer.
11. THE Knowledge_Base SHALL record, as manually labelled test data, for each test query the set of document IDs that are genuinely relevant, where each referenced document ID exists in the loaded Knowledge_Base.
12. IF a source file is missing, empty, or contains zero valid document records, THEN THE Document_Loader SHALL return an error indicating the specific failure cause and SHALL leave the Knowledge_Base unpopulated.

### Requirement 3: Text Chunking (Phase 1)

**User Story:** As a RAG engineer, I want configurable chunking, so that I can experiment with how chunk size and overlap affect retrieval.

#### Acceptance Criteria

1. WHEN the Chunker receives document text, THE Chunker SHALL operate on the plain text produced by the Source Loader / Text Extraction step regardless of whether the original source file was `.pdf`, `.md`, `.txt`, `.json`, or `.jsonl`.
2. WHEN the Chunker receives document text and a configured chunk size (integer from 1 to 8,192 units) and chunk overlap (integer from 0 to chunk size minus 1 units), THE Chunker SHALL split the text into sequential chunks where each chunk contains at most the configured chunk size in units.
3. WHEN a chunk overlap greater than 0 is configured, THE Chunker SHALL share exactly the configured number of trailing units of each chunk as the leading units of the next chunk of the same source document.
4. WHEN the Chunker produces chunks from a document, THE Chunker SHALL assign each chunk an identifier that is unique across all chunks produced in the run.
5. IF the configured chunk overlap is greater than or equal to the configured chunk size, THEN THE Chunker SHALL reject the configuration, return an error message indicating that overlap must be less than chunk size, and produce no chunks.
6. IF the configured chunk size is less than 1 or the configured chunk overlap is less than 0, THEN THE Chunker SHALL reject the configuration, return an error message indicating the invalid parameter and its allowed range, and produce no chunks.
7. WHEN the Chunker receives empty document text, THE Chunker SHALL produce zero chunks and return successfully.

### Requirement 4: Embedding Generation via Ollama (Phase 1)

**User Story:** As a RAG engineer, I want to generate embeddings for queries and documents through Ollama, so that I can compute semantic similarity.

#### Acceptance Criteria

1. WHEN the Ollama_Embedder is given a non-empty text input of up to 2048 characters, THE Ollama_Embedder SHALL request an embedding from the Ollama embedding API at the configured Ollama Endpoint using the `embeddinggemma` model and return a vector of floating-point numbers.
2. THE Ollama_Embedder SHALL produce embeddings of identical, fixed dimensionality for every input text within a single session, and SHALL expose that dimensionality as a queryable value.
3. IF the input text is empty or exceeds 2048 characters, THEN THE Ollama_Embedder SHALL reject the request, return an error indicating the invalid input, and SHALL NOT call the Ollama embedding API.
4. IF the Ollama_Embedder cannot reach the configured Ollama Endpoint within a 30-second timeout, THEN THE Ollama_Embedder SHALL return an error indicating the connection failure without returning a partial or empty vector.
5. IF the Ollama embedding API returns a response that is not a valid numeric vector of the expected dimensionality, THEN THE Ollama_Embedder SHALL return an error indicating the invalid response and SHALL NOT return the malformed vector.

### Requirement 5: Dense Retrieval with Cosine Similarity from Scratch (Phase 1)

**User Story:** As a RAG engineer, I want cosine similarity and top-k retrieval implemented from scratch in NumPy, so that I can reproduce and explain every ranking independently.

#### Acceptance Criteria

1. WHEN the Dense_Retriever receives a query embedding, THE Dense_Retriever SHALL compute cosine similarity, bounded between -1.0 and 1.0, between the query embedding and each document embedding using a from-scratch NumPy implementation that does not call an external cosine-similarity or ranking library.
2. WHEN the Dense_Retriever is given a query and a top-k value k where 1 is less than or equal to k and k is less than or equal to the number of available documents N, THE Dense_Retriever SHALL return the top-k documents ranked by descending cosine similarity, breaking ties by ascending document index.
3. THE Dense_Retriever SHALL return no more than the requested number of documents and SHALL NOT return any document more than once (correctness property: retriever returns no more than the requested number of documents).
4. IF the requested top-k value exceeds the number of available documents, THEN THE Dense_Retriever SHALL return all available documents ranked by descending cosine similarity, breaking ties by ascending document index.
5. IF the requested top-k value is less than 1 or the document collection is empty, THEN THE Dense_Retriever SHALL return an empty result set together with an indication of the condition and SHALL NOT raise an unhandled error.
6. IF a query or document embedding has zero magnitude or a dimensionality that does not match the indexed embeddings, THEN THE Dense_Retriever SHALL return an error indicating the invalid embedding and SHALL NOT perform a division by zero.
7. WHEN the Dense_Retriever completes a retrieval request, THE Dense_Retriever SHALL report the retrieval time as a numeric value in milliseconds alongside the returned results.

### Requirement 6: Phase 1 Retrieval Deliverable and Experiments

**User Story:** As a RAG engineer, I want a Phase 1 script that reports retrieval details, so that I can explain why a document appears at a given rank.

#### Acceptance Criteria

1. WHEN `phase1_retrieval.py` is run with a query, THE Phase1_Script SHALL print the original query text, the retrieved document IDs, each document's similarity score formatted to at least 4 decimal places, the ranked document contents in descending similarity order, and the retrieval time in milliseconds.
2. IF `phase1_retrieval.py` is run with a query that is empty or exceeds 2048 characters, THEN THE Phase1_Script SHALL reject the query and print an error message indicating the query is empty or too long, without performing retrieval.
3. THE Phase1_Script SHALL expose configurable query text, chunk size (integer from 50 to 2048 tokens), and chunk overlap (integer from 0 to chunk size minus 1) for experimentation.
4. IF a configured chunk size or chunk overlap falls outside its valid range, or chunk overlap is greater than or equal to chunk size, THEN THE Phase1_Script SHALL reject the configuration and print an error message indicating the invalid parameter and its allowed range, without performing retrieval.
5. WHEN the same query is submitted, THE Phase1_Script SHALL print the keyword-match ranking and the semantic-match ranking side by side, each showing document IDs and scores, so the two rankings can be compared.
6. WHEN identical inputs (query, document set, chunk size, and chunk overlap) are provided across separate runs, THE Dense_Retriever SHALL produce identical rankings and cosine similarity scores that match an independently recalculated cosine similarity to within a tolerance of 1e-6.

### Requirement 7: Cross-Encoder Reranking (Phase 2)

**User Story:** As a RAG engineer, I want a real cross-encoder reranker, so that I can observe how reranking changes the ordering of retrieved documents.

#### Acceptance Criteria

1. THE Cross_Encoder_Reranker SHALL load the `cross-encoder/ms-marco-MiniLM-L6-v2` model and execute it locally through Sentence Transformers without using Ollama and without calling a paid external API.
2. IF the `cross-encoder/ms-marco-MiniLM-L6-v2` model cannot be loaded, THEN THE Cross_Encoder_Reranker SHALL return an error indicating the model could not be loaded and SHALL NOT return any reranked documents.
3. WHEN the Cross_Encoder_Reranker receives a query and a candidate set of 1 to 1000 candidates, THE Cross_Encoder_Reranker SHALL construct exactly one query-document pair per candidate and compute exactly one numeric relevance score for every candidate.
4. THE Cross_Encoder_Reranker SHALL return documents sorted by descending relevance score, and WHEN two or more candidates have equal relevance scores, THE Cross_Encoder_Reranker SHALL preserve their relative order from the received candidate set (correctness property: reranker returns documents sorted by descending score with stable ordering for ties).
5. THE Cross_Encoder_Reranker SHALL return only documents that were present in the candidate set it received, with no document returned more than once and no document count exceeding the received candidate count (correctness property: every reranked document came from the original candidate set exactly once).
6. WHEN a top-k value k is supplied where k is an integer of 1 or greater, THE Cross_Encoder_Reranker SHALL return the first min(k, candidate count) documents after sorting by descending relevance score.
7. IF a supplied top-k value is less than 1 or is not an integer, THEN THE Cross_Encoder_Reranker SHALL return an error indicating the top-k value is invalid and SHALL NOT return any reranked documents.
8. IF the candidate set is empty, THEN THE Cross_Encoder_Reranker SHALL return an empty result without raising an error (correctness property: an empty candidate list is handled correctly).
9. THE Cross_Encoder_Reranker SHALL treat each relevance score as a raw comparison value for ordering documents and SHALL NOT present a relevance score as a calibrated probability or percentage.

### Requirement 8: Phase 2 Reranking Deliverable and Experiments

**User Story:** As a RAG engineer, I want a Phase 2 script and report, so that I can explain why a document moved position and verify whether the change improved relevant-document ranking.

#### Acceptance Criteria

1. WHEN `phase2_reranking.py` is run, THE Phase2_Script SHALL take the documents retrieved in Phase 1 as the candidate set and SHALL apply the Cross_Encoder_Reranker to that candidate set.
2. WHEN the Cross_Encoder_Reranker has scored the candidate set, THE Phase2_Script SHALL return the top 5 reranked documents and SHALL record, for every document in the candidate set, both its position before reranking and its position after reranking.
3. THE Phase2_Script SHALL produce a persisted CSV report containing the fields `query_id`, `document_id`, `initial_rank`, `initial_similarity_score`, `rerank_score`, `reranked_position`, and `is_relevant`, where `is_relevant` is taken from the manually assigned relevance labels and is never derived from the reranker's scores.
4. WHEN the CSV report has been written, THE Phase2_Script SHALL report the location of the written report file.
5. THE Phase2_Script SHALL support the following experiments and SHALL, for each one, produce a report using the field set defined in criterion 3: (a) initial retrieval only without reranking, (b) retrieve 20 and rerank 20, (c) retrieve 50 and rerank 50, (d) retrieve 100 and rerank 100, (e) reranking of the candidate set for a reworded version of the same query, and (f) reranking of a candidate set into which at least one irrelevant document has been inserted.
6. THE Phase2_Script SHALL record, for every candidate document, the pairing of `initial_rank` with `reranked_position` and the pairing of `initial_similarity_score` with `rerank_score`, so that any single document's rank movement is explained by its two scores and so that the change in position of documents whose `is_relevant` is true can be determined by comparing `initial_rank` against `reranked_position`.
7. WHEN a relevant document is absent from the candidate set, THE Phase2_Script SHALL record that document as absent from both the candidate set and the reranked results (correctness property: a relevant document outside the retrieved candidate set cannot appear in the reranked results).
8. IF the Phase 1 candidate set is empty or cannot be obtained, THEN THE Phase2_Script SHALL return a descriptive error indicating the candidate set is unavailable and SHALL NOT produce a reranked report for that query.

### Requirement 9: Labelled Evaluation Dataset (Phase 3)

**User Story:** As a RAG engineer, I want a manually labelled evaluation dataset, so that I can measure retrieval quality against ground truth rather than anecdotes.

#### Acceptance Criteria

1. THE Evaluation_Dataset SHALL contain at least 30 and at most 50 questions, each stored as one record in `evaluation_queries.jsonl` with a unique non-empty `query_id`, a non-empty `query` string, and a `relevant_documents` list of corpus document IDs.
2. THE Evaluation_Dataset SHALL derive relevance labels from manual judgment and SHALL NOT derive relevance labels from any reranker's scores.
3. THE Evaluation_Dataset SHALL include at least 3 questions whose answers use terminology different from the query, at least 3 questions with 2 or more relevant documents, at least 3 questions having several documents on the same topic that answer different questions, and at least 3 questions for which no corpus document contains the answer.
4. WHERE a question has no corpus document containing the answer, THE Evaluation_Dataset SHALL record that question's `relevant_documents` as an empty list.
5. WHERE a metric uses graded relevance, THE Evaluation_Dataset SHALL provide a graded relevance label on an integer scale of 0 to 3, where 0 denotes non-relevant and 3 denotes fully relevant, for each relevant document of the affected queries.
6. IF a record has a duplicate `query_id`, a missing or empty `query`, or a `relevant_documents` entry that does not match any corpus document ID, THEN THE Evaluation_Dataset SHALL be treated as invalid and the loading process SHALL reject it with an error indicating the offending record and field.

### Requirement 10: Evaluation Metrics (Phase 3)

**User Story:** As a RAG engineer, I want standard ranking metrics implemented, so that I can prove whether reranking improves retrieval quality.

#### Acceptance Criteria

1. WHEN the Evaluation receives a set of ranked results and the relevance labels, THE Evaluation SHALL compute and return Recall@5, Recall@50, Precision@5, MRR@10, NDCG@5, and latency.
2. THE Evaluation SHALL compute each metric to match an independently hand-calculated example for the same inputs to within an absolute tolerance of 0.0001 (correctness property: evaluation metrics match manually calculated examples).
3. THE Evaluation SHALL measure and report retrieval latency and reranking latency separately, each as a numeric value in milliseconds (correctness property: retrieval and reranking latency are measured separately).
4. WHERE NDCG@5 is computed and graded relevance labels are available, THE Evaluation SHALL use the graded relevance labels; otherwise THE Evaluation SHALL treat relevance as binary.
5. IF a query has no relevant documents in its labels, THEN THE Evaluation SHALL define Recall and MRR for that query as 0 and SHALL exclude it from averages in a documented, consistent manner.
6. IF a ranked result list is empty or the cutoff K exceeds the number of available results, THEN THE Evaluation SHALL compute each metric over the available results without raising an error.

### Requirement 11: Phase 3 Baseline vs Reranked Comparison and Deliverables

**User Story:** As a RAG engineer, I want a controlled baseline-vs-reranked comparison, so that any measured difference is attributable to reranking alone.

#### Acceptance Criteria

1. WHEN `phase3_evaluation.py` is run, THE Phase3_Script SHALL evaluate a Baseline configuration and a Reranked configuration using the same queries, the same documents, the same relevance labels, and the same initial retrieval results.
2. IF the Baseline and Reranked configurations do not share an identical initial candidate set for a query, THEN THE Phase3_Script SHALL return an error indicating that the candidate sets differ and SHALL NOT report a comparison for that query.
3. THE Phase3_Script SHALL produce `baseline_results.csv`, `reranked_results.csv`, and `evaluation_summary.csv`, where `evaluation_summary.csv` reports NDCG@5, Recall@5, and Recall@50 for both configurations.
4. THE Phase3_Script SHALL record in `evaluation_summary.csv` whether reranking changed each metric, and SHALL document that an improved NDCG@5 with an unchanged Recall@50 indicates reordering of the same candidate set rather than a change to the candidate set contents.
5. THE Phase3_Script SHALL document that an improved Recall@5 indicates relevant documents moved into the top 5 rather than the reranker discovering documents the retriever did not include.
6. WHEN `phase3_evaluation.py` is run more than once on identical inputs, THE Phase3_Script SHALL produce identical metric values across runs (correctness property: results are reproducible enough to compare experiments meaningfully).

### Requirement 12: Retrieval Strategy Comparison (Phase 4)

**User Story:** As a RAG engineer, I want to compare dense, keyword, and hybrid retrieval strategies, so that I understand production retrieval trade-offs.

#### Acceptance Criteria

1. THE Phase4_Script SHALL compare exactly five retrieval configurations: dense retrieval without a reranker, BM25 retrieval without a reranker, dense retrieval with the cross-encoder reranker, hybrid retrieval without a reranker, and hybrid retrieval with the cross-encoder reranker.
2. THE BM25_Retriever SHALL rank documents using the `rank-bm25` library and SHALL return a ranked list of between 1 and the retrieval top-k documents, where top-k is a configured value between 1 and 1000.
3. THE Hybrid_Retriever SHALL combine the dense and BM25 ranked lists using Reciprocal Rank Fusion with a rank constant k of 60, and SHALL NOT combine them by adding raw BM25 and cosine scores directly.
4. WHEN comparing configurations, THE Phase4_Script SHALL use the identical labelled question set and the identical candidate document set across all configurations that share a retrieval stage.
5. WHEN all five configurations have been executed, THE Phase4_Script SHALL record, for each configuration, the same set of retrieval evaluation metrics (Recall@5, Recall@50, Precision@5, MRR@10, and NDCG@5) computed over the identical labelled question set.
6. IF any configuration fails to execute or produces no ranked results for a question, THEN THE Phase4_Script SHALL record an error indication identifying the failed configuration and the affected question, exclude that configuration from the comparison output, and continue executing the remaining configurations.

### Requirement 13: Reranking Approach Comparison (Phase 4)

**User Story:** As a RAG engineer, I want to compare the cross-encoder reranker against an Ollama LLM reranker, so that I can judge their trade-offs rather than assuming one is better.

#### Acceptance Criteria

1. WHEN the LLM_Reranker receives a query and a candidate set, THE LLM_Reranker SHALL connect to the configured Ollama Endpoint and prompt an Ollama-hosted instruct model to evaluate each candidate document against the query using a fixed integer relevance scale of 0 to 3 and the same prompt and scale for every candidate.
2. WHEN the LLM_Reranker receives a model response, THE LLM_Reranker SHALL parse the structured relevance judgment programmatically into a numeric score on the 0-to-3 scale.
3. IF a model response cannot be parsed into a valid score on the 0-to-3 scale, THEN THE LLM_Reranker SHALL assign the lowest score, record a parse failure for that document, and continue reranking the remaining candidates without aborting.
4. THE LLM_Reranker SHALL return only documents present in the candidate set it received, sorted by descending score (correctness property applied to the LLM reranker, mirroring the cross-encoder reranker).
5. THE Phase4_Script SHALL run the cross-encoder reranker and the LLM reranker against the same candidate set and the same labelled questions.
6. THE Phase4_Script SHALL measure, for each reranking approach, NDCG@5, MRR@10, Recall@50 of the candidate set before reranking, reranking latency in milliseconds, peak memory usage, the number of model evaluations, and ranking consistency defined as the proportion of identical top-5 orderings across at least 3 repeated runs on the same inputs.
7. THE Phase4_Script SHALL treat the LLM reranker as an experimental approach and SHALL NOT assume the LLM reranker outperforms the dedicated cross-encoder.

### Requirement 14: Local Answer Generation (Phase 5)

**User Story:** As a RAG engineer, I want grounded answer generation through Ollama, so that the system answers from retrieved context and cites its sources.

#### Acceptance Criteria

1. WHEN the Generator receives a query and a non-empty list of selected document chunks, THE Generator SHALL construct a prompt containing the retrieved context and the query, call the Ollama chat API at the configured Ollama Endpoint, and return a generated answer.
2. WHEN the Generator returns an answer, THE Generator SHALL include a reference to the supporting document ID for each chunk used, such that each citation maps to a document ID present in the supplied chunks.
3. THE Generator SHALL instruct the model to answer using only the supplied context and to cite the supporting document IDs.
4. IF the supplied context does not contain information that answers the query, THEN THE Generator SHALL return a response stating that the answer cannot be determined from the context, without generating unsupported content.
5. IF the Generator receives an empty list of document chunks, THEN THE Generator SHALL return a response indicating that no context is available to answer the query and SHALL NOT call the Ollama chat API.
6. IF the Ollama chat API call to the configured Ollama Endpoint fails or does not return a response within 60 seconds, THEN THE Generator SHALL return an error indication reporting that answer generation failed, and SHALL NOT return a fabricated answer.

### Requirement 15: Phase 5 End-to-End Answer Evaluation and Deliverable

**User Story:** As a RAG engineer, I want to evaluate final answers under baseline and reranked retrieval, so that I can establish whether improved context actually helps the LLM.

#### Acceptance Criteria

1. WHEN `phase5_end_to_end.py` is run, THE Phase5_Script SHALL connect document loading, chunking, embedding, retrieval, reranking, and generation into a pipeline that runs entirely on the user's local machine.
2. WHEN a test question is evaluated, THE Phase5_Script SHALL run it once using baseline retrieval results and once using reranked retrieval results, and SHALL report, side by side for both runs, the answer correctness, faithfulness, context relevance, citation accuracy, and latency in seconds.
3. THE Phase5_Script SHALL support manual assessment of answer correctness and citation accuracy.
4. WHERE Ragas is enabled, THE Phase5_Script SHALL use Ragas to compute answer evaluation metrics in addition to the manual assessment.
5. THE Phase5_Script SHALL report both the retrieval metrics and the answer-quality metrics for each configuration so that whether improved retrieval metrics coincide with improved answers can be observed directly.
6. IF any pipeline stage fails for a test question, THEN THE Phase5_Script SHALL record an error indication naming the failed stage and the affected question and SHALL continue evaluating the remaining questions.
7. WHEN all test questions have been evaluated, THE Phase5_Script SHALL identify which retrieval and reranking configuration performs best on the labelled evaluation dataset.

### Requirement 16: Optional Efficient Vector Search

**User Story:** As a RAG engineer, I want the option to replace NumPy similarity search with FAISS later, so that I can scale retrieval without changing the pipeline interface.

#### Acceptance Criteria

1. WHERE FAISS is enabled as the vector search backend, WHEN a query requests the top-k documents, THE Dense_Retriever SHALL return no more than k documents ordered by descending similarity score, where k is a positive integer from 1 to the total number of indexed documents.
2. WHERE FAISS is enabled as the vector search backend, THE Dense_Retriever SHALL expose the same public method signatures, input parameters, and return structure as the from-scratch NumPy implementation, such that switching backends requires no change to pipeline caller code.
3. WHERE FAISS is enabled as the vector search backend, WHEN the same query and document set are submitted to both backends, THE Dense_Retriever SHALL return rankings whose top-k document ordering matches the NumPy implementation for at least 99 percent of the returned positions, with any per-document similarity score differing by no more than 0.0001.
4. IF FAISS is enabled but the FAISS backend fails to initialize or execute a search, THEN THE Dense_Retriever SHALL return an error indicating the vector search backend is unavailable, SHALL preserve the existing index data without modification, and SHALL NOT return partial or unranked results.
5. IF a query requests a top-k value less than 1 or the index contains zero documents, THEN THE Dense_Retriever SHALL return an error indicating the requested top-k is invalid and SHALL return an empty result set.

### Requirement 17: Ollama Runtime and Connectivity

**User Story:** As a RAG engineer, I want to run Ollama either as a native local install or as a Docker container and have all components connect to it through a configurable local endpoint, so that I can choose my runtime without changing code.

#### Acceptance Criteria

1. THE System SHALL read the Ollama Endpoint base URL from configuration, provided as an environment variable and/or a configuration value, and SHALL default the Ollama Endpoint to `http://localhost:11434` when it is unset.
2. WHERE Ollama runs as a native local installation, THE System SHALL connect to it at the configured Ollama Endpoint without source-code changes.
3. WHERE Ollama runs as a Docker container, THE System SHALL connect to it at the configured Ollama Endpoint, such as a published container port on localhost, without source-code changes other than configuration.
4. THE System SHALL ensure that every component that requires an Ollama model, including the Ollama_Embedder, the Generator, and the LLM_Reranker, uses the same configured Ollama Endpoint.
5. WHEN the System starts a phase that needs Ollama, THE System SHALL verify that the configured Ollama Endpoint is reachable and that the required model or models are available.
6. IF the configured Ollama Endpoint is not reachable or a required model is missing when a phase that needs Ollama starts, THEN THE System SHALL return a descriptive error naming the configured Ollama Endpoint and the missing model, without falling back to any external or paid API.
7. THE System SHALL document, such as in the README, how to run Ollama both as a native installation and as a Docker container, including pulling the required models (`embeddinggemma` and the chosen instruct model) and setting the Ollama Endpoint, and SHALL treat a native installation as equally supported so that Docker is not required.
8. THE System SHALL connect only to a locally reachable Ollama instance at the configured Ollama Endpoint and SHALL NOT bundle or require any cloud-hosted or external Ollama service.
