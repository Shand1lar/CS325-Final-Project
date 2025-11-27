# Refactoring

## 1) SOLID principles 
- **Single Responsibility Principle (SRP)**
- **Dependency Inversion Principle (DIP)**

## 2) Why these principles

### Single Responsibility Principle
SRP says each module or class should have only one reason to change. A common explanation: a class should have one job. 
In the original, more monolithic version of the script, most of the logic lived in one place: fetching data, cleaning data, calling an embedding API, computing similarity, and writing outputs. That meant any change to one stage risked touching the same code for another stage. Splitting responsibilities reduces fragility and makes testing each stage easier.

### Dependency Inversion Principle
DIP says high‑level modules should not depend on low‑level modules; both should depend on abstractions. Abstractions should not depend on details; details should depend on abstractions. 
The original code used a concrete embedding API class directly inside the main logic. That couples the pipeline to one provider. Using DIP allows swapping embedding services (e.g., Gemini, OpenAI, or a mock) without rewriting the main pipeline, improving extensibility and testability.

## 3) How they were implemented

Below are before‑and‑after snippets

### A) Single Responsibility Principle (SRP)

SRP says each module or class should have a single responsibility or reason to change.
This means separating distinct stages or duties of the pipeline into their own units.

#### Before — monolithic script

The main script mixes many responsibilities in one file and in one flat structure. Example from Stage 1–4 flow:

```python
# --- STAGE 1 & 2 DATA PREPARATION (Job Postings) ---
if LOAD_EXISTING_CLEANED_DATA:
    cleaned_jobs = load_preprocessed_jobs(CLEANED_OUTPUT_FILE)

if not cleaned_jobs:
    raw_data = get_external_data(search_term, OUTPUT_FILE)
    if raw_data:
        cleaned_jobs = preprocess_job_data(raw_data, CLEANED_OUTPUT_FILE)
    else:
        print("API fetch failed. Cannot proceed to embedding.")
        return

# ... later ...

embedding_results = create_embeddings(cleaned_jobs, processed_resume_text)
# then immediately rank and print
top_jobs = calculate_similarity_and_rank(embedding_results, TOP_N_RECOMMENDATIONS)
```

Functions for fetching, cleaning, embedding, ranking, and even file I/O are all in one file with limited separation between their responsibilities. The main orchestration and stage logic are tightly coupled in the same namespace and script.

#### After refactor — SRP applied

The pipeline is split into clearly defined classes, each focused on one stage or one type of work.

1) **JobFetcher** — Stage 1 only

```python
class JobFetcher:
    """
    [SRP] Responsible ONLY for Stage 1: Communicating with the JSearch (RapidAPI) source.
    """
    def __init__(self, api_key, host, url):
        self.api_key = api_key
        self.host = host
        self.url = url

    def fetch_jobs(self, query, output_file):
        # Fetch raw job data, handle errors, save to file
        ...
```

2) **DataProcessor** — Stage 2 only

```python
class DataProcessor:
    """
    [SRP] Responsible ONLY for Stage 2: Cleaning text, structuring JSON, and File I/O.
    """

    def clean_text(self, text):
        # text cleaning logic
        ...

    def preprocess_job_data(self, raw_data, output_file):
        # combine, filter, save cleaned jobs
        ...

    def load_resume_data(self, resume_file):
        # load and clean resume text
        ...
```

3) **SimilarityCalculator** — Stage 4 only

```python
class SimilarityCalculator:
    """
    [SRP] Responsible ONLY for Stage 4: Mathematical calculations (Cosine Similarity) and Ranking.
    """
    def calculate_similarity_and_rank(self, embedding_results, top_n, output_file):
        # vector math, ranking, save results
        ...
```

**How SRP was implemented:**
- Each class encapsulates exactly one pipeline stage or responsibility.
- File I/O, API calls, data cleaning, and ranking live in separate classes, not in one monolithic script.
- The main orchestration simply wires these classes together instead of doing stage logic itself.

---

### B) Dependency Inversion Principle (DIP)

DIP states high‑level modules should not depend on low‑level modules; both should depend on abstractions. Abstractions should not depend on details; details should depend on abstractions.

#### Before — direct dependency on embedding

Embedding logic is called directly through other functions and parameters.

```python
# Stage 3 – direct call to embed_text
all_embeddings = embed_text(
    texts=embedding_texts,
    model=EMBEDDING_MODEL,
    api_key=EMBEDDING_API_KEY,
    api_url_base=EMBEDDING_URL
)
```

Here the main flow depends directly on the specific implementation of embedding via the `embed_text` function and Gemini parameters. Swapping providers or mocking would still require careful changes around this call or in the helper itself.

#### After — DIP applied 

An abstract interface for embedding is added, and the main flow depends on this abstraction rather than a concrete implementation.

1) **Abstract embedding interface**

```python
class IEmbeddingService(ABC):
    """
    Abstract Interface for an embedding service.
    """
    @abstractmethod
    def generate_embeddings(self, texts):
        pass
```

2) **Concrete Gemini implementation**

```python
class GeminiEmbeddingService(IEmbeddingService):
    """
    Concrete implementation using Google Gemini.
    Responsible ONLY for the technical details of calling the Google API.
    """
    def __init__(self, api_key, model, base_url):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url

    def generate_embeddings(self, texts):
        # perform API call, retries, return embeddings
        ...
```

3) **Main orchestration uses the interface method**

```python
# --- MAIN EXECUTION ---
def main():
    # initialize classes
    job_fetcher = JobFetcher(RAPIDAPI_KEY, RAPIDAPI_HOST, RAPIDAPI_URL)
    data_processor = DataProcessor()
    embedding_service = GeminiEmbeddingService(EMBEDDING_API_KEY, EMBEDDING_MODEL, EMBEDDING_URL)
    similarity_calculator = SimilarityCalculator()

    # ... after cleaning ...
    embedding_texts = [processed_resume_text] + [job['embedding_text'] for job in cleaned_jobs]

    # [SOLID - DIP] call through interface
    all_embeddings = embedding_service.generate_embeddings(embedding_texts)
    ...
```

**How DIP was implemented:**
- High‑level pipeline code depends on `IEmbeddingService`, not on a concrete embedding helper or provider.
- `GeminiEmbeddingService` implements the interface; a different provider or a mock could implement the same interface without changing the main logic.
- Embedding parameters are encapsulated in the concrete provider; main flow only needs to know the provider implements the abstract `generate_embeddings` method.

---
