import requests  # Used for making HTTP requests (for JSearch API and Gemini API)
import json      # Used for saving and loading data locally
import re        # Used for regex operations for text cleaning
import time      # Used for exponential backoff delay
import numpy as np # For vector operations
from scipy.spatial.distance import cosine # For similarity calculation
from abc import ABC, abstractmethod # [SOLID - DIP] Required for creating abstract interfaces

# --- CONFIGURATION -------------------------------------------------------------------------

# Filepaths for saving/loading data across stages
OUTPUT_FILE = "jsearch_results.json"                      # Raw API data (Stage 1 Output)
CLEANED_OUTPUT_FILE = "cleaned_job_data.json"             # Cleaned job data (Stage 2 Output)
EMBEDDINGS_OUTPUT_FILE = "embedded_job_data.json"         # Job data with embeddings (Stage 3 Output)
RECOMMENDATIONS_OUTPUT_FILE = "top_recommendations.json"  # Final ranked jobs (Stage 4 Output)

RESUME_FILE = "Resume.txt"        # Resume for embedding
TOP_N_RECOMMENDATIONS = 10        # Target number of recommendations

#---------Testing Flag---------------------------------------------------------------------
# Set to True to skip the live API call and load data from CLEANED_OUTPUT_FILE
# Set to False to run the full pipeline
LOAD_EXISTING_CLEANED_DATA = False

# --- RAPID API CONFIG (JSearch - Stage 1) ------------------------------------------
RAPIDAPI_KEY = ""                                        # !!! RapidAPI Key !!! 
RAPIDAPI_HOST = "jsearch.p.rapidapi.com"                 # RapidAPI Host
RAPIDAPI_URL = "https://jsearch.p.rapidapi.com/search"   # RapidAPI URL 

# --- GEMINI EMBEDDING API CONFIG (Stage 3) ------------------------------------------
EMBEDDING_API_KEY = ""   # !!! Google AI stuido API Key !!!
EMBEDDING_URL = "https://generativelanguage.googleapis.com/v1beta/models" 
EMBEDDING_MODEL = "text-embedding-004" 

# --- JOB SEARCH string ------------------------------------------------
JOB_SEARCH_QUERY = "Cyber Security jobs in chicago" 

# =================================================================================================
# [SOLID PRINCIPLE 2] DEPENDENCY INVERSION PRINCIPLE (DIP)
# High-level modules should not depend on low-level modules. Both should depend on abstractions.
# We create an Interface (Abstract Base Class) for embedding. 
# The main app will talk to this Interface, not directly to the Gemini API code.
# =================================================================================================

class IEmbeddingService(ABC):
    """
    Abstract Interface for an embedding service. 
    This allows us to swap Gemini for OpenAI without breaking the rest of the code.
    """
    @abstractmethod
    def generate_embeddings(self, texts):
        pass

class GeminiEmbeddingService(IEmbeddingService):
    """
    Concrete implementation of the Embedding Service using Google Gemini.
    Responsible ONLY for the technical details of calling the Google API.
    """
    def __init__(self, api_key, model, base_url):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url

    def generate_embeddings(self, texts):
        """
        Calls the Gemini API's batchEmbedContents endpoint to generate embeddings for a list of texts 
        using exponential backoff for rate limit handling.
        """
        if not self.api_key:
            print("Embedding Error: EMBEDDING_API_KEY is missing. Cannot generate embeddings.")
            return None

        full_api_url = f"{self.base_url}/{self.model}:batchEmbedContents"
        
        requests_payload = [
            {
                "model": f"models/{self.model}", 
                "content": {"parts": [{"text": text}]}
            } for text in texts
        ]

        payload = {"requests": requests_payload}
        headers = {'Content-Type': 'application/json'}
        
        max_retries = 5
        initial_delay = 1 # seconds
        
        for attempt in range(max_retries):
            try:
                print(f"-> Sending batch of {len(texts)} texts to embedding API (Attempt {attempt + 1})...")
                # Using the key parameter in the URL for authentication
                response = requests.post(f"{full_api_url}?key={self.api_key}", headers=headers, data=json.dumps(payload), timeout=30)
                response.raise_for_status()
                
                result = response.json()
                embeddings = [item['values'] for item in result.get('embeddings', [])]
                return embeddings

            except requests.exceptions.HTTPError as e:
                if response.status_code == 429 and attempt < max_retries - 1:
                    delay = initial_delay * (2 ** attempt)
                    # Print rate limit hit without logging as an error
                    print(f"-> Rate limit hit (429). Retrying in {delay:.2f} seconds...")
                    time.sleep(delay)
                else:
                    print(f"-> HTTP Error during embedding (Status {response.status_code}): {e}")
                    print("-> Exceeded max retries or unrecoverable error. Aborting embedding.")
                    return None
            except Exception as e:
                print(f"-> API call or response error. Error: {e}")
                return None
        return None

# =================================================================================================
# [SOLID PRINCIPLE 1] SINGLE RESPONSIBILITY PRINCIPLE (SRP)
# Break the monolithic script into classes that have one specific reason to change.
# =================================================================================================

class JobFetcher:
    """
    [SRP] Responsible ONLY for Stage 1: Communicating with the JSearch (RapidAPI) source.
    """
    def __init__(self, api_key, host, url):
        self.api_key = api_key
        self.host = host
        self.url = url 

    def fetch_jobs(self, query, output_file):
        """
        Fetches job data from the JSearch API (via RapidAPI) based on the search query.
        Implements error handling for network requests.
        """
        print(f"\n--- STAGE 1: FETCHING JOB DATA for '{query}' ---")
        
        if not self.api_key or not self.host:
            print("API Error: RAPIDAPI_KEY or RAPIDAPI_HOST is missing. Skipping external data fetch.")
            return None

        querystring = {
            "query": query,
            "page": "1",
            "num_pages": "5",
            "country": "us"
        }

        headers = {
            "X-RapidAPI-Key": self.api_key,
            "X-RapidAPI-Host": self.host
        }

        try:
            response = requests.get(self.url, headers=headers, params=querystring, timeout=30)
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

            raw_data = response.json()
            
            # Save raw data for inspection
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(raw_data, f, indent=4)
            print(f"[Data Saved] Raw job data saved to {output_file}")

            return raw_data
            
        except requests.exceptions.RequestException as e:
            print(f"API Fetch Error: Could not connect or retrieve data from JSearch API. Error: {e}")
            return None

class DataProcessor:
    """
    [SRP] Responsible ONLY for Stage 2: Cleaning text, structuring JSON, and File I/O.
    """
    
    def clean_text(self, text):
        if text is None:
            return ""
        
        text=str(text).lower()                   # Lowercase
        text=text.replace('\xa0', ' ')           # Replace non-breaking spaces
        text=re.sub(r'<.*?>', ' ', text)         # Remove HTML tags
        text=re.sub(r'[^a-z0-9\s\.]', ' ', text) # Remove special characters exluding periods
        text=re.sub(r'\s+', ' ', text).strip()   # Final whitespace cleanup

        return text

    def load_preprocessed_jobs(self, file_path):
        """Loads cleaned job data from a file for Stage 3/4 processing."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                cleaned_data = json.load(f)
            print(f"[Data Loaded] Loaded {file_path} for Stage 3 processing, skipping API fetch.")
            return cleaned_data
        except FileNotFoundError:
            print(f"[File Error] Cannot load preprocessed jobs: Required file '{file_path}' not found.")
            return None
        except Exception as e:
            print(f"[File Error] Error loading file {file_path}: {e}")
            return None

    def preprocess_job_data(self, raw_data, output_file):
        """Cleans job data fields and combines them into a single text field for embedding."""
        
        print("\n--- STAGE 2: PREPROCESSING JOB DATA ---")
        jobs = raw_data.get('data', [])
        if not jobs:
            print("No job data found in the raw API response to preprocess.")
            return None
        
        cleaned_jobs = []
        
        for job in jobs:
            # 1. Handle missing data and clean text
            cleaned_job = {
                'job_id': job.get('job_id', 'N/A'),
                'job_title': self.clean_text(job.get('job_title')),
                'company_name': self.clean_text(job.get('employer_name')),
                'job_location': self.clean_text(job.get('job_location')),
                'job_description_cleaned': self.clean_text(job.get('job_description')),
                'job_apply_link': job.get('job_apply_link', 'N/A')
            }
            
            # 2. Combine key fields for the final embedding text
            cleaned_job['embedding_text'] = (
                f"Title: {cleaned_job['job_title']}. "
                f"Company: {cleaned_job['company_name']}. "
                f"Location: {cleaned_job['job_location']}. "
                f"Description: {cleaned_job['job_description_cleaned']}"
            )
            
            # 3. Filter out entries with no meaningful data
            if not cleaned_job['job_description_cleaned']:
                continue

            cleaned_jobs.append(cleaned_job)

        try:
            # Save cleaned and structured job posting data
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(cleaned_jobs, f, indent=4)
            print(f"[Data Saved] Cleaned job data ({len(cleaned_jobs)} entries) saved to {output_file}")
            return cleaned_jobs
        except Exception as e:
            print(f"[File Error] WARNING: Could not save cleaned data to file {output_file}. Error: {e}")
            return None

    def load_resume_data(self, resume_file):
        """Loads and cleans the resume text from a file."""
        print(f"\n--- STAGE 2: PROCESSING RESUME DATA ---")
        try:
            with open(resume_file, 'r', encoding='utf-8') as f:
                resume_text = f.read()
            cleaned_resume_text = self.clean_text(resume_text)
            print(f"[Resume Loaded] Successfully loaded and cleaned text from {resume_file}.")
            return cleaned_resume_text
        except FileNotFoundError:
            print(f"[File Error] WARNING: Resume file '{resume_file}' not found. Cannot proceed.")
            return ""
        except Exception as e:
            print(f"[File Error] WARNING: Could not read or process resume file. Error: {e}")
            return ""

class SimilarityCalculator:
    """
    [SRP] Responsible ONLY for Stage 4: Mathematical calculations (Cosine Similarity) and Ranking.
    """
    def calculate_similarity_and_rank(self, embedding_results, top_n, output_file):
        """
        Calculates cosine similarity, ranks jobs, and saves the top N recommendations.
        """
        print("\n--- STAGE 4: CALCULATING SIMILARITY AND RANKING ---")

        if not embedding_results or not embedding_results.get('jobs_with_embeddings'):
            print("Ranking Skipped: No embedded job data found.")
            return None

        jobs_with_embeddings = embedding_results['jobs_with_embeddings']
        # Convert list to numpy array for fast vector operations
        resume_vector = np.array(embedding_results['resume_embedding']) 
        
        ranked_jobs = []

        for job in jobs_with_embeddings:
            # Convert the job embedding list to a numpy array
            job_vector = np.array(job['job_embedding'])
            
            # Cosine similarity is 1 - Cosine distance.
            try:
                similarity = 1 - cosine(resume_vector, job_vector)
            except ValueError as e:
                # Skip if vector dimensions do not match
                print(f"Skipping job {job.get('job_id')}: Vector dimension mismatch. {e}")
                continue
                
            ranked_job = {
                'similarity_score': round(similarity, 4), 
                'job_title': re.sub(r'[^a-zA-Z\s\.]', ' ', job['job_title']).title().strip(), 
                'company_name': re.sub(r'\s*\.\s*', '.', job['company_name']).title().strip(),
                'job_location': job['job_location'].title(),
                'job_apply_link': job['job_apply_link']
            }
            ranked_jobs.append(ranked_job)

        # Sort the jobs by similarity score in descending order
        ranked_jobs.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        # Select the top N jobs (Stage 4 Deliverable)
        top_recommendations = ranked_jobs[:top_n]

        # Save the final recommendations
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(top_recommendations, f, indent=4)
            print(f"[Data Saved] Top {len(top_recommendations)} recommendations saved to {output_file}")
        except Exception as e:
            print(f"[File Error] WARNING: Could not save recommendations to file {output_file}. Error: {e}")

        return top_recommendations

# --- MAIN EXECUTION ---
def main():
    """The main function to orchestrate the data flow through all four stages."""
    
    job_fetcher = JobFetcher(RAPIDAPI_KEY, RAPIDAPI_HOST, RAPIDAPI_URL)
    data_processor = DataProcessor()
    embedding_service = GeminiEmbeddingService(EMBEDDING_API_KEY, EMBEDDING_MODEL, EMBEDDING_URL)
    similarity_calculator = SimilarityCalculator()

    search_term = JOB_SEARCH_QUERY
    cleaned_jobs = None
    
    # Load and process the resume text (Stage 2)
    processed_resume_text = data_processor.load_resume_data(RESUME_FILE)
    if not processed_resume_text:
        return # Cannot proceed without a resume

    # 1. --- STAGE 1 & 2 DATA PREPARATION (Job Postings) ---
    if LOAD_EXISTING_CLEANED_DATA:
        # Load previously cleaned job data to skip live API fetch
        cleaned_jobs = data_processor.load_preprocessed_jobs(CLEANED_OUTPUT_FILE)
    
    if not cleaned_jobs:
        # If loading failed or flag is False, run the full Stage 1 & 2 flow
        raw_data = job_fetcher.fetch_jobs(search_term, OUTPUT_FILE) 
        
        if raw_data:
            # Stage 2: Preprocess the fetched data
            cleaned_jobs = data_processor.preprocess_job_data(raw_data, CLEANED_OUTPUT_FILE)
        else:
            print("API fetch failed. Cannot proceed to embedding.")
            return

    if cleaned_jobs and processed_resume_text:
        print("\n--- STAGE 2 SUMMARY ---")
        print(f"Cleaned Jobs Ready: {len(cleaned_jobs)} entries")
        print(f"Processed Resume Text Length: {len(processed_resume_text)} characters")
        
        # 2. --- STAGE 3: EMBEDDING GENERATION ---
        print("\n--- STAGE 3: GENERATING EMBEDDINGS (Gemini API) ---")
        
        # Prepare texts for batch request: resume first, then all job descriptions
        embedding_texts = [processed_resume_text] + [job['embedding_text'] for job in cleaned_jobs]
        
        # call the interface method .generate_embeddings()
        all_embeddings = embedding_service.generate_embeddings(embedding_texts)
        
        if not all_embeddings:
            print("Embedding Failed: Could not retrieve embeddings from the API.")
            print("\n--- STAGE 3 FAILED ---")
            print("Embedding process failed. Check API key, quota, and file paths.")
            return

        # Separate and assign embeddings
        resume_embedding = all_embeddings[0]
        job_embeddings = all_embeddings[1:]
        
        jobs_with_embeddings = []
        if len(job_embeddings) != len(cleaned_jobs):
            print(f"Error: Job embedding count ({len(job_embeddings)}) does not match job count ({len(cleaned_jobs)}). Aborting.")
            return

        for job, embedding in zip(cleaned_jobs, job_embeddings):
            job['job_embedding'] = embedding
            jobs_with_embeddings.append(job)

        # Save the job data including embeddings (Stage 3 Output)
        try:
            with open(EMBEDDINGS_OUTPUT_FILE, 'w', encoding='utf-8') as f:
                json.dump(jobs_with_embeddings, f, indent=4)
            print(f"[Data Saved] Job data with embeddings saved to {EMBEDDINGS_OUTPUT_FILE}")
        except Exception as e:
            print(f"[File Error] WARNING: Could not save embedded data to file {EMBEDDINGS_OUTPUT_FILE}. Error: {e}")

        embedding_results = {
            'jobs_with_embeddings': jobs_with_embeddings,
            'resume_embedding': resume_embedding
        }
        
        if embedding_results and embedding_results.get('jobs_with_embeddings'):
            print("\n--- STAGE 3 SUMMARY ---")
            print(f"Resume Embedding Generated: Vector length {len(embedding_results['resume_embedding'])}")
            print(f"Job Embeddings Generated: {len(embedding_results['jobs_with_embeddings'])} entries")
            
            # 3. --- STAGE 4: SIMILARITY CALCULATION AND RANKING ---
            # Delegate to SimilarityCalculator
            top_jobs = similarity_calculator.calculate_similarity_and_rank(
                embedding_results, 
                TOP_N_RECOMMENDATIONS, 
                RECOMMENDATIONS_OUTPUT_FILE
            )
            
            if top_jobs:
                # Display the final ranked recommendations
                print(f"\n--- STAGE 4 FINAL RECOMMENDATIONS (Top {len(top_jobs)}) ---")
                for i, job in enumerate(top_jobs):
                    print(f"Rank {i+1} (Score: {job['similarity_score']:.4f}): {job['job_title']} at {job['company_name']}")
        

if __name__ == "__main__":
    main()