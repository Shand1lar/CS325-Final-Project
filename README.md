# Job Recommendation Pipeline

## What this code does

This is a Python pipeline that fetches job postings using the JSearch API, cleans and preprocesses the data, creates text embeddings using an embedding service (e.g., Google Gemini), and finally ranks the jobs by similarity to a user’s resume. The end result is a list of top job recommendations tailored to the resume content.

## How to set up the environment and install dependencies

1. **Clone or copy the project files** to your local machine.

2. **Create or update `requirements.txt`** with the necessary packages. At minimum, this project uses:
   ```
   requests
   numpy
   scipy
   pytest
   ```
   Add any additional packages you plan to use.

3. **Install the dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure API keys and settings**:
   - Open `CS325Project2.py`.
   - Set `RAPIDAPI_KEY` with your RapidAPI key for JSearch, if you want to fetch jobs live.
   - Set `EMBEDDING_API_KEY` for your embedding service (e.g., Google Gemini).
   - Optionally modify `JOB_SEARCH_QUERY`, file paths, or the `LOAD_EXISTING_CLEANED_DATA` flag.
   - Ensure the resume file path (`RESUME_FILE`) points to your resume text file.

## How to run the application

1. **Ensure your dependencies are installed.**

2. **Run the main script**:
   ```bash
   python CS325Project2.py
   ```

   What happens:
   - Stage 1: Fetch raw job data from the JSearch API, unless `LOAD_EXISTING_CLEANED_DATA` is `True`.
   - Stage 2: Clean and preprocess the job data, and load the resume text.
   - Stage 3: Generate embeddings for the resume and job postings using the configured embedding service.
   - Stage 4: Compute cosine similarity between the resume embedding and each job embedding, rank the jobs, and save the top results.

3. **Check output files** created during the run:
   - `jsearch_results.json` — raw API output.
   - `cleaned_job_data.json` — cleaned and structured job postings.
   - `embedded_job_data.json` — job postings with embeddings.
   - `top_recommendations.json` — final ranked recommendations for the top N jobs.

4. **Adjust parameters** as needed, for example:
   - Change `TOP_N_RECOMMENDATIONS` to get more or fewer suggestions.
   - Change the search query to target different job types or locations.

## How to run the test suite

1. **Ensure pytest is installed** in your environment:
   ```bash
   pip install pytest
   ```

2. **Run pytest** from the project root:
   ```bash
   pytest tests_CS325Project2.py
   ```
   This executes the tests defined in `tests_CS325Project2.py` (or any other test files), which use mocks or fake services to isolate components such as the embedding service or similarity calculator.

3. **Interpreting results**:
   - Passing tests confirm that components behave correctly on predefined inputs.
   - Failing tests can point to issues in data processing, dependency handling, or expected outputs when using mocks.

## Building and running with Docker

These commands assume you’re in the project root, where the `Dockerfile` is.

### 1) Build the Docker image

Create a `Dockerfile` 

```dockerfile
FROM python:3.10-slim

WORKDIR /home

COPY requirements.txt /home

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD {"python", "CS325Project2.py"}
```

Ensure you have all the files nescessary in the same folder including your `requirements.txt`
The CMD line will run the code that is in `CS325Project2.py`. Ensure you have *Configured API keys and settings* in the `CS325Project2.py`.

```bash
# Build an image and tag it as "CS325Project2:latest" (change the tag if you want)
docker build -t CS325Project2:latest .
```

- `.` tells Docker to use the current directory as the build context, picking up the `Dockerfile` in the root.
- You can choose a different tag or name if desired.

### 2) Run a container from the image

```bash
# Run the container interactively, removing it when stopped
docker run CS325Project2:latest
```
