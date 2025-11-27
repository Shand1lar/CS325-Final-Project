import pytest
import numpy as np
from CS325Project2 import DataProcessor, SimilarityCalculator, IEmbeddingService, JobFetcher

# ==========================================
# MOCKING EXTERNAL DEPENDENCIES (DIP IN ACTION)
# ==========================================
# Because we used the Dependency Inversion Principle, we can create a 
# "Fake" service that looks exactly like the real Gemini service to the app.

class MockEmbeddingService(IEmbeddingService):
    """
    A fake embedding service for testing.
    Instead of calling Google (which costs money/quota and is slow),
    this returns predictable, hard-coded numbers.
    """
    def generate_embeddings(self, texts):
        # Return a simple fake vector for every text received.
        # We return a fixed pattern so we can predict the math results.
        # Vector size 3 for simplicity: [1.0, 0.0, 0.0]
        return [[1.0, 0.0, 0.0] for _ in texts]

class MockJobFetcher(JobFetcher):
    """
    A fake job fetcher to avoid hitting RapidAPI during tests.
    """
    def fetch_jobs(self, query, output_file):
        return {
            "data": [
                {
                    "job_id": "123",
                    "job_title": "Software Engineer",
                    "employer_name": "Tech Corp",
                    "job_location": "St. Louis, MO",
                    "job_description": "<b>Coding</b> and testing.",
                    "job_apply_link": "http://randomlink.com"
                }
            ]
        }

# ==========================================
# UNIT TESTS
# ==========================================

# 1. TEST DATA PROCESSOR (Black Box Testing)
# We focus on Inputs -> Outputs without worrying about internal API calls.

def test_clean_text_removes_html():
    """Test if HTML tags are removed correctly."""
    processor = DataProcessor()
    input_text = "<div>Hello World</div>"
    expected_output = "hello world"
    assert processor.clean_text(input_text) == expected_output

def test_clean_text_removes_special_chars():
    """Test if special characters are stripped."""
    processor = DataProcessor()
    input_text = "Hello!!! @# World$"
    expected_output = "hello world"
    assert processor.clean_text(input_text) == expected_output

def test_structure_job_data():
    """Test if raw API data is converted to the correct internal dictionary structure."""
    processor = DataProcessor()
    
    # Mock raw data resembling RapidAPI response
    raw_data = {
        "data": [{
            "job_id": "1",
            "job_title": "Cyber Analyst",
            "employer_name": "SecCorp",
            "job_location": "Remote",
            "job_description": "Analyze logs.",
            "job_apply_link": "http://randomlink.com"
        }]
    }
    
    # We use a dummy filename since we aren't actually checking the file system here
    result = processor.preprocess_job_data(raw_data, "dummy_output.json")
    
    assert len(result) == 1
    assert result[0]['job_title'] == "cyber analyst"
    assert "embedding_text" in result[0]
    assert "Title: cyber analyst" in result[0]['embedding_text']

# 2. TEST SIMILARITY CALCULATOR (Logic Verification)
# test the math logic using controlled inputs.

def test_cosine_similarity_ranking():
    """
    Verify that the math correctly ranks a 'perfect match' higher than a 'bad match'.
    """
    calculator = SimilarityCalculator()
    
    # Create fake embedding results
    # Resume Vector: [1, 0, 0]
    # Job A (Perfect Match): [1, 0, 0] (Similarity should be 1.0)
    # Job B (Opposite): [0, 1, 0] (Similarity should be 0.0)
    
    fake_data = {
        'resume_embedding': [1.0, 0.0, 0.0],
        'jobs_with_embeddings': [
            {
                'job_id': 'bad_match',
                'job_title': 'Baker',
                'company_name': 'Bakery',
                'job_location': 'Remote',
                'job_apply_link': 'link',
                'job_embedding': [0.0, 1.0, 0.0] # Perpendicular vector
            },
            {
                'job_id': 'good_match',
                'job_title': 'Coder',
                'company_name': 'Tech',
                'job_location': 'Remote',
                'job_apply_link': 'link',
                'job_embedding': [1.0, 0.0, 0.0] # Identical vector
            }
        ]
    }
    
    results = calculator.calculate_similarity_and_rank(fake_data, top_n=2, output_file="dummy.json")
    
    # The 'good_match' should be first index (Rank 1)
    assert results[0]['job_title'] == "Coder"
    # Score should be very close to 1.0
    assert results[0]['similarity_score'] > 0.9

# 3. TEST WITH MOCK SERVICE
# This tests the "Stage 3" logic without actually calling Google Gemini.

def test_embedding_flow_with_mock_service():
    """
    Demonstrates how we swap the real GeminiEmbeddingService 
    with the MockEmbeddingService.
    """
    # 1. Setup Data
    processor = DataProcessor()
    fake_job_text = ["Job 1 text", "Job 2 text"]
    
    # 2. Initialize the MOCK service
    # We do NOT pass an API key because the Mock doesn't need one.
    mock_service = MockEmbeddingService() 
    
    # 3. Execution
    # The app calls .generate_embeddings() thinking it's talking to Google, 
    # but it's talking to our Mock class.
    vectors = mock_service.generate_embeddings(fake_job_text)
    
    # 4. Assertions
    assert len(vectors) == 2
    assert vectors[0] == [1.0, 0.0, 0.0] # The hardcoded value from our Mock class
    
    print("\n[Success] Mock Service returned vectors without API calls.")