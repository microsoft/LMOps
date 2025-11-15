"""
Global variables for Deepscaler repo.
"""
# Gemini Vertex AI Config (for dataset preprocessing and LLM as ORM).
GCP_PROJECT_ID = None # Fill this in!
GCP_LOCATION = None # Fill this in!
GEMINI_MODEL = "gemini-1.5-pro-002"
OAI_RM_MODEL = "gpt-4o-mini"

# Reward function constants
THOUGHT_DELIMITER_START = "<think>"
THOUGHT_DELIMITER_END = "</think>"