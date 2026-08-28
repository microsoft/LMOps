"""
Global variables for Deepscaler repo.
"""
# Gemini Vertex AI Config (for dataset preprocessing and LLM as ORM).
GCP_PROJECT_ID = None # Fill this in!
GCP_LOCATION = None # Fill this in!
GEMINI_MODEL = "gemini-1.5-pro-002"
OAI_RM_MODEL = "gpt-4o-mini"
# OrcaRouter config: an OpenAI-compatible AI gateway for models and agents
# (https://www.orcarouter.ai). Set ORCAROUTER_API_KEY in your environment and
# override ORCAROUTER_RM_MODEL to use a different OrcaRouter model.
ORCAROUTER_API_BASE = "https://api.orcarouter.ai/v1"
ORCAROUTER_RM_MODEL = "openai/gpt-4o-mini"

# Reward function constants
THOUGHT_DELIMITER_START = "<think>"
THOUGHT_DELIMITER_END = "</think>"