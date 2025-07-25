import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Model Configuration
LLM_MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
CLASSIFIER_MODEL_ID = "all-MiniLM-L6-v2"
SPACY_MODEL_ID = "en_core_web_sm"

# CSA Configuration
CSA_DEPENDENCY_THRESHOLD = 0.65
PRONOUN_SCORE = 0.95
ENTITY_DEFICIT_SCORE = 0.80
SELF_CONTAINED_PENALTY = 0.5

# Authentication
HF_TOKEN = os.getenv("HF_TOKEN")

# Global model cache
_MODEL_CACHE = {
    "llm_tokenizer": None,
    "llm_model": None,
    "classifier_model": None,
    "nlp_model": None,
    "loaded": False
}
