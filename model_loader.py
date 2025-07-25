import torch
import spacy
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
from kv_cache_manager import KVCacheManager
from config import *

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_spacy_model():
    """Load SpaCy model with automatic download."""
    try:
        spacy.load(SPACY_MODEL_ID)
    except OSError:
        print(f"> Downloading SpaCy model: {SPACY_MODEL_ID}")
        spacy.cli.download(SPACY_MODEL_ID)

def load_models():
    """Load all models with caching and optimization."""
    global _MODEL_CACHE
    
    # Return cached models if already loaded
    if _MODEL_CACHE["loaded"]:
        print("> Using cached models")
        return (
            _MODEL_CACHE["llm_tokenizer"],
            _MODEL_CACHE["llm_model"],
            _MODEL_CACHE["classifier_model"],
            _MODEL_CACHE["nlp_model"]
        )
    
    print(f"> Loading models...")
    
    # Load LLM
    llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID, trust_remote_code=True)
    if llm_tokenizer.pad_token is None:
        llm_tokenizer.pad_token = llm_tokenizer.eos_token
    
    if torch.cuda.is_available():
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.float16,
        )
        
        llm_model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_ID,
            token=HF_TOKEN,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        print(f"> LLM loaded on GPU ({torch.cuda.memory_allocated() / 1e9:.1f} GB)")
    else:
        llm_model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_ID,
            token=HF_TOKEN,
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True,
        )
        print("> LLM loaded on CPU")
    
    # Create KV cache manager
    kv_manager = KVCacheManager(llm_model, llm_tokenizer)
    
    # Load classifier
    classifier_model = SentenceTransformer(CLASSIFIER_MODEL_ID, device=device)
    
    # Load SpaCy
    load_spacy_model()
    nlp_model = spacy.load(SPACY_MODEL_ID)
    
    # Cache all models
    _MODEL_CACHE.update({
        "llm_tokenizer": llm_tokenizer,
        "llm_model": llm_model,
        "classifier_model": classifier_model,
        "nlp_model": nlp_model,
        "loaded": True
    })
    
    print("> ✅ All models loaded successfully!")
    return llm_tokenizer, llm_model, kv_manager, classifier_model, nlp_model
