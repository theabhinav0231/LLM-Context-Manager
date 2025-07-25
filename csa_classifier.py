from sentence_transformers import util
from config import *
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

def csa_classifier(nlp, classifier, new_prompt, last_context):
    """
    Contextual Scaffolding Analysis for intelligent branching.
    
    Args:
        nlp: SpaCy NLP model
        classifier: Sentence transformer model  
        new_prompt: Current user prompt
        last_context: Previous conversation context
        
    Returns:
        bool: True for same branch, False for new branch
    """
    print("> Running CSA analysis...")
    
    if not last_context:
        return True
    
    doc = nlp(new_prompt)
    dependency_score = 0.0
    
    # 1. Pronoun check (strongest signal)
    anchor_pronouns = {"it", "its", "that", "those", "they", "their", "them"}
    if any(token.lower_ in anchor_pronouns for token in doc):
        dependency_score = PRONOUN_SCORE
        print(f"> Pronoun detected. Score: {dependency_score}")
    
    # 2. Entity deficit check
    elif dependency_score == 0.0:
        is_question = doc[0].pos_ == "AUX" or doc[0].tag_ in ["WDT", "WP", "WP$", "WRB"]
        has_entities = len(doc.ents) > 0
        if is_question and not has_entities:
            dependency_score = ENTITY_DEFICIT_SCORE
            print(f"> Entity-less question detected. Score: {dependency_score}")
    
    # 3. Semantic similarity fallback
    if dependency_score == 0.0:
        context_text = f"User Asked: {last_context['prompt']} | Model Response: {last_context['response']}"
        
        embedding_new = classifier.encode(new_prompt, convert_to_tensor=True).to(device)
        embedding_context = classifier.encode(context_text, convert_to_tensor=True).to(device)
        
        topic_similarity = util.cos_sim(embedding_new, embedding_context).item()
        
        # Apply self-contained penalty if prompt has entities
        is_self_contained = len(doc.ents) > 0
        if is_self_contained:
            dependency_score = topic_similarity * SELF_CONTAINED_PENALTY
        else:
            dependency_score = topic_similarity
    
    # Final decision
    if dependency_score > CSA_DEPENDENCY_THRESHOLD:
        print(f"> Decision: Continue branch (Score: {dependency_score:.2f})")
        return True
    else:
        print(f"> Decision: New branch (Score: {dependency_score:.2f})")
        return False
