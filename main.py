from model_loader import load_models
from conversation_manager import ConversationManager
from csa_classifier import csa_classifier
from config import LLM_MODEL_ID

def run_context_manager():
    """Main context manager with smart branching."""
    
    # Load models
    tokenizer, model, kv_manager, classifier, nlp = load_models()
    
    # Initialize conversation manager
    conversation_manager = ConversationManager()
    
    print(f"\n LLM Context Manager")
    print(f"Model: {LLM_MODEL_ID}")
    print("Type 'exit' to end session.\n")
    
    while True:
        user_prompt = input("Your Prompt: ").strip()
        if user_prompt.lower() == 'exit':
            print("Session ended.")
            break
        
        # Branch decision with CSA
        if conversation_manager.current_branch_id is None:
            # No context - start new branch
            branch_id = conversation_manager.start_new_branch()
            conversation_manager.current_branch_id = branch_id
            past_kv_cache = None
            print(f"> Starting new branch: {branch_id}")
            
        else:
            # Get last context for CSA analysis
            last_turn = conversation_manager.get_last_turn(conversation_manager.current_branch_id)
            
            if last_turn:
                last_context = {
                    'prompt': last_turn['prompt'],
                    'response': last_turn['response']
                }
                
                # Run CSA classifier
                should_continue = csa_classifier(
                    nlp, classifier, user_prompt, last_context
                )
                
                if should_continue:
                    # Continue on same branch - USE KV CACHE
                    branch_id = conversation_manager.current_branch_id
                    past_kv_cache = conversation_manager.get_last_kv_cache(branch_id)
                    print(f"> Continuing branch: {branch_id}")
                else:
                    # Start new branch - FRESH CONTEXT
                    branch_id = conversation_manager.start_new_branch()
                    conversation_manager.current_branch_id = branch_id
                    past_kv_cache = None
                    print(f"> Starting new branch: {branch_id}")
            else:
                # No previous turn - start new
                branch_id = conversation_manager.start_new_branch()
                conversation_manager.current_branch_id = branch_id
                past_kv_cache = None
                print(f"> Starting new branch: {branch_id}")
        
        # Generate using KV cache optimization
        response, new_kv_cache = kv_manager.generate_with_kv_cache(
            user_prompt, past_kv_cache
        )
        
        # Store turn with KV cache
        conversation_manager.add_turn_to_branch(
            branch_id, user_prompt, response, new_kv_cache
        )
        
        # Display tree state
        conversation_manager.display_conversation_tree()

if __name__ == "__main__":
    run_context_manager()
