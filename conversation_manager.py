class ConversationManager:
    """Manages conversation branches and KV cache optimization."""
    
    def __init__(self):
        self.conversation_tree = {}
        self.current_branch_id = None
    
    def start_new_branch(self):
        """Start a new conversation branch."""
        branch_id = f"branch_{len(self.conversation_tree)}"
        self.conversation_tree[branch_id] = {'turns': []}
        return branch_id
    
    def add_turn_to_branch(self, branch_id, prompt, response, kv_cache):
        """Store conversation turn with KV cache."""
        if branch_id not in self.conversation_tree:
            self.conversation_tree[branch_id] = {'turns': []}
        
        self.conversation_tree[branch_id]['turns'].append({
            'prompt': prompt,
            'response': response,
            'kv_cache': kv_cache,
        })
    
    def get_last_kv_cache(self, branch_id):
        """Get KV cache for context optimization."""
        if branch_id in self.conversation_tree and self.conversation_tree[branch_id]['turns']:
            return self.conversation_tree[branch_id]['turns'][-1]['kv_cache']
        return None
    
    def get_last_turn(self, branch_id):
        """Get last turn for CSA analysis."""
        if branch_id in self.conversation_tree and self.conversation_tree[branch_id]['turns']:
            return self.conversation_tree[branch_id]['turns'][-1]
        return None
    
    def display_conversation_tree(self):
        """Display conversation state."""
        print(f"\n📊 Conversation Tree:")
        for branch_id, branch_data in self.conversation_tree.items():
            turns = len(branch_data['turns'])
            print(f"   {branch_id}: {turns} turns")
