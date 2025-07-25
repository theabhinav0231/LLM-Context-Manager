import torch

class KVCacheManager:
    """KV cache manager with manual generation for optimization."""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        
    def generate_with_kv_cache(self, prompt, past_kv_cache=None, max_new_tokens=200):
        """Generate response using KV cache optimization."""
        print(f"\n> Generating response...")
        
        if past_kv_cache is None:
            # First turn - establish context
            messages = [{"role": "user", "content": prompt}]
            
            try:
                text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except:
                text = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            
            model_inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **model_inputs,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                    return_dict_in_generate=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                )
            
            # Extract response
            input_length = model_inputs["input_ids"].shape[1]
            response_tokens = outputs.sequences[0][input_length:]
            response_text = self.tokenizer.decode(response_tokens, skip_special_tokens=True).strip()
            new_kv_cache = outputs.past_key_values
            
            print(f"> LLM: {response_text}")
            return response_text, new_kv_cache
            
        else:
            # KV cache reuse - manual generation
            print(f"> 🚀 Using KV cache optimization!")
            
            follow_up = f"<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            new_tokens = self.tokenizer(follow_up, return_tensors="pt").to(self.device)
            
            return self._manual_generation_with_cache(new_tokens["input_ids"], past_kv_cache, max_new_tokens)
    
    def _manual_generation_with_cache(self, input_ids, past_kv_cache, max_new_tokens):
        """Manual token-by-token generation preserving KV cache."""
        generated_ids = input_ids.clone()
        current_kv_cache = past_kv_cache
        
        for step in range(max_new_tokens):
            with torch.no_grad():
                if step == 0:
                    outputs = self.model(
                        input_ids=input_ids,
                        past_key_values=current_kv_cache,
                        use_cache=True,
                    )
                else:
                    outputs = self.model(
                        input_ids=generated_ids[:, -1:],
                        past_key_values=current_kv_cache,
                        use_cache=True,
                    )
            
            # Apply sampling
            next_token_logits = outputs.logits[0, -1, :] / 0.7
            
            # Top-p sampling
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > 0.9
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
            next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample next token
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Check for EOS
            if next_token.item() == self.tokenizer.eos_token_id:
                break
            
            # Add token to sequence
            generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0)], dim=1)
            current_kv_cache = outputs.past_key_values
        
        # Extract response
        response_start = input_ids.shape[1]
        response_tokens = generated_ids[0][response_start:]
        response_text = self.tokenizer.decode(response_tokens, skip_special_tokens=True).strip()
        response_text = response_text.replace("<|eot_id|>", "").strip()
        
        print(f"> LLM: {response_text}")
        return response_text, current_kv_cache
