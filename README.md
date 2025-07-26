# LLM-Context-Manager
LLM inference optimization system which smartly manages the context that is fed into the model for response generation through branching. It uses a novel contextual branching algorithm named as Contextual Scaffolding Algorithm (CSA) which determines if the previous context is needed for response generation or not. KV-cache for all responses is stored and can be called back and fed into the model if successive prompts require past context.

## This system optimizes LLM conversation handling by:

> Prevents context rot or context pollution. The LLM only recieves context which it requires. (Only signal No noise)

> Reducing inference costs by up to 50% (theoretically) through smart KV cache reuse.

> Managing conversation branches dynamically based on context relevance.

> Preserving conversation coherence while optimizing computational efficiency.

## Repo Structure
```
LLM_Context_Manager/
├── main.py                 # Main implementation
├── config.py               # Configuration settings
├── conversation_manager.py # Manages conversation branches and KV cache optimization
├── model_loader.py         # Loads required models
├── csa_classifier.py       # Implements CSA algorithm
├── kv_cache_manager.py     # KV-cache Manager
├── requirements.txt        # Dependencies
├── supported_models.txt    # List of all supported models
├── .env                    # Environment variable (HF_TOKEN)
└── README.md               

notebook/
├── llm_context_manager.ipynb  # .ipynb file with logs to print what's happening
└── README.md
```

## Working
Think of it as tree like branching where first prompt start as a new branch in the tree, if the next prompt is contextually related to the previous one then, conversation continues on the same branch. If however, next prompt is not contextually related to previous one or it doesn't require previous context, is "self-dependent" then, kv-cache in the models is cleaned up and stored (for future use). The model now starts with fresh new kv-cache values. This continues for the rest of conversation.

The contextual similarity of the next prompt is determined by a novel Contextual Similarity Algorithm (CSA). The working is as follows:
This algorithm combines three checks to produce a final "Dependency Score." It uses lightweight NLP tools like Part-of-Speech (POS) tagging and Named Entity Recognition (NER).

1. The Pronoun Check (Strong Signal):

  This is the most obvious signal of dependency.
  
  Action: Scan the new prompt for "anchor pronouns" that almost always refer to a previous context.
  
  Examples: it, its, that, those, they, their, them.
  
  Scoring: If an anchor pronoun is found, the Dependency Score is immediately set very high (e.g., 0.95). This is a strong indicator of a follow-up.
  
  Example: "What is its population?" -> High score. Dependent.

2. The Entity Deficit Check (Medium Signal):

  This checks if the new prompt is "missing" a core subject or entity.
  
  Action:
  
  Use NER to find all named entities (like places, people, organizations) in the new prompt.
  
  Check if the prompt is phrased as a question (starts with "What," "Why," "How," etc.).
  
  Scoring: If the prompt is a question but contains zero named entities, it has an "entity deficit." It's likely borrowing its subject from the previous context. The Dependency Score is set high (e.g., 0.80).
  
  Example: "Tell me more about the economy." -> Contains an entity ("economy"). No deficit.
  Example: "Why is that?" -> Is a question, has zero entities. Dependent.

3. The Semantic Fallback Check (The Tie-Breaker):

  This handles the tricky cases, like your "capital of India" example. It only runs if the first two checks don't find a clear dependency.
  
  Action:
  
  Calculate the standard semantic similarity score between the new prompt and the last context, just like we did before. Let's call this TopicSimilarity.
  
  Perform the Entity Deficit Check again. If the new prompt is self-contained (has its own entities), we penalize the TopicSimilarity score.
  
  Scoring Logic:
  
  if prompt_is_self_contained:
  
  DependencyScore = TopicSimilarity * 0.5  (We halve the score because even though the topic is similar, the prompt doesn't need the old context.)
  
  else:
  
  DependencyScore = TopicSimilarity (The topic is similar and the prompt isn't self-contained, so it's likely dependent.)



Note: **This project is work under progress. I plan to implement more optimization techniques like branch summarisation. A complete new chat UI and CLI!
