# Test questions for RAG evaluation
# Edit each subset to add, remove, or swap queries without touching eval logic.
#
# Three subsets:
#   standalone_questions  — each query explicitly contains a filterable value; evaluated with no history
#   history_blocks        — conversation sequences; later turns rely on history carry-forward for filter extraction
#                           each block is a list of user turns; only the LAST turn per block is evaluated,
#                           with prior turns passed as user_messages_history (sliced to MAX_TURNS)
#   safeguard_questions   — multi-field combos likely to yield 0 results; tests AND-fallback to priority field

# --- Subset 1: Standalone ---
standalone_questions = [
    "What are the best irrigation techniques for wheat?",
    "How can I protect my maize crops from insects?",
]

# --- Subset 2: History blocks ---
# Each block = list of user turns (oldest → newest).
# The last turn is the one being evaluated; all prior turns become user_messages_history.
history_blocks = [
    [
        "I'm looking for information about wheat crop management.",   # establishes crop filter
        "What are the recommended pesticide applications?",           # vague — relies on history
    ],
    [
        "Tell me about maize cultivation in arid regions.",           # establishes crop filter
        "What soil preparation is needed?",                           # vague — relies on history
    ],
]

# --- Subset 3: Safeguard ---
# Queries that combine fields in a way unlikely to match — triggers AND-filter fallback to priority field.
safeguard_questions = [
    "What does the maize report on wheat fertilisation say?",         # contradictory crop + title combo
]
