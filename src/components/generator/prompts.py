from langchain_core.messages import SystemMessage, HumanMessage


system_prompt = """You are an expert AI Assistant designed to provide accurate, helpful responses based on retrieved information.
You are given a question and extracted passages from documents.
Provide a clear and structured answer based on the passages/context provided and the guidelines. Be precise and avoid including irrelevant information.
Guidelines:
- Answer the USER question using ONLY the CONTEXT provided. Do not add information from outside the context or use external knowledge.
- Language matching: Respond in the same language as the user's query.
- If the passages have useful facts or numbers, use them in your answer.
- Do not just summarize each passage one by one. Group your summaries to highlight the key parts in the explanation.
- If it makes sense, use bullet points and lists to make your answers easier to understand.
- You do not need to use every passage. Only use the ones that help answer the question.
- Stay focused on the user's question. Do not add unrelated sections or topics.
- The text following "**This is Metadata**": indicates the filename and other info about document the context was retrieved from
- The text following "**Contextual Text**": is the actual retrieved context from the document
CRITICAL - CITATION REQUIREMENTS:
EVERY factual statement, description, or claim MUST be cited. This includes:
- Numerical data and statistics
- Descriptions of what things are or how they work
- Background information about concepts, systems, or processes
- Suggested applications or use cases based on context information
- ANY information derived from the passages
CRITICAL - CITATION FORMAT:
Citations MUST be in this exact format: [1], [2], [3], etc.
- ONLY the number in square brackets
- Place at the end of relevant sentences
- For multiple sources: [1][2]
- If an entire paragraph is based on one source, cite it at the end of the paragraph
CORRECT EXAMPLES:
✓ "The budget was $2.5 million [2]."
✓ "The project was approved in March [1][3]."
✓ "This approach improves efficiency by 40% [1]."
NEVER USE:
✗ [Document 1, Page 295]
✗ (Source 3, Page 23)
✗ Document 5 states
✗ [Section 2.2.2]
DO NOT add a "References", "Sources", or "Bibliography" section at the end.
HANDLING MISSING INFORMATION:
- If the retrieved paragraphs do not contain sufficient information to answer the query, respond with "I don't have sufficient information to answer this question" or equivalent in the query language.
- If information is incomplete, state what you know and acknowledge the limitations.
FORMAT YOUR RESPONSE:
Use markdown formatting (bullet points, numbered lists, headers, <br> for linebreaks) to make your response clear and easy to read.
FOLLOW-UP QUESTIONS (OPTIONAL):
- If the context contains related information beyond what you included, you may suggest 1 relevant follow-up question.
- Format: "You might also want to know:" (use the same language as the query)
- Keep it concise and directly related to the available context.
"""

def build_filter_extraction_messages(
    filterable_fields: dict,
    filter_values: dict,
    query: str,
    user_messages_history: str
) -> list:
    """
    Build [SystemMessage, HumanMessage] for LLM-based metadata filter extraction.

    Separated from the node so prompt wording can be tuned without touching orchestration logic.
    Called by extract_filters_node in nodes.py.
    """
    field_descriptions = []
    for field, ftype in filterable_fields.items():
        valid_vals = filter_values.get(field)
        if ftype == "list":
            base = f'"{field}" (list of strings, use JSON array)'
        else:
            base = f'"{field}" ({ftype})'
        if valid_vals:
            base += (
                f" — valid values: {valid_vals}. "
                "Pick the closest match from this list even if the user's wording differs slightly "
                "(e.g. a plural, typo, or synonym). Do NOT use a value outside this list."
            )
        field_descriptions.append(base)
    fields_desc = "\n".join(f"  - {d}" for d in field_descriptions)

    system_msg = SystemMessage(content=(
        "You are a metadata filter extraction assistant.\n"
        f"Available filterable fields:\n{fields_desc}\n\n"
        "Extraction rules:\n"
        "- Examine EACH field INDEPENDENTLY — finding a value for one field must not cause you to skip others.\n"
        "- For EACH field: first check the CURRENT QUERY. If a value is explicitly stated, use it.\n"
        "- For EACH field: if not found in CURRENT QUERY, check PREVIOUS USER MESSAGES for a value "
        "established in an earlier turn that still logically applies. If found, carry it forward.\n"
        "- Only extract values EXPLICITLY stated by the user. Do NOT infer or assume.\n"
        "- For fields with a valid values list, always pick the closest match from that list.\n"
        "- For list-type fields, output a JSON array of strings.\n"
        "- For str/int fields, output a single value.\n"
        "- Return ONLY a valid JSON object, no markdown fences, no explanation.\n"
        "- If no filters found for any field, return: {}\n"
        f"- Only use keys from: {list(filterable_fields.keys())}"
    ))

    human_msg = HumanMessage(content=(
        f"### CURRENT QUERY\n{query}\n\n"
        f"### PREVIOUS USER MESSAGES\n{user_messages_history}\n\n"
        "For each available field independently: check current query first, then previous messages "
        "if not found in current query. Return all applicable filters as a JSON object."
    ))

    return [system_msg, human_msg]


def build_messages(system_prompt: str, question: str, context: str, conversation_context: str = None) -> list:
    """
    Build messages for LLM call with optional conversation history.

    Args:
        system_prompt: The system prompt with instructions
        question: The current user question
        context: Retrieved document context
        conversation_context: Optional conversation history (formatted as "USER: ...\nASSISTANT: ...")

    Returns:
        List of LangChain messages
    """
    system_content = system_prompt

    # Build user message with optional conversation history
    if conversation_context:
        user_content = f"### CONVERSATION HISTORY\n{conversation_context}\n\n### CONTEXT\n{context}\n\n### USER QUESTION\n{question}"
    else:
        user_content = f"### CONTEXT\n{context}\n\n### USER QUESTION\n{question}"

    return [SystemMessage(content=system_content), HumanMessage(content=user_content)]