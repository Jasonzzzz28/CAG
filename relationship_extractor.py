"""
Relationship Extractor

This script uses Google's Gemini 2.0 Flash Lite model
to extract relationships from a given text corpus.
"""

import os
import json
import argparse
from typing import Dict, List, Any, Optional
import google.generativeai as genai
from google.api_core.exceptions import GoogleAPIError

# Constants
MODEL_NAME = "models/gemini-2.0-flash-lite"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_OUTPUT_TOKENS = 2048

from dotenv import load_dotenv
load_dotenv()

def setup_gemini_api() -> None:
    """
    Set up the Gemini API with the API key from environment variables.
    """
    api_key = os.environ.get("GOOGLE_API_KEY")
    # api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY environment variable not set. "
            "Please set it with your Google API key."
        )
    
    genai.configure(api_key=api_key)

def extract_relationships_summary(
    text: str,
    temperature: float = DEFAULT_TEMPERATURE,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS
) -> str:
    """
    Extract relationships summary from the given text.
    
    Args:
        text: The text corpus to analyze
        temperature: Controls randomness in the output (0.0 to 1.0)
        max_output_tokens: Maximum number of tokens to generate
        
    Returns:
        A string containing the extracted relationships summary
    """
    # Set up the model
    setup_gemini_api()
    model = genai.GenerativeModel(
        model_name=MODEL_NAME,
        generation_config={
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
            "top_p": 0.95,
            "top_k": 40,
        }
    )
    
    # Create the prompt
    summary_instruction = (
    "Generate a concise summary of the provided context, focusing on the most important relationships, "
    "interactions, and dependencies between concepts, entities, or events. The summary should help a language model "
    "understand the structure and meaning of the content."
    )

    prompt = f"""
    # GOAL
    You are a summarization assistant that extracts the most critical relationships and logical structures from context for efficient language model understanding.
    Context is provided below. Summarize it briefly by highlighting key relationships (e.g., cause-effect, topic-subtopic, function-purpose, concept comparisons).

    # CONTEXT:
    {text}

    # TASK:
    {summary_instruction}

    # SUMMARY:
    """
    
    # Generate the response
    response = model.generate_content(prompt)
    return response.text

def extract_relationships(
    text: str,
    temperature: float = DEFAULT_TEMPERATURE,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    max_triplets: int = 20
) -> str:
    """
    Extract relationships from the given text.
    
    Args:
        text: The text corpus to analyze
        temperature: Controls randomness in the output (0.0 to 1.0)
        max_output_tokens: Maximum number of tokens to generate
        max_triplets: Maximum number of entity-relation triplets to extract
        
    Returns:
        A json format string containing the extracted relationships
    """
    # Set up the model
    model = genai.GenerativeModel(
        model_name=MODEL_NAME,
        generation_config={
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
            "top_p": 0.95,
            "top_k": 40,
        }
    )
    
    # Create the prompt
    max_knowledge_triplets = max_triplets
    
    prompt = f"""
    -Goal-
    Given a text document, identify all entities and their entity types from the text and all relationships among the identified entities.
    From the text, extract up to {max_knowledge_triplets} entity-relation triplets.

    -Output Constraint-
    Please ensure that the final output remains concise and within {max_output_tokens} tokens. Prioritize the most informative and relevant entities and relationships if needed.

    -Steps-
    1. Identify all entities. For each identified entity, extract the following fields:
        - "entity_name": Name of the entity, capitalized
        - "entity_type": Type of the entity
        - "entity_description": Comprehensive description of the entity's attributes and activities

    2. From the entities identified above, identify all clearly related entity pairs.
    For each relationship, extract the following fields:
        - "source_entity": Name of the source entity
        - "target_entity": Name of the target entity
        - "relation": Relationship type between source_entity and target_entity
        - "relationship_description": Brief explanation of the relationship

    3. Return a valid JSON object with the following structure:

    {{
    "entities": [
        {{
        "entity_name": "string",
        "entity_type": "string",
        "entity_description": "string"
        }},
        ...
    ],
    "relationships": [
        {{
        "source_entity": "string",
        "target_entity": "string",
        "relation": "string",
        "relationship_description": "string"
        }},
        ...
    ]
    }}

    -Real Data-
    ######################
    text: {text}
    ######################
    output:
    """

    
    response = model.generate_content(prompt)
    return response.text