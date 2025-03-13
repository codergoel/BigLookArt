#!/usr/bin/env python3
"""
End-to-End Artwork-Buyer Matching Using a Local Ollama Server (Completions Mode)
----------------------------------------------------------------------------------
This script demonstrates how to:
  1. Identify relevant buyer attributes for art matching using a local Ollama server.
  2. Gather unique values for each relevant attribute from buyer_data.csv.
  3. Enrich art_data.csv with those attributes by calling Ollama (one call per artwork).
  4. Compute a match score for (artwork, buyer) pairs and output recommendations.
The script uses the /v1/completions endpoint (via a Flask wrapper) since your
Ollama instance is running in completions mode.
Usage:
  python recommendation_pipeline.py
Requirements:
  pip install requests json5 rich
Author: Tanmay Goel (Adapted)
Date: 2025-02-16
"""

import time
import os
import csv
import requests
import json5  # More forgiving JSON parser
import functools
import logging
from rich.logging import RichHandler
from typing import List, Dict, Set

##############################################################################
# 1. CONFIGURATION
##############################################################################
# Set up Rich logging for structured, colorful output
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="[%H:%M:%S]",
    handlers=[RichHandler()]
)
logger = logging.getLogger("recommendation_pipeline")

# File paths and parameters
ART_DATA_FILE = "art_data.csv"
BUYER_DATA_FILE = "buyer_250.csv"
ENRICHED_ART_DATA_FILE = "enriched_art_data.csv"
RECOMMENDATIONS_FILE   = "recommendations.csv"
TOP_K = 5                      # Number of top matches per artwork
CALL_COUNT = 0                 # Count of calls to Ollama

# Define the model and API endpoint for Ollama (using the Flask wrapper)
MODEL_NAME = "llama2"
OLLAMA_SERVER_URL = "http://172.24.16.73:8001/infer"

##############################################################################
# 1a. HELPER FUNCTION TO EXTRACT JSON FROM RESPONSE TEXT
##############################################################################
def extract_json_from_text(text: str) -> str:
    """
    Extracts the JSON portion from a text string by identifying the first '{'
    and the last '}'. If found, returns the substring; otherwise, returns the
    original text.
    """
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and end > start:
        return text[start:end+1]
    return text

##############################################################################
# 2. LOCAL OLLAMA HELPER FUNCTION
##############################################################################
@functools.lru_cache(maxsize=100)
def call_ollama(prompt: str) -> str:
    """
    Calls the local Ollama server's endpoint with the given prompt.
    Returns the generated text.
    Retries up to 3 times on failure.
    """
    global CALL_COUNT
    CALL_COUNT += 1
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "temperature": 0.7,
        "max_tokens": 512
    }
    headers = {"Content-Type": "application/json"}
    logger.debug("----- Ollama Call #%d -----", CALL_COUNT)
    logger.debug("Prompt length: %d chars", len(prompt))
    logger.debug("Prompt excerpt:\n%s...", prompt[:200])
    retry_attempts = 3
    for attempt in range(retry_attempts):
        try:
            response = requests.post(OLLAMA_SERVER_URL, json=payload, headers=headers, timeout=300)
            if response.status_code == 429:
                logger.warning("429 Rate limit encountered. Sleeping 2 seconds before retrying...")
                time.sleep(2)
                continue
            if response.status_code != 200:
                raise RuntimeError(
                    f"Ollama server call failed. Status: {response.status_code}\nResponse: {response.text}"
                )
            data = response.json()
            # Check for "generated_text" first, fallback to "choices" if necessary.
            if "generated_text" in data:
                text = data["generated_text"]
            elif "choices" in data and len(data["choices"]) > 0:
                text = data["choices"][0]["text"]
            else:
                raise RuntimeError("Unexpected response format: " + str(data))
            logger.debug("Raw JSON response: %s", data)
            logger.debug("Response excerpt:\n%s...", text[:200])
            time.sleep(5)
            return text.strip()
        except requests.exceptions.RequestException as e:
            logger.error("Attempt %d/%d failed with error: %s", attempt + 1, retry_attempts, e)
            time.sleep(2)
    raise RuntimeError("Ollama server call failed after multiple attempts.")

##############################################################################
# 3. STEP ONE: Identify Relevant Buyer Attributes
##############################################################################
def identify_relevant_buyer_attributes(buyer_columns: List[str]) -> List[str]:
    """
    Uses Ollama to determine which buyer data columns are relevant for matching artwork.
    """
    logger.info("Identifying relevant buyer attributes from columns: %s", buyer_columns)
    columns_str = "\n- ".join(buyer_columns)
    prompt = f"""
We have a buyer dataset with the following columns:
- {columns_str}
Which columns are relevant for matching an artwork 
(e.g., style, medium, motivation, etc.) from an artistic point of view?
IMPORTANT: Output valid JSON only, with the key "relevant_columns". 
No extra text or repeated prompt. For example:
{{
  "relevant_columns": ["Preferred Art Styles", "Favorite Mediums", "Buying Motivation"]
}}
"""
    response_text = call_ollama(prompt)
    json_text = extract_json_from_text(response_text)
    try:
        data = json5.loads(json_text)
        relevant = data.get("relevant_columns", [])
    except Exception as e:
        logger.error("JSON parsing error: %s", e)
        relevant = ["Preferred Art Styles", "Favorite Mediums", "Buying Motivation"]
    logger.info("Relevant columns determined: %s", relevant)
    return relevant

##############################################################################
# 4. STEP TWO: Gather Unique Values
##############################################################################
def gather_unique_values_for_attributes(
    buyer_data: List[Dict[str, str]],
    relevant_columns: List[str]
) -> Dict[str, Set[str]]:
    """
    Extracts unique values for each relevant column from the buyer data.
    """
    logger.info("Gathering unique values from %d buyer rows for columns: %s", len(buyer_data), relevant_columns)
    attribute_values = {attr: set() for attr in relevant_columns}
    for row in buyer_data:
        for attr in relevant_columns:
            raw_value = row.get(attr, "")
            items = [x.strip() for x in raw_value.split(",") if x.strip()]
            attribute_values[attr].update(items)
    for attr in relevant_columns:
        vals = list(attribute_values[attr])[:10]
        logger.debug("Unique values for %s (partial): %s", attr, vals)
    return attribute_values

##############################################################################
# 5. STEP THREE: Enrich Artwork Data
##############################################################################
def enrich_artwork_data(
    art_data: List[Dict[str, str]],
    attribute_values: Dict[str, Set[str]],
    relevant_columns: List[str]
) -> List[Dict[str, str]]:
    """
    Labels each artwork with buyer attribute values using Ollama.
    """
    logger.info("Enriching artwork data for %d artworks.", len(art_data))
    enriched_data = []
    for idx, row in enumerate(art_data, start=1):
        artwork_id = row["Artwork ID"]
        description = row["Description"]
        possible_dict = {attr: list(attribute_values[attr]) for attr in relevant_columns}
        prompt = f"""
Below is an artwork description:
\"\"\"{description}\"\"\"
We have these attributes to label: {', '.join(relevant_columns)}.
Below is the possible set of values for each attribute in JSON:
{json5.dumps(possible_dict, indent=2)}
IMPORTANT: Output valid JSON only. No extra text or repeated prompt.
For example:
{{
  "Preferred Art Styles": ["Abstract", "Realist Figurative"],
  "Favorite Mediums": ["Oil"],
  "Buying Motivation": ["Support Artists"]
}}
"""
        logger.info("Processing Artwork #%d (ID: %s)", idx, artwork_id)
        logger.debug("Artwork description (partial): %s...", description[:150])
        response_text = call_ollama(prompt)
        json_text = extract_json_from_text(response_text)
        new_row = dict(row)
        try:
            data = json5.loads(json_text)
            logger.debug("JSON response for Artwork ID=%s: %s", artwork_id, data)
            for attr in relevant_columns:
                chosen_vals = data.get(attr, [])
                new_row[attr] = ", ".join(chosen_vals)
        except Exception as e:
            logger.error("JSON parsing error for Artwork ID=%s: %s", artwork_id, e)
            for attr in relevant_columns:
                new_row[attr] = ""
        enriched_data.append(new_row)
    return enriched_data

##############################################################################
# 6. STEP FOUR: Compute Match Score & Generate Recommendations
##############################################################################
def compute_match_score(
    artwork_attrs: Dict[str, str],
    buyer_attrs: Dict[str, str],
    relevant_columns: List[str]
) -> float:
    """
    Computes a simple Jaccard similarity score between artwork and buyer attributes.
    """
    total_score = 0.0
    for attr in relevant_columns:
        art_values = {v.strip() for v in artwork_attrs.get(attr, "").split(",") if v.strip()}
        buyer_values = {v.strip() for v in buyer_attrs.get(attr, "").split(",") if v.strip()}
        if not art_values and not buyer_values:
            continue
        overlap = art_values.intersection(buyer_values)
        union = art_values.union(buyer_values)
        score_attr = (len(overlap) / len(union)) if union else 0.0
        total_score += score_attr
    return (total_score / len(relevant_columns)) if relevant_columns else 0.0

def generate_recommendations(
    enriched_art_data: List[Dict[str, str]],
    buyer_data: List[Dict[str, str]],
    relevant_columns: List[str],
    top_k: int = 5
) -> List[Dict[str, str]]:
    """
    Computes match scores for each artwork against all buyers and selects the top matches.
    """
    recommendations = []
    for art in enriched_art_data:
        artwork_id = art["Artwork ID"]
        scores = []
        for buyer in buyer_data:
            buyer_id = buyer["Buyer ID"]
            score = compute_match_score(art, buyer, relevant_columns)
            scores.append((buyer_id, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        top_matches = scores[:top_k]
        for (b_id, sc) in top_matches:
            recommendations.append({
                "Artwork ID": artwork_id,
                "Buyer ID": b_id,
                "Match Score": f"{sc:.3f}"
            })
    return recommendations

##############################################################################
# 7. MAIN LOGIC
##############################################################################
def main():
    # 1) Read buyer_data.csv
    with open(BUYER_DATA_FILE, "r", encoding="utf-8") as f:
        buyer_rows = list(csv.DictReader(f))
    buyer_columns = list(buyer_rows[0].keys())
    logger.info("Buyer columns found: %s", buyer_columns)
    logger.info("Number of buyer rows: %d", len(buyer_rows))
    
    # 2) Identify relevant buyer attributes using Ollama
    relevant_attrs = identify_relevant_buyer_attributes(buyer_columns)
    logger.info("Relevant buyer attributes: %s", relevant_attrs)
    
    # 3) Gather unique attribute values from buyer data
    attr_values_dict = gather_unique_values_for_attributes(buyer_rows, relevant_attrs)
    logger.info("Unique attribute values:")
    for k, v in attr_values_dict.items():
        logger.info("  %s: %s", k, list(v)[:10])
    
    # 4) Read art_data.csv
    with open(ART_DATA_FILE, "r", encoding="utf-8") as f:
        art_rows = list(csv.DictReader(f))
    logger.info("Number of artwork rows: %d", len(art_rows))
    for idx, art in enumerate(art_rows, start=1):
        logger.info("Artwork #%d -> ID: %s, Description: %s...", idx, art["Artwork ID"], art["Description"][:100])
    
    # 5) Enrich artwork data using Ollama calls
    logger.info("Enriching artwork data with relevant attributes via Ollama...")
    enriched_art = enrich_artwork_data(art_rows, attr_values_dict, relevant_attrs)
    with open(ENRICHED_ART_DATA_FILE, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(enriched_art[0].keys()) if enriched_art else ["Artwork ID", "Description"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(enriched_art)
    logger.info("Saved enriched artwork data to %s.", ENRICHED_ART_DATA_FILE)
    
    # 6) Compute match scores and generate recommendations
    logger.info("Computing match scores and generating recommendations...")
    recommendations = generate_recommendations(enriched_art, buyer_rows, relevant_attrs, TOP_K)
    with open(RECOMMENDATIONS_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Artwork ID", "Buyer ID", "Match Score"])
        writer.writeheader()
        writer.writerows(recommendations)
    logger.info("Saved recommendations to %s.", RECOMMENDATIONS_FILE)
    logger.info("Process complete. Total calls to Ollama: %d", CALL_COUNT)

if __name__ == "__main__":
    main()
