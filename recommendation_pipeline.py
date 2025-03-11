#!/usr/bin/env python3
"""
End-to-End Artwork-Buyer Matching Using a Local Ollama Server (Completions Mode)
----------------------------------------------------------------------------------
This script demonstrates how to:
  1. Identify relevant buyer attributes for art matching using a local Ollama server.
  2. Gather unique values for each relevant attribute from buyer_data.csv.
  3. Enrich art_data.csv with those attributes by calling Ollama (one call per artwork).
  4. Compute a match score for (artwork, buyer) pairs and output recommendations.
The script uses the /v1/completions endpoint (rather than /v1/chat) since your
Ollama instance is running in completions mode.
Usage:
  python recommendation_pipeline.py
Requirements:
  pip install requests json5
Author: Tanmay Goel (Adapted)
Date: 2025-02-16
"""
import time
import os
import csv
import requests
import json5  # More forgiving JSON parser
import functools
from typing import List, Dict, Set
##############################################################################
# 1. CONFIGURATION
##############################################################################
DEBUG = True                   # Toggle debug prints
ART_DATA_FILE = "art_data.csv"
BUYER_DATA_FILE = "sample_250_buyer_data_updated.csv"
ENRICHED_ART_DATA_FILE = "enriched_art_data.csv"
RECOMMENDATIONS_FILE   = "recommendations.csv"
TOP_K = 5                      # Number of top matches per artwork
CALL_COUNT = 0                 # Count how many times we call Ollama
# Define the model and API endpoint for Ollama (completions endpoint)
MODEL_NAME = "llama2"
OLLAMA_SERVER_URL = "http://172.24.16.73:11434/v1/completions"
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
    Calls the local Ollama server's /v1/completions endpoint with the given prompt.
    Returns the raw text output.
    Retries up to 3 times on failure.
    """
    global CALL_COUNT
    CALL_COUNT += 1
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "temperature": 0.7,
        "max_tokens": 128
    }
    headers = {"Content-Type": "application/json"}
    if DEBUG:
        print(f"\n[DEBUG] ----- Ollama Call #{CALL_COUNT} -----")
        print(f"[DEBUG] Prompt length: {len(prompt)} chars")
        print(f"[DEBUG] Prompt excerpt:\n{prompt[:200]}...\n")
    retry_attempts = 3
    for attempt in range(retry_attempts):
        try:
            response = requests.post(OLLAMA_SERVER_URL, json=payload, headers=headers, timeout=300)
            if response.status_code == 429:
                if DEBUG:
                    print("[DEBUG] 429 Rate limit encountered. Sleeping 2 seconds before retrying...")
                time.sleep(2)
                continue
            if response.status_code != 200:
                raise RuntimeError(
                    f"Ollama server call failed. Status: {response.status_code}\nResponse: {response.text}"
                )
            data = response.json()
            # Expected response format:
            # {
            #   "id": "...",
            #   "object": "text_completion",
            #   "choices": [{"text": "generated output here", ...}],
            #   ...
            # }
            text = data["choices"][0]["text"]
            if DEBUG:
                print(f"[DEBUG] Raw JSON response: {data}")
                print(f"[DEBUG] Response excerpt:\n{text[:200]}...\n")
            time.sleep(5)
            return text.strip()
        except requests.exceptions.RequestException as e:
            if DEBUG:
                print(f"[DEBUG] Attempt {attempt + 1}/{retry_attempts} failed with error: {e}")
            time.sleep(2)
    raise RuntimeError("Ollama server call failed after multiple attempts.")
##############################################################################
# 3. STEP ONE: Identify Relevant Buyer Attributes
##############################################################################
def identify_relevant_buyer_attributes(buyer_columns: List[str]) -> List[str]:
    """
    Given a list of buyer data columns, uses Ollama to determine which columns
    are relevant for matching artwork (e.g., style, medium, motivation).
    """
    if DEBUG:
        print("\n[DEBUG] identify_relevant_buyer_attributes()")
        print(f"[DEBUG] Buyer columns: {buyer_columns}")
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
    # Extract JSON portion from the response
    json_text = extract_json_from_text(response_text)
    try:
        data = json5.loads(json_text)
        relevant = data.get("relevant_columns", [])
    except Exception as e:
        if DEBUG:
            print(f"[DEBUG] JSON parsing error: {e}")
        # Fallback default if parsing fails.
        relevant = ["Preferred Art Styles", "Favorite Mediums", "Buying Motivation"]
    if DEBUG:
        print(f"[DEBUG] relevant columns from Ollama: {relevant}")
    return relevant
##############################################################################
# 4. STEP TWO: Gather Unique Values
##############################################################################
def gather_unique_values_for_attributes(
    buyer_data: List[Dict[str, str]],
    relevant_columns: List[str]
) -> Dict[str, Set[str]]:
    """
    Given buyer data and a list of relevant columns, extracts all unique values for each column.
    """
    if DEBUG:
        print("\n[DEBUG] gather_unique_values_for_attributes()")
        print(f"[DEBUG] # Buyer rows: {len(buyer_data)}")
        print(f"[DEBUG] Relevant columns: {relevant_columns}")
    attribute_values = {attr: set() for attr in relevant_columns}
    for row in buyer_data:
        for attr in relevant_columns:
            raw_value = row.get(attr, "")
            # Assume comma-separated values; strip whitespace.
            items = [x.strip() for x in raw_value.split(",") if x.strip()]
            attribute_values[attr].update(items)
    if DEBUG:
        for attr in relevant_columns:
            vals = list(attribute_values[attr])[:10]
            print(f"[DEBUG] Unique values for {attr} (partial): {vals}")
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
    For each artwork, uses Ollama to label the artwork with the relevant buyer attribute values.
    """
    if DEBUG:
        print("\n[DEBUG] enrich_artwork_data()")
        print(f"[DEBUG] # Artwork rows: {len(art_data)}")
        print(f"[DEBUG] relevant_columns: {relevant_columns}")
    enriched_data = []
    for idx, row in enumerate(art_data, start=1):
        artwork_id = row["Artwork ID"]
        description = row["Description"]
        # Prepare a dictionary of possible values for each attribute.
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
        if DEBUG:
            print(f"\n[DEBUG] Artwork #{idx}, ID = {artwork_id}")
            print(f"[DEBUG] Artwork description (partial): {description[:150]}...")
        response_text = call_ollama(prompt)
        # Extract JSON portion from the response
        json_text = extract_json_from_text(response_text)
        new_row = dict(row)
        try:
            data = json5.loads(json_text)
            if DEBUG:
                print(f"[DEBUG] JSON response from Ollama for Artwork ID={artwork_id}: {data}")
            # For each attribute, store the chosen values as a comma-separated string.
            for attr in relevant_columns:
                chosen_vals = data.get(attr, [])
                new_row[attr] = ", ".join(chosen_vals)
        except Exception as e:
            if DEBUG:
                print(f"[DEBUG] JSON parsing error for Artwork ID={artwork_id}: {e}")
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
    Computes a simple Jaccard similarity between artwork and buyer attribute sets.
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
    For each artwork, computes match scores against all buyers and selects the top_k matches.
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
    print("[DEBUG] Buyer columns found in CSV:", buyer_columns)
    print(f"[DEBUG] # Buyer rows: {len(buyer_rows)}")
    # 2) Identify relevant buyer attributes using Ollama
    relevant_attrs = identify_relevant_buyer_attributes(buyer_columns)
    print("[DEBUG] Relevant buyer attributes (from Ollama):", relevant_attrs)
    # 3) Gather unique attribute values from buyer data
    attr_values_dict = gather_unique_values_for_attributes(buyer_rows, relevant_attrs)
    print("[DEBUG] Unique values for each relevant attribute:")
    for k, v in attr_values_dict.items():
        print(f"  {k}: {v}")
    # 4) Read art_data.csv
    with open(ART_DATA_FILE, "r", encoding="utf-8") as f:
        art_rows = list(csv.DictReader(f))
    print(f"[DEBUG] # Artwork rows: {len(art_rows)}")
    for idx, art in enumerate(art_rows, start=1):
        print(f"[DEBUG] Artwork #{idx} -> ID: {art['Artwork ID']}, Desc partial: {art['Description'][:100]}...")
    # 5) Enrich artwork data using Ollama calls
    print("[DEBUG] Enriching art_data.csv with relevant attributes using Ollama calls...")
    enriched_art = enrich_artwork_data(art_rows, attr_values_dict, relevant_attrs)
    # Save enriched art data to CSV
    if enriched_art:
        all_cols = list(enriched_art[0].keys())
    else:
        all_cols = ["Artwork ID", "Description"]
    with open(ENRICHED_ART_DATA_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_cols)
        writer.writeheader()
        writer.writerows(enriched_art)
    print(f"[DEBUG] Saved enriched artwork data to {ENRICHED_ART_DATA_FILE}.")
    # 6) Compute match scores & generate recommendations
    print("[DEBUG] Computing match scores and generating final recommendations...")
    recommendations = generate_recommendations(enriched_art, buyer_rows, relevant_attrs, TOP_K)
    rec_cols = ["Artwork ID", "Buyer ID", "Match Score"]
    with open(RECOMMENDATIONS_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rec_cols)
        writer.writeheader()
        writer.writerows(recommendations)
    print(f"[DEBUG] Saved final recommendations to {RECOMMENDATIONS_FILE}.")
    print("[DEBUG] Done!")
    print("[DEBUG] Total calls to Ollama:", CALL_COUNT)
if __name__ == "__main__":
    main()

