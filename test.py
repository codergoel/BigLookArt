#!/usr/bin/env python3
"""
test_mistral_remote.py
----------------------
A minimal script to test Mistral's response for a single prompt from your local machine.
It sends a prompt instructing the LLM to return valid JSON only.
"""

import requests
import json
import time

def call_mistral_remote(prompt: str) -> str:
    """
    Calls a remote Mistral 7B server (running on GPU).
    Expects a JSON response: {"generated_text": "..."}.
    """
    MISTRAL_SERVER_URL = "http://172.24.16.73:8000/infer"  # <-- Replace with your server's IP or domain
    payload = {
        "prompt": prompt,
        "max_tokens": 128,
        "temperature": 0.8
    }

    try:
        resp = requests.post(MISTRAL_SERVER_URL, json=payload, timeout=300)
        resp.raise_for_status()
        data = resp.json()
        text = data.get("generated_text", "")
        # Optional: small delay to avoid spamming
        time.sleep(2)
        return text.strip()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Error calling Mistral server: {str(e)}")

def main():
    # Example prompt instructing Mistral to return valid JSON only.
    prompt = """
IMPORTANT: Return valid JSON only. No extra text or repeated prompt.

For example:
{
  "example_key": ["SomeValue", "AnotherValue"]
}

Now, specifically, create a JSON object with a single key "test_columns"
that has a list of 3 strings: "One", "Two", and "Three".
"""

    print("Sending prompt to Mistral server...\n")
    response_text = call_mistral_remote(prompt)

    print("Raw response from Mistral:\n")
    print(response_text)
    print("\nAttempting to parse as JSON...\n")
    try:
        parsed = json.loads(response_text)
        print("JSON parsed successfully:", parsed)
    except json.JSONDecodeError as e:
        print("Failed to parse JSON:", str(e))

if __name__ == "__main__":
    main()
