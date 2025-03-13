#!/usr/bin/env python3
import requests
import json

# URL for the Flask server on your virtual server.
# Update the IP if needed.
OLLAMA_FLASK_URL = "http://172.24.16.73:8001/infer"

def test_ollama():
    payload = {
        "prompt": "Hello, please generate a test response.",
        "max_tokens": 400,
        "temperature": 0.7,
        "model": "llama2"
    }
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.post(OLLAMA_FLASK_URL, json=payload, headers=headers, timeout=30)
        if response.status_code == 200:
            data = response.json()
            generated_text = data.get("generated_text", "No generated text in response")
            print("Generated Text:")
            print(generated_text)
        else:
            print(f"Request failed with status code {response.status_code}")
            print("Response:", response.text)
    except Exception as e:
        print("Error occurred:", e)

if __name__ == "__main__":
    test_ollama()
