"""
Probe the Token Company API to discover response schema.
Run: python baselines/probe_tc.py
"""
import os
import requests
from dotenv import load_dotenv
load_dotenv()

url = os.environ.get("TOKEN_COMPANY_URL", "")
key = os.environ.get("TOKEN_COMPANY_API_KEY", "")

if not url or "your_" in url:
    print("ERROR: Set TOKEN_COMPANY_URL in .env first")
    exit(1)
if not key or "your_" in key:
    print("ERROR: Set TOKEN_COMPANY_API_KEY in .env first")
    exit(1)

print(f"URL: {url}")
print(f"Key: {key[:8]}...")

resp = requests.post(
    url,
    headers={"Authorization": f"Bearer {key}"},
    json={
        "context": "Alice founded Tech Corp in 2010. She raised 50 million dollars from Sequoia Capital. Bob joined as CTO in 2012.",
        "question": "Who founded Tech Corp?"
    },
    timeout=30,
)

print(f"\nStatus: {resp.status_code}")
print(f"Headers: {dict(resp.headers)}")
print(f"\nResponse body:")
try:
    import json
    body = resp.json()
    print(json.dumps(body, indent=2))
    print(f"\nKeys: {list(body.keys())}")
except Exception:
    print(resp.text[:500])
