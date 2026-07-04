"""Quick API connectivity test for Gemini and Tavily"""
import os
import sys

# Set keys from environment (passed via command line)
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "")

print("=" * 50)
print("API CONNECTIVITY TEST")
print("=" * 50)

# Test 1: Google Gemini
print("\n[1/3] Testing Google Gemini API...")
try:
    import google.generativeai as genai
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content("Say 'Hello, API test successful!' in one line.")
    print(f"  Result: {response.text.strip()}")
    print("  Status: PASS")
except Exception as e:
    print(f"  Error: {e}")
    print("  Status: FAIL")

# Test 2: Google Embeddings
print("\n[2/3] Testing Google Embeddings API...")
try:
    result = genai.embed_content(
        model="models/text-embedding-004",
        content="Test embedding for automotive analytics"
    )
    vec = result['embedding']
    print(f"  Embedding dimension: {len(vec)}")
    print(f"  First 5 values: {vec[:5]}")
    print("  Status: PASS")
except Exception as e:
    print(f"  Error: {e}")
    print("  Status: FAIL")

# Test 3: Tavily Search
print("\n[3/3] Testing Tavily Search API...")
try:
    from tavily import TavilyClient
    client = TavilyClient(api_key=TAVILY_API_KEY)
    results = client.search("Indian automotive industry market size 2025", max_results=2)
    answer = results.get('answer', 'N/A')
    num_results = len(results.get('results', []))
    print(f"  Answer preview: {answer[:150]}...")
    print(f"  Results returned: {num_results}")
    print("  Status: PASS")
except Exception as e:
    print(f"  Error: {e}")
    print("  Status: FAIL")

print("\n" + "=" * 50)
print("ALL TESTS COMPLETE")
print("=" * 50)
