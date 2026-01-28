"""
Test script for the proxy MCP server.

Usage:
    uv run python test_proxy.py
"""
import asyncio
import os

from dotenv import load_dotenv

load_dotenv()

# Import the proxy server module
import proxy_server


async def main():
    print("=== Testing BioMCP Proxy Server ===\n")

    biomcp_url = os.environ.get("BIOMCP_URL", "http://localhost:3319/mcp")
    print(f"BioMCP URL: {biomcp_url}\n")

    # Test 1: List available domains
    print("1. Listing available domains...")
    try:
        result = await proxy_server.list_domains()
        print(f"   Available domains: {result}\n")
    except Exception as e:
        print(f"   Error: {e}\n")
        print("   (Make sure biomcp is running at the configured URL)\n")
        return

    # Test 2: Search
    print("2. Testing search...")
    query = "lung cancer treatment"
    print(f"   Query: {query}")
    try:
        result = await proxy_server.search(query)
        print(f"   Found {len(result.get('results', []))} results")
        for r in result.get("results", [])[:3]:
            print(f"   - [{r['id']}] {r['title'][:50]}...")
        print()
    except Exception as e:
        print(f"   Error: {e}\n")

    # Test 3: Fetch (if we have search results)
    if result.get("results"):
        print("3. Testing fetch...")
        first_id = result["results"][0]["id"]
        print(f"   Fetching ID: {first_id}")
        try:
            fetch_result = await proxy_server.fetch(first_id)
            print(f"   Title: {fetch_result.get('title', 'N/A')[:50]}...")
            text = fetch_result.get("text", "")
            print(f"   Text preview: {text[:100]}..." if len(text) > 100 else f"   Text: {text}")
        except Exception as e:
            print(f"   Error: {e}")

    print("\n=== Tests complete ===")

    # Cleanup
    await proxy_server.biomcp_client.close()


if __name__ == "__main__":
    asyncio.run(main())
