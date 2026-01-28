"""
Proxy MCP Server that forwards search and fetch requests to biomcp.

This server acts as a bridge between clients and biomcp, using an LLM to
translate queries into appropriate biomcp tool calls.
"""
import asyncio
import json
import os
import re
from typing import Any

from dotenv import load_dotenv
from fastmcp import FastMCP
from openai import AsyncOpenAI
import httpx

load_dotenv()

# Configuration
BIOMCP_URL = os.environ.get("BIOMCP_URL", "http://localhost:3319/mcp")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

# Initialize FastMCP server
mcp = FastMCP(
    name="BioMCP Proxy",
    instructions="A proxy server that provides unified search and fetch interface for biomcp.",
)

# OpenAI client
openai_client = AsyncOpenAI()


class BioMCPClient:
    """Client for communicating with biomcp server via MCP protocol."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.http_client = httpx.AsyncClient(timeout=60.0)
        self._tools_cache: list[dict] | None = None

    async def close(self):
        await self.http_client.aclose()

    async def list_tools(self) -> list[dict]:
        """List available tools from biomcp."""
        if self._tools_cache is not None:
            return self._tools_cache

        response = await self.http_client.post(
            self.base_url,
            json={"jsonrpc": "2.0", "method": "tools/list", "id": 1},
            headers={"Content-Type": "application/json"},
        )
        result = response.json()
        self._tools_cache = result.get("result", {}).get("tools", [])
        return self._tools_cache

    async def call_tool(self, name: str, arguments: dict) -> Any:
        """Call a tool on biomcp."""
        response = await self.http_client.post(
            self.base_url,
            json={
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {"name": name, "arguments": arguments},
                "id": 1,
            },
            headers={"Content-Type": "application/json"},
        )
        result = response.json()
        if "error" in result:
            raise Exception(f"Tool call failed: {result['error']}")
        return result.get("result", {})


# Global biomcp client
biomcp_client = BioMCPClient(BIOMCP_URL)


def convert_tools_to_openai_format(tools: list[dict]) -> list[dict]:
    """Convert MCP tools to OpenAI function calling format."""
    openai_tools = []
    for tool in tools:
        openai_tools.append({
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("inputSchema", {"type": "object", "properties": {}}),
            },
        })
    return openai_tools


def extract_domain_from_tool_name(tool_name: str) -> str:
    """Extract domain from tool name (e.g., 'article_search' -> 'article')."""
    # Common patterns: article_search, trial_search, etc.
    if "_" in tool_name:
        return tool_name.split("_")[0]
    return tool_name


def wrap_id_with_domain(domain: str, original_id: str) -> str:
    """Wrap an ID with domain prefix using @@@ separator."""
    return f"{domain}@@@{original_id}"


def parse_wrapped_id(wrapped_id: str) -> tuple[str, str]:
    """Parse a wrapped ID into domain and original ID."""
    if "@@@" in wrapped_id:
        parts = wrapped_id.split("@@@", 1)
        return parts[0], parts[1]
    # Fallback: assume it's an article ID
    return "article", wrapped_id


def extract_content_from_mcp_result(result: dict) -> str:
    """Extract text content from MCP tool result."""
    content = result.get("content", [])
    texts = []
    for item in content:
        if isinstance(item, dict) and item.get("type") == "text":
            texts.append(item.get("text", ""))
        elif isinstance(item, str):
            texts.append(item)
    return "\n".join(texts)


async def run_llm_agent_for_search(query: str) -> list[dict]:
    """
    Use LLM to search biomcp and return formatted results.

    Returns list of results with id, title, text snippet, and optional url.
    """
    tools = await biomcp_client.list_tools()
    openai_tools = convert_tools_to_openai_format(tools)

    # System prompt for the search agent
    system_prompt = """You are a search assistant that helps find biomedical information.
Your task is to search for relevant information based on the user's query.

When you receive a search query:
1. Identify which search tools are most relevant (article, trial, variant, etc.)
2. Call the appropriate search tool(s) with relevant parameters
3. Return the search results

Available tool categories:
- article_* : For searching scientific articles/papers
- trial_* : For searching clinical trials
- variant_* : For searching genetic variants

Call the most appropriate search tool based on the query."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Search for: {query}"},
    ]

    all_results = []
    max_iterations = 5

    for _ in range(max_iterations):
        response = await openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            tools=openai_tools,
            tool_choice="auto",
        )

        message = response.choices[0].message

        if message.tool_calls:
            messages.append(message)

            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                try:
                    arguments = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    arguments = {}

                # Call biomcp tool
                try:
                    result = await biomcp_client.call_tool(tool_name, arguments)
                    content = extract_content_from_mcp_result(result)

                    # Parse results and add to all_results
                    domain = extract_domain_from_tool_name(tool_name)
                    parsed = parse_search_results(content, domain)
                    all_results.extend(parsed)

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": content[:2000],  # Truncate for context
                    })
                except Exception as e:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": f"Error: {str(e)}",
                    })
        else:
            # No more tool calls, done
            break

    return all_results


def parse_search_results(content: str, domain: str) -> list[dict]:
    """
    Parse search results from biomcp content and format them.

    This is a heuristic parser that tries to extract structured results.
    """
    results = []

    # Try to parse as JSON first
    try:
        data = json.loads(content)
        if isinstance(data, list):
            for item in data:
                result = format_result_item(item, domain)
                if result:
                    results.append(result)
            return results
        elif isinstance(data, dict):
            # Single result or wrapped results
            if "results" in data:
                for item in data["results"]:
                    result = format_result_item(item, domain)
                    if result:
                        results.append(result)
                return results
            else:
                result = format_result_item(data, domain)
                if result:
                    results.append(result)
                return results
    except json.JSONDecodeError:
        pass

    # Heuristic parsing for text content
    # Look for common patterns like IDs, titles, etc.
    lines = content.split("\n")

    current_item = {}
    for line in lines:
        line = line.strip()
        if not line:
            if current_item:
                result = format_result_item(current_item, domain)
                if result:
                    results.append(result)
                current_item = {}
            continue

        # Try to extract key-value pairs
        if ":" in line:
            key, _, value = line.partition(":")
            key = key.strip().lower()
            value = value.strip()

            if key in ("id", "pmid", "nct", "nct_id", "trial_id", "article_id"):
                current_item["id"] = value
            elif key in ("title", "name", "brief_title", "official_title"):
                current_item["title"] = value
            elif key in ("abstract", "description", "summary", "text"):
                current_item["text"] = value
            elif key in ("url", "link"):
                current_item["url"] = value

        # Look for PMID patterns
        pmid_match = re.search(r"PMID[:\s]*(\d+)", line, re.IGNORECASE)
        if pmid_match and "id" not in current_item:
            current_item["id"] = pmid_match.group(1)

        # Look for NCT patterns
        nct_match = re.search(r"(NCT\d+)", line, re.IGNORECASE)
        if nct_match and "id" not in current_item:
            current_item["id"] = nct_match.group(1)

    # Don't forget the last item
    if current_item:
        result = format_result_item(current_item, domain)
        if result:
            results.append(result)

    return results


def format_result_item(item: dict, domain: str) -> dict | None:
    """Format a single result item with wrapped ID."""
    if not isinstance(item, dict):
        return None

    # Try to find an ID
    original_id = None
    for key in ("id", "pmid", "nct_id", "trial_id", "article_id", "nct", "PMID"):
        if key in item:
            original_id = str(item[key])
            break

    if not original_id:
        return None

    # Try to find a title
    title = None
    for key in ("title", "name", "brief_title", "official_title", "Title"):
        if key in item:
            title = item[key]
            break

    title = title or f"{domain} result"

    # Try to find text/snippet
    text = None
    for key in ("abstract", "description", "summary", "text", "snippet", "Abstract"):
        if key in item:
            text = item[key]
            break

    text = text or ""
    # Truncate text to snippet (200 chars)
    if len(text) > 200:
        text = text[:197] + "..."

    # Try to find URL
    url = item.get("url") or item.get("link")

    wrapped_id = wrap_id_with_domain(domain, original_id)

    result = {
        "id": wrapped_id,
        "title": title,
        "text": text,
    }
    if url:
        result["url"] = url

    return result


async def run_fetch_for_domain(domain: str, original_id: str) -> dict:
    """
    Fetch full content for a specific domain and ID.

    This directly calls the appropriate biomcp tool without LLM.
    """
    tools = await biomcp_client.list_tools()

    # Map domain to fetch tool
    fetch_tool_mappings = {
        "article": ["article_get_details", "article_fetch", "article_details", "get_article"],
        "trial": ["trial_get_details", "trial_fetch", "get_trial", "trial_details"],
        "clinicaltrial": ["trial_get_details", "trial_fetch", "get_trial", "trial_details"],
        "variant": ["variant_get_details", "variant_fetch", "get_variant", "variant_details"],
    }

    # Find matching tool
    tool_names = [t["name"] for t in tools]
    fetch_tool = None

    if domain in fetch_tool_mappings:
        for candidate in fetch_tool_mappings[domain]:
            if candidate in tool_names:
                fetch_tool = candidate
                break

    # Fallback: look for any tool with domain prefix that looks like a fetch/get tool
    if not fetch_tool:
        for tool_name in tool_names:
            if tool_name.startswith(domain) and any(x in tool_name for x in ("get", "fetch", "detail")):
                fetch_tool = tool_name
                break

    if not fetch_tool:
        # Use LLM as fallback
        return await run_llm_agent_for_fetch(domain, original_id)

    # Determine the argument name for ID
    tool_def = next((t for t in tools if t["name"] == fetch_tool), None)
    id_param_name = "id"

    if tool_def:
        schema = tool_def.get("inputSchema", {})
        properties = schema.get("properties", {})
        for param_name in properties:
            if "id" in param_name.lower():
                id_param_name = param_name
                break

    # Call the fetch tool
    try:
        result = await biomcp_client.call_tool(fetch_tool, {id_param_name: original_id})
        content = extract_content_from_mcp_result(result)

        # Parse and return
        return parse_fetch_result(content, domain, original_id)
    except Exception as e:
        raise ValueError(f"Failed to fetch {domain}/{original_id}: {str(e)}")


async def run_llm_agent_for_fetch(domain: str, original_id: str) -> dict:
    """Fallback: use LLM to fetch content when direct mapping fails."""
    tools = await biomcp_client.list_tools()
    openai_tools = convert_tools_to_openai_format(tools)

    system_prompt = f"""You are a fetch assistant. Your task is to retrieve the full details for a specific item.

Domain: {domain}
ID: {original_id}

Call the appropriate tool to fetch the complete details for this item."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Fetch full details for {domain} ID: {original_id}"},
    ]

    response = await openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        tools=openai_tools,
        tool_choice="auto",
    )

    message = response.choices[0].message

    if message.tool_calls:
        tool_call = message.tool_calls[0]
        tool_name = tool_call.function.name
        try:
            arguments = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError:
            arguments = {}

        result = await biomcp_client.call_tool(tool_name, arguments)
        content = extract_content_from_mcp_result(result)
        return parse_fetch_result(content, domain, original_id)

    raise ValueError(f"Could not fetch {domain}/{original_id}")


def parse_fetch_result(content: str, domain: str, original_id: str) -> dict:
    """Parse fetch result and return formatted response."""
    wrapped_id = wrap_id_with_domain(domain, original_id)

    # Try JSON first
    try:
        data = json.loads(content)
        if isinstance(data, dict):
            title = None
            for key in ("title", "name", "brief_title", "official_title", "Title"):
                if key in data:
                    title = data[key]
                    break

            text = None
            for key in ("abstract", "description", "summary", "text", "content", "Abstract"):
                if key in data:
                    text = data[key]
                    break

            return {
                "id": wrapped_id,
                "title": title or f"{domain} {original_id}",
                "text": text or json.dumps(data, indent=2),
                "url": data.get("url") or data.get("link"),
                "metadata": data,
            }
    except json.JSONDecodeError:
        pass

    # Return raw content
    return {
        "id": wrapped_id,
        "title": f"{domain} {original_id}",
        "text": content,
    }


@mcp.tool()
async def search(query: str) -> dict[str, list[dict[str, Any]]]:
    """
    Search biomedical databases using natural language query.

    Searches across multiple biomedical sources including scientific articles,
    clinical trials, and genetic variants.

    Args:
        query: Natural language search query. Works best with specific terms.

    Returns:
        Dictionary with 'results' key containing list of matching documents.
        Each result includes id, title, text snippet, and optional URL.
        IDs are prefixed with domain (e.g., 'article@@@12345', 'trial@@@NCT123').
    """
    try:
        results = await run_llm_agent_for_search(query)
        return {"results": results}
    except Exception as e:
        return {"results": [], "error": str(e)}


@mcp.tool()
async def fetch(id: str) -> dict[str, Any]:
    """
    Fetch complete document content by ID.

    Retrieves the full text and metadata for a document previously found via search.

    Args:
        id: Document ID in format 'domain@@@original_id' (e.g., 'article@@@12345').
            Use IDs returned from the search function.

    Returns:
        Dictionary containing id, title, full text content, and optional metadata.

    Raises:
        ValueError: If the specified ID is not found.
    """
    domain, original_id = parse_wrapped_id(id)
    return await run_fetch_for_domain(domain, original_id)


@mcp.tool()
async def list_domains() -> dict[str, list[str]]:
    """
    List available search domains from biomcp.

    Returns:
        Dictionary with 'domains' key containing list of available domain names.
    """
    try:
        tools = await biomcp_client.list_tools()
        domains = set()
        for tool in tools:
            name = tool["name"]
            if "_" in name:
                domain = name.split("_")[0]
                domains.add(domain)
        return {"domains": sorted(domains)}
    except Exception as e:
        return {"domains": [], "error": str(e)}


if __name__ == "__main__":
    mcp.run(
        transport="sse", 
        host="127.0.0.1", 
        port=3017
    )
