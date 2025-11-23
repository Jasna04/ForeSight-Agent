"""
Test to verify sources are properly flowing through the RAG analysis
"""

print("="*80)
print("TESTING SOURCES DATA FLOW")
print("="*80)

# Simulate rag_hits from SerpAPI
mock_rag_hits = [
    {
        "title": "Pump Bearing Maintenance Guide",
        "snippet": "Learn how to prevent bearing failure in industrial pumps...",
        "link": "https://example.com/bearing-guide"
    },
    {
        "title": "Vibration Analysis for Predictive Maintenance",
        "snippet": "Understanding vibration patterns in rotating equipment...",
        "link": "https://example.com/vibration"
    },
    {
        "title": "Temperature Monitoring Best Practices",
        "snippet": "Monitor critical temperature thresholds...",
        "link": "https://example.com/temperature"
    }
]

print(f"\n1. Mock rag_hits: {len(mock_rag_hits)} items")

# Simulate the source extraction logic from rag_analysis_tool
sources = []
top_sources = []

if mock_rag_hits:
    print(f"\n2. Processing {len(mock_rag_hits)} hits for sources")
    for i, h in enumerate(mock_rag_hits[:3]):
        try:
            link = h.get("link", "")
            if link:
                sources.append(link)
            
            top_sources.append({
                "title": h.get("title", "Untitled Source"),
                "snippet": h.get("snippet", ""),
                "link": link
            })
            print(f"   - Processed hit {i+1}: {h.get('title', 'No title')}")
        except Exception as e:
            print(f"   - Error processing hit {i}: {e}")
    print(f"\n3. Created {len(top_sources)} top_sources entries")
else:
    print("\n2. No rag_hits available - sources will be empty")

# Simulate the return structure
rag_result = {
    "status": "success",
    "insights": "Summary: Test\n\nRoot Cause: Test\n\nRecommended Actions:\n1. Test",
    "sources": sources,
    "llm_analysis": {
        "summary": "Test summary",
        "root_cause": "Test root cause",
        "recommended_actions": ["Action 1", "Action 2"],
        "evidence": [],
        "confidence": 0.7
    },
    "top_sources": top_sources,
    "timestamp": "2025-11-23T12:00:00"
}

print(f"\n4. RAG Result structure:")
print(f"   - status: {rag_result['status']}")
print(f"   - sources: {len(rag_result['sources'])} links")
print(f"   - top_sources: {len(rag_result['top_sources'])} detailed entries")

# Simulate storage in session state
rag_summary = {
    "summary": rag_result["llm_analysis"].get("summary"),
    "root_cause": rag_result["llm_analysis"].get("root_cause"),
    "recommended_actions": rag_result["llm_analysis"].get("recommended_actions", []),
    "evidence": rag_result["llm_analysis"].get("evidence", []),
    "confidence": rag_result["llm_analysis"].get("confidence", 0),
    "top_sources": rag_result.get("top_sources", [])
}

print(f"\n5. RAG Summary for storage:")
print(f"   - top_sources in summary: {len(rag_summary['top_sources'])} entries")

# Simulate display
top_sources_display = rag_summary.get("top_sources", [])
print(f"\n6. Display phase:")
print(f"   - Retrieved top_sources: {len(top_sources_display)} entries")

if top_sources_display:
    print("\n   📚 Top Sources:")
    for i, src in enumerate(top_sources_display, 1):
        title = src.get("title", "").strip()
        link = src.get("link", "").strip()
        snippet = src.get("snippet", "").strip()
        print(f"   {i}. [{title}]({link})")
        if snippet:
            print(f"      {snippet[:80]}...")
else:
    print("   ❌ NO SOURCES TO DISPLAY!")

print("\n" + "="*80)
print("✅ Test complete - verify all 6 steps show data flowing correctly")
print("="*80)
