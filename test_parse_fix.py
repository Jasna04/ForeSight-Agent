"""Test script to verify the parse_llm_analysis function correctly extracts Summary, Root Cause, and Recommended Actions"""

import re

def clean_source_markers(text):
    """Remove source citations like (Source 1), (Source 4), etc. from text"""
    if not text:
        return text
    # Remove patterns like (Source N) or (source N)
    cleaned = re.sub(r'\s*\(Source\s+\d+\)', '', text, flags=re.IGNORECASE)
    # Remove patterns like [Source N] or [source N]
    cleaned = re.sub(r'\s*\[Source\s+\d+\]', '', cleaned, flags=re.IGNORECASE)
    # Clean up extra whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    # Capitalize first letter
    if cleaned:
        cleaned = cleaned[0].upper() + cleaned[1:]
    return cleaned


def parse_llm_analysis(text):
    """Try to extract structured fields from a raw LLM text reply.
    Returns dict with keys: summary, root_cause, recommended_actions (list), evidence (list), confidence (0-1).
    Uses heuristics to prefer causal phrases for root cause and selects evidence sentences that mention sensors or failure modes.
    """
    if not text:
        return {"summary": "", "root_cause": "", "recommended_actions": [], "evidence": [], "confidence": 0.0}

    s = text.strip()
    
    # Check if this is an error message rather than actual LLM output
    if s.startswith("[") and s.endswith("]"):
        return {
            "summary": s,
            "root_cause": "LLM service not available",
            "recommended_actions": ["Configure API keys to enable AI-powered insights"],
            "evidence": [],
            "confidence": 0.0
        }

    # Normalize and strip empty lines
    lines = [l.strip() for l in re.split(r"\r?\n", s) if l.strip()]
    full = "\n".join(lines)

    # Initialize
    root = None
    actions = []

    # 1) Extract explicit 'Root' and 'Recommended' sections if present
    # Updated to better match section headers and avoid matching inline text
    m_root = re.search(r"(?i)(?:^|\n)\s*root\s*cause[:\-]?\s*(.+?)(?=(?:\n\s*(?:recommended\s*actions?|recommendations?|next\s*steps?)[:\-])|$)", full, re.S)
    if m_root:
        root = m_root.group(1).strip()

    m_actions = re.search(r"(?i)(?:^|\n)\s*(?:recommended\s*actions?|recommendations?|next\s*steps?)[:\-]?\s*(.+?)(?=(?:\n\s*(?:top\s*sources?|sources?|summary)[:\-])|$)", full, re.DOTALL)
    if m_actions:
        tail = m_actions.group(1).strip()
        # First try to split by numbered items (1., 2., etc.)
        numbered_items = re.split(r'\n\s*\d+\.\s+', tail)
        if len(numbered_items) > 1:
            # Clean up first item which might start with a number
            first_item = re.sub(r'^\d+\.\s+', '', numbered_items[0]).strip()
            # Remove first empty element if exists
            actions = [first_item] if first_item and len(first_item) > 3 else []
            actions.extend([it.strip() for it in numbered_items[1:] if it and len(it.strip()) > 3])
        else:
            # Try bullet points or dashes at start of line
            bullet_items = re.split(r'\n\s*[-*\u2022]\s+', tail)
            if len(bullet_items) > 1:
                # Clean up first item which might start with a bullet
                first_item = re.sub(r'^[-*\u2022]\s+', '', bullet_items[0]).strip()
                actions = [first_item] if first_item and len(first_item) > 3 else []
                actions.extend([it.strip() for it in bullet_items[1:] if it and len(it.strip()) > 3])
            else:
                # Split by newlines and filter meaningful lines
                line_items = [line.strip() for line in tail.split('\n') if line.strip()]
                # If it's a single paragraph, keep it as one action
                if len(line_items) == 1 or '\n\n' not in tail:
                    actions = [tail.strip()] if tail.strip() else []
                else:
                    actions = [it for it in line_items if len(it) > 10]

    # 2) If no explicit root, look for causal connectors or causal sentences
    if not root:
        m_cause = re.search(r"(?i)(because|due to|caused by|result of|as a result of)\s+([^\.]+\.?)", full)
        if m_cause:
            # Get the full sentence containing the cause
            root = m_cause.group(0).strip()
        else:
            # select sentences with failure-related keywords
            sents = re.split(r"(?<=[.!?])\s+", s)
            chosen_sents = []
            for sent in sents:
                if re.search(r"(?i)overheat|overheating|wear|corrod|leak|short circuit|vibration|imbalance|misalign|bearing|lubricat|temperature|pressure|humidity|power|current|voltage|fault|fail", sent):
                    chosen_sents.append(sent.strip())
                    if len(chosen_sents) >= 2:  # Get up to 2 relevant sentences
                        break
            
            if chosen_sents:
                root = " ".join(chosen_sents)
            elif len(sents) > 1:
                root = sents[1].strip()
            else:
                root = sents[0].strip() if sents else ""

    # 3) Evidence extraction: sentences that mention sensors, readings, or failure modes
    evidence = []
    evid_candidates = []
    all_sents = re.split(r"(?<=[.!?])\s+", s)
    for sent in all_sents:
        if re.search(r"(?i)temperature|vibration|pressure|humidity|power|current|voltage|bearing|lubricat|sensor|reading|rpm|Hz|°C|deg|leak|smoke|sparks|anomal", sent):
            evid_candidates.append(sent.strip())
    # Prefer sentences that contain numbers or units as stronger evidence
    def score_evidence(sent):
        score = 0
        if re.search(r"\d+", sent):
            score += 2
        if re.search(r"(?i)°C|deg|rpm|Hz|kW|V|A|%", sent):
            score += 2
        if re.search(r"(?i)temperature|vibration|pressure|humidity|bearing|lubricat", sent):
            score += 1
        return score

    evid_candidates_sorted = sorted(evid_candidates, key=score_evidence, reverse=True)
    for e in evid_candidates_sorted[:3]:
        evidence.append(e)
    # If none found, take up to two sentences around the root cause
    if not evidence:
        if root and root in s:
            # find sentences near root
            idx = None
            for i, sent in enumerate(all_sents):
                if root.strip() == sent.strip():
                    idx = i
                    break
            if idx is not None:
                if idx > 0:
                    evidence.append(all_sents[idx-1].strip())
                evidence.append(all_sents[idx].strip())
        else:
            for sent in all_sents[:2]:
                if sent.strip():
                    evidence.append(sent.strip())

    # 4) If no actions found, heuristically pick imperative lines or last sentences
    if not actions:
        cand_actions = []
        for ln in lines:
            if re.match(r"^(fix|check|replace|inspect|monitor|schedule|update|apply|restart|test|tighten|lubricate|calibrate)\b", ln, re.I):
                cand_actions.append(ln)
        if cand_actions:
            actions = cand_actions
        else:
            sents_tail = [sent.strip() for sent in all_sents if sent.strip()]
            actions = sents_tail[-2:] if len(sents_tail) >= 2 else sents_tail

    # 5) Build concise summary (one sentence) preferring explicit Summary section
    # Updated regex to match section headers more precisely (require colon or newline after keywords)
    m_sum = re.search(r"(?i)(summary[:\-]?\s*)(.+?)(?=(?:\n\s*(?:root\s*cause|recommended\s*actions?)[:\-])|$)", full, re.S)
    if m_sum:
        summary_raw = m_sum.group(2).strip()
    else:
        # If no explicit summary section, take content before "Root Cause:" or "Recommended Actions:"
        # Require colon or newline to distinguish section headers from inline text
        summary_match = re.search(r"^(.+?)(?=(?:\n\s*(?:root\s*cause|recommended\s*actions?)[:\-]))", full, re.DOTALL | re.IGNORECASE)
        if summary_match:
            summary_raw = summary_match.group(1).strip()
        else:
            summary_raw = s
    
    # Keep full summary instead of just first sentence
    summary = summary_raw if summary_raw else ""
    # Limit length only if extremely long
    if len(summary) > 1000:
        summary = summary[:1000] + "..."

    # 6) Simple confidence heuristic (0-1)
    conf = 0.3
    # more signals increase confidence
    if m_root:
        conf += 0.35
    # presence of numeric evidence increases confidence
    numeric_evidence_count = sum(1 for e in evidence if re.search(r"\d+", e))
    conf += min(0.3, 0.1 * numeric_evidence_count)
    # higher if explicit recommended actions present
    if m_actions:
        conf += 0.05
    conf = max(0.0, min(0.95, conf))

    # Ensure recommended actions is a list of strings and clean source markers
    recommended = actions if isinstance(actions, list) else [actions]
    recommended = [clean_source_markers(a) for a in recommended if a]

    return {
        "summary": clean_source_markers(summary),
        "root_cause": clean_source_markers(root),
        "recommended_actions": recommended,
        "evidence": [clean_source_markers(e) for e in evidence],
        "confidence": round(conf, 2)
    }


# Test cases
test_cases = [
    {
        "name": "Test 1: Well-formatted response",
        "input": """Summary: The motor is showing high temperature and vibration levels indicating potential bearing failure.

Root Cause: Bearing wear due to inadequate lubrication and high operational load.

Recommended Actions:
1. Inspect and replace motor bearings
2. Check lubrication system
3. Verify motor load is within specifications""",
        "expected": {
            "summary": "The motor is showing high temperature and vibration levels indicating potential bearing failure.",
            "root_cause": "Bearing wear due to inadequate lubrication and high operational load.",
            "recommended_actions": [
                "Inspect and replace motor bearings",
                "Check lubrication system",
                "Verify motor load is within specifications"
            ]
        }
    },
    {
        "name": "Test 2: Summary with 'root' word in content",
        "input": """Summary: The equipment shows signs of degradation at the root level of the system architecture.

Root Cause: Mechanical wear in rotating components.

Recommended Actions:
1. Perform maintenance
2. Replace worn parts""",
        "expected": {
            "summary": "The equipment shows signs of degradation at the root level of the system architecture.",
            "root_cause": "Mechanical wear in rotating components.",
            "recommended_actions": [
                "Perform maintenance",
                "Replace worn parts"
            ]
        }
    },
    {
        "name": "Test 3: Summary with 'recommended' word in content",
        "input": """Summary: Based on the recommended operating parameters, the pump is experiencing cavitation issues.

Root Cause: Insufficient inlet pressure causing cavitation.

Recommended Actions:
1. Increase inlet pressure
2. Check suction line for blockages""",
        "expected": {
            "summary": "Based on the recommended operating parameters, the pump is experiencing cavitation issues.",
            "root_cause": "Insufficient inlet pressure causing cavitation.",
            "recommended_actions": [
                "Increase inlet pressure",
                "Check suction line for blockages"
            ]
        }
    }
]

print("Testing parse_llm_analysis function...\n")
print("="*80)

all_passed = True
for test in test_cases:
    print(f"\n{test['name']}")
    print("-"*80)
    result = parse_llm_analysis(test['input'])
    
    # Check summary
    if result['summary'] == test['expected']['summary']:
        print(f"✓ Summary: PASS")
    else:
        print(f"✗ Summary: FAIL")
        print(f"  Expected: {test['expected']['summary']}")
        print(f"  Got:      {result['summary']}")
        all_passed = False
    
    # Check root cause
    if result['root_cause'] == test['expected']['root_cause']:
        print(f"✓ Root Cause: PASS")
    else:
        print(f"✗ Root Cause: FAIL")
        print(f"  Expected: {test['expected']['root_cause']}")
        print(f"  Got:      {result['root_cause']}")
        all_passed = False
    
    # Check recommended actions
    if result['recommended_actions'] == test['expected']['recommended_actions']:
        print(f"✓ Recommended Actions: PASS")
    else:
        print(f"✗ Recommended Actions: FAIL")
        print(f"  Expected: {test['expected']['recommended_actions']}")
        print(f"  Got:      {result['recommended_actions']}")
        all_passed = False

print("\n" + "="*80)
if all_passed:
    print("✓ All tests PASSED!")
else:
    print("✗ Some tests FAILED")
