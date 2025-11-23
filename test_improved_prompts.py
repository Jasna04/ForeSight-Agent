"""
Test to demonstrate the improved LLM prompts that ensure distinct sections
"""

print("="*80)
print("IMPROVED LLM PROMPT STRATEGY")
print("="*80)

print("\n### BEFORE (Generic Prompt):")
print("-"*80)
old_prompt = """You are an expert predictive maintenance assistant. Use context sources. If answer not present, say so.

QUESTION:
Predictive maintenance for Industrial Pump showing:
- Temperature: 185°C
- Vibration: 6.2 mm/s
- Pressure: 82 PSI
- Humidity: 65%
- Power consumption: 145 kW
- Predicted failure in 3.5 days
What are the likely failure causes and recommended maintenance actions?"""

print(old_prompt)

print("\n❌ PROBLEM: This prompt is too vague and doesn't clearly separate:")
print("   - What's happening (Summary)")
print("   - Why it's happening (Root Cause)")
print("   - What to do (Actions)")
print("\n   Result: LLM often provides similar content in all three sections")

print("\n" + "="*80)
print("\n### AFTER (Structured Prompt):")
print("-"*80)

new_system = """You are an expert predictive maintenance assistant. Analyze the equipment data and provide a structured response.

IMPORTANT: Format your response with these THREE DISTINCT sections:

Summary:
[Brief overview of the current equipment status and predicted failure timeline - 2-3 sentences]

Root Cause:
[Detailed analysis of WHY the failure is occurring, based on sensor readings and technical factors - focus on underlying mechanisms]

Recommended Actions:
[Numbered list of specific maintenance tasks to perform]

Each section must contain DIFFERENT information. Do not repeat the same content across sections."""

new_question = """Analyze this Industrial Pump predictive maintenance scenario:

Sensor Readings:
- Temperature: 185°C
- Vibration: 6.2 mm/s
- Pressure: 82 PSI
- Humidity: 65%
- Power consumption: 145 kW
- Predicted failure in 3.5 days

Provide:
1. Summary: Brief status overview and urgency level
2. Root Cause: Technical explanation of why these specific sensor readings indicate failure
3. Recommended Actions: Specific numbered maintenance tasks

Make each section distinct with different information."""

print(new_system)
print("\nQUESTION:")
print(new_question)

print("\n" + "="*80)
print("\n✅ IMPROVEMENTS:")
print("-"*80)
print("1. Explicit instructions to format with THREE DISTINCT sections")
print("2. Clear description of what each section should contain:")
print("   - Summary: WHAT is happening + timeline")
print("   - Root Cause: WHY it's happening + technical explanation")
print("   - Recommended Actions: WHAT TO DO + specific tasks")
print("3. Direct instruction: 'Each section must contain DIFFERENT information'")
print("4. Increased max_tokens from 512 to 800 for detailed responses")
print("5. Temperature increased from 0.2 to 0.3 for more varied responses")
print("\n" + "="*80)

print("\n### EXPECTED LLM RESPONSE FORMAT:")
print("-"*80)
print("""
Summary: The industrial pump is experiencing critical degradation with elevated temperature (185°C) and excessive vibration (6.2 mm/s). Failure is predicted within 3.5 days, requiring immediate maintenance intervention to prevent unplanned downtime.

Root Cause: The high vibration levels combined with elevated temperature indicate bearing wear and potential shaft misalignment. When pump bearings deteriorate, friction increases, causing both temperature rise and mechanical vibration. The below-normal pressure (82 PSI) suggests reduced pumping efficiency due to internal component wear, likely from cavitation or impeller degradation.

Recommended Actions:
1. Immediately inspect and replace all pump bearings
2. Check and correct shaft alignment using dial indicators
3. Examine impeller for cavitation damage or erosion
4. Verify suction line pressure and check for air leaks
5. Replace mechanical seals if showing signs of leakage
""")
print("-"*80)
print("\n✅ Each section now provides UNIQUE, COMPLEMENTARY information!")
print("="*80)
