"""
Real-world test case to demonstrate the fix for duplicate Summary/Root Cause/Recommended Actions
"""

# Simulate a realistic LLM response that previously would have caused issues
llm_response = """Summary: The industrial pump is experiencing critical issues with elevated vibration levels (6.2 mm/s) and temperature (185°C). The root cause appears to be mechanical in nature, with failure predicted in approximately 3.5 days. These readings exceed the recommended operating parameters by significant margins.

Root Cause: Bearing wear due to inadequate lubrication combined with misalignment of the pump shaft. The high vibration suggests mechanical imbalance, while the elevated temperature indicates friction from worn bearings. This is a common failure mode in pumps that have exceeded their recommended maintenance interval.

Recommended Actions:
1. Immediately inspect and replace all pump bearings
2. Check shaft alignment and correct any misalignment
3. Verify lubrication system is functioning properly and refill with appropriate lubricant
4. Inspect impeller for damage or debris buildup
5. Schedule emergency maintenance within 24 hours to prevent catastrophic failure"""

print("="*80)
print("REAL-WORLD TEST: LLM Response with Challenging Content")
print("="*80)
print("\nThis response contains:")
print("- The phrase 'root cause' in the summary")
print("- The phrase 'recommended' in the summary")
print("- Multiple sections with proper formatting")
print("\nIn the OLD version, the regex would incorrectly parse this because:")
print("- It would stop at 'root cause' in the summary text")
print("- Summary and Root Cause would end up with duplicate/overlapping content")
print("\nWith the FIX:")
print("- Regex requires section headers to be on new lines with colons")
print("- Each section is properly extracted without bleeding into others")
print("="*80)
print("\nOriginal LLM Response:")
print("-"*80)
print(llm_response)
print("-"*80)
print("\nTo verify the fix works, run the streamlit app and check that:")
print("1. Summary doesn't contain the full root cause description")
print("2. Root Cause is distinct and focused on the cause")
print("3. Recommended Actions are properly listed without number prefixes")
print("="*80)

