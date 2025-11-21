#!/usr/bin/env python3
"""Test if all required imports work"""

print("Testing imports...")

try:
    import streamlit as st
    print("✓ streamlit")
except Exception as e:
    print(f"✗ streamlit: {e}")

try:
    import pandas as pd
    print("✓ pandas")
except Exception as e:
    print(f"✗ pandas: {e}")

try:
    import numpy as np
    print("✓ numpy")
except Exception as e:
    print(f"✗ numpy: {e}")

try:
    import joblib
    print(f"✓ joblib (version {joblib.__version__})")
except Exception as e:
    print(f"✗ joblib: {e}")

try:
    import requests
    print("✓ requests")
except Exception as e:
    print(f"✗ requests: {e}")

try:
    import plotly.graph_objects as go
    print("✓ plotly")
except Exception as e:
    print(f"✗ plotly: {e}")

try:
    import shap
    print("✓ shap")
except Exception as e:
    print(f"✗ shap: {e}")

try:
    import matplotlib.pyplot as plt
    print("✓ matplotlib")
except Exception as e:
    print(f"✗ matplotlib: {e}")

try:
    import openai
    print("✓ openai")
except Exception as e:
    print(f"✗ openai: {e}")

try:
    from sendgrid import SendGridAPIClient
    print("✓ sendgrid")
except Exception as e:
    print(f"✗ sendgrid: {e}")

print("\nAll imports tested!")
