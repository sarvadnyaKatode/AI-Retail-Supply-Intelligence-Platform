"""
Streamlit Cloud entry point.
This file is at project root so Streamlit Cloud can find it directly.
It simply re-exports the dashboard app.
"""
import sys
import os

# Make sure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Run the dashboard
exec(open(os.path.join(os.path.dirname(__file__), "dashboard", "app.py")).read())
