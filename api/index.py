"""
Vercel entry point.

Vercel runs every Python file inside /api as a serverless function.
This file only adds the backend folder to the import path and exposes the Flask app.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend"))

from app import app  # noqa: E402,F401  (Vercel looks for a variable named `app`)
