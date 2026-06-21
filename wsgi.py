"""
WSGI Entry Point — Event Impact & Response Intelligence Platform
================================================================
Used by gunicorn in production:
    gunicorn wsgi:app --bind 0.0.0.0:$PORT

Assets are loaded at module level in src/server.py when it's imported,
so no additional initialization is needed here.
"""

import sys
import os

# Add the project root to Python path so `src` is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from src.server import app  # noqa: E402

if __name__ == "__main__":
    app.run()
