#!/bin/bash
# Convenience script to run tests and demo
set -e

echo "🔍 Running Tests..."
source .venv/bin/activate
cd toondb_rag
python -m pytest tests/ -v

echo -e "\n🚀 Running Demo..."
python demo.py
