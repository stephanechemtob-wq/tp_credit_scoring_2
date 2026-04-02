# conftest.py racine — configure le PYTHONPATH pour pytest
import os
import sys

# Ajoute src/ au path pour les imports du package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
