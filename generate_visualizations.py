# -*- coding: utf-8 -*-
"""
generate_visualizations.py
--------------------------
Generates all comparative visualizations and scorecards comparing Qwen and Gemini
across the 5 benchmark datasets (SciQ, OpenBookQA, ARC-Challenge, RACE, and SQuAD v1.1).

Outputs figures into the figures/ directory.
"""

from generate_clean_white_visualizations import generate_all_clean_white_visualizations

if __name__ == "__main__":
    generate_all_clean_white_visualizations()
