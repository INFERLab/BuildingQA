# Standard library imports
import csv
import itertools
import json
import os
import re
import time
import traceback
import uuid
import warnings
from typing import Any, Dict, List, Optional

# Third-party imports
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError
from pyparsing import ParseException
from rdflib import BNode, Graph, Literal, URIRef
from SPARQLWrapper import JSON, SPARQLWrapper
from metrics import (
    get_arity_matching_f1,
    get_entity_and_row_matching_f1,
    get_exact_match_f1
)



def get_kg_subset_content(original_ttl_path: str, max_triples: int) -> str:
    """
    Parses a TTL file and returns a string containing the prefixes and the first `max_triples`.
    If the graph is smaller than max_triples, it returns the full content.
    """
    full_graph = Graph()
    try:
        full_graph.parse(original_ttl_path, format="turtle")
        print(f"🔎 Original graph '{os.path.basename(original_ttl_path)}' contains {len(full_graph)} triples.")
    except Exception as e:
        print(f"❌ ERROR: Could not parse original TTL file at {original_ttl_path}. Reason: {e}")
        return ""

    # If the graph is small enough, use the whole thing
    if len(full_graph) <= max_triples:
        print(f"   -> Graph has {len(full_graph)} triples or fewer. Using full graph content for prompt.")
        return full_graph.serialize(format="turtle")

    print(f"   -> Graph is larger than {max_triples} triples. Creating a subset for the prompt...")
    subset_graph = Graph()
    # Copy namespaces to the subset graph
    for prefix, namespace in full_graph.namespace_manager.namespaces():
        subset_graph.bind(prefix, namespace)

    # Add the first `max_triples`
    for i, triple in enumerate(full_graph):
        if i >= max_triples:
            break
        subset_graph.add(triple)

    print(f"   -> ✅ Successfully created subset context with {len(subset_graph)} triples.")
    return subset_graph.serialize(format="turtle")

# --- Evaluation and Helper Functions ---

def extract_prefixes_from_ttl(ttl_path: str) -> str:
    """Dynamically extracts PREFIX declarations from a TTL file."""
    prefixes = []
    try:
        with open(ttl_path, 'r', encoding='utf-8') as f:
            for line in f:
                stripped_line = line.strip()
                if stripped_line.lower().startswith('@prefix'):
                    parts = stripped_line.split()
                    if len(parts) >= 3:
                        prefixes.append(f"PREFIX {parts[1]} {parts[2]}")
        print(f"✅ Successfully extracted {len(prefixes)} prefixes from {os.path.basename(ttl_path)}.")
        return "\n".join(prefixes) + "\n\n"
    except FileNotFoundError:
        print(f"❌ ERROR: TTL file not found at {ttl_path}.")
        return ""
    except Exception as e:
        print(f"❌ ERROR: Could not read prefixes from {ttl_path}. Reason: {e}")
        return ""


def check_if_question_exists(question_text: str, log_filename: str, model_name: str) -> bool:
    """Checks if a question has already been logged for a specific model."""
    if not os.path.exists(log_filename):
        return False
    try:
        with open(log_filename, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('question') == question_text and row.get('model') == model_name:
                    print(f"✅ Result for question '{question_text[:50]}...' and model '{model_name}' already exists. Skipping.")
                    return True
    except (FileNotFoundError, Exception) as e:
        print(f"Could not read log file {log_filename}. Error: {e}")
        return False
    return False

# --- CSV Logger Class ---

LOG_FIELDNAMES = [
    'query_id', 'question_number', 'source', 'question', 'model',
    'ground_truth_sparql', 'generated_sparql',
    'syntax_ok', 'returns_results', 'perfect_match',
    'gt_num_rows', 'gt_num_cols',
    'gen_num_rows', 'gen_num_cols',
    'arity_matching_f1',
    'exact_match_f1',
    'entity_set_f1',
    'row_matching_f1',
    'less_columns_flag',
    'prompt_tokens', 'completion_tokens', 'total_tokens'
]

class CsvLogger:
    """Handles writing log data to a CSV file, appending if it exists."""
    def __init__(self, filename: str, fieldnames: List[str]):
        self.filename = filename
        self.fieldnames = fieldnames
        
        # Check if file exists AND has content (size > 0)
        file_exists_and_has_content = False
        if os.path.exists(self.filename):
            if os.path.getsize(self.filename) > 0:
                file_exists_and_has_content = True
        # --- END MODIFIED LOGIC ---
        
        self.file = open(self.filename, 'a', newline='', encoding='utf-8')
        self.writer = csv.DictWriter(self.file, fieldnames=self.fieldnames)
        
        if not file_exists_and_has_content:
            self.writer.writeheader()
            self.file.flush()  # <-- ADD THIS to write header to disk immediately
            print(f"📝 New log file created (or empty file found). Writing header to {self.filename}")
        else:
            print(f"📝 Logger initialized. Appending to {self.filename}")

    def log(self, data: Dict[str, Any]):
        """Writes a single entry to the log file."""
        self.writer.writerow(data)
        self.file.flush()

    def close(self):
        """Closes the log file."""
        self.file.close()
        print(f"✅ Logger closed. Final results saved to {self.filename}")