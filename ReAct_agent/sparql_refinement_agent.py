"""
sparql_refinement_agent.py

Implements the SparqlRefinementAgent class for iterative SPARQL query generation and critique.
"""

import csv
import itertools
import json
import os
import re
import sys
import time
import traceback
import uuid
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Third-party imports
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError
from pyparsing import ParseException
from rdflib import BNode, Graph, Literal, URIRef
from SPARQLWrapper import JSON, SPARQLWrapper

# Local imports
from metrics import (
    get_arity_matching_f1,
    get_entity_and_row_matching_f1,
    get_exact_match_f1
)

from ReAct_agent.utils import (
    get_kg_subset_content, 
    extract_prefixes_from_ttl, 
    check_if_question_exists, 
    CsvLogger
)
# --- Pydantic Models for the Two-Agent Workflow ---

class SparqlQuery(BaseModel):
    """Model for the Query Writer Agent's output."""
    sparql_query: str = Field(..., description="The generated or revised SPARQL query.")


class QueryCritique(BaseModel):
    """Model for the Critique Agent's structured feedback."""
    decision: str = Field(..., description="The decision, either 'IMPROVE' or 'FINAL'.")
    feedback: str = Field(..., description="Natural language feedback explaining the decision.")


# --- The Orchestrating Agent Class ---

class SparqlRefinementAgent:
    """
    An agent that orchestrates a conversation between a Query Writer and a Critique Agent
    to iteratively develop, evaluate, and log a SPARQL query.
    Can query a remote SPARQL endpoint or a local TTL file.
    """

    def __init__(self, sparql_endpoint: str, model_name: str = "openai/o4-mini", max_iterations: int = 5, client: OpenAI = None):
        self.client = client
        self.sparql_endpoint_url = sparql_endpoint
        self.model_name = model_name
        self.max_iterations = max_iterations
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0

        # --- Differentiate between remote endpoint and local file ---
        self.graph = None
        self.is_remote = sparql_endpoint.lower().startswith("http")

        if self.is_remote:
            print(f"🌐 Remote SPARQL endpoint mode activated: {self.sparql_endpoint_url}")
        else:
            print(f"🗂️ Local TTL file mode activated. Loading graph from: {self.sparql_endpoint_url}")
            if not os.path.exists(self.sparql_endpoint_url):
                print(f"   -> ❌ ERROR: File not found at {self.sparql_endpoint_url}. Queries will fail.")
                return
            try:
                self.graph = Graph()
                self.graph.parse(self.sparql_endpoint_url, format="turtle")
                print(f"   -> ✅ Graph loaded successfully with {len(self.graph)} triples.")
            except Exception as e:
                print(f"   -> ❌ ERROR: Failed to load or parse the TTL file: {e}")
                self.graph = None

    def _format_rdflib_results(self, qres) -> Dict[str, Any]:
        """Converts rdflib QueryResult to the same dict format as SPARQLWrapper."""
        variables = [str(v) for v in qres.vars]
        bindings = []
        for row in qres:
            binding_row = {}
            for var_name in variables:
                term = row[var_name]
                if term is None:
                    continue
                
                term_dict = {}
                if isinstance(term, URIRef):
                    term_dict = {'type': 'uri', 'value': str(term)}
                elif isinstance(term, Literal):
                    term_dict = {'type': 'literal', 'value': str(term)}
                    if term.datatype:
                        term_dict['datatype'] = str(term.datatype)
                    if term.language:
                        term_dict['xml:lang'] = term.language
                elif isinstance(term, BNode):
                    term_dict = {'type': 'bnode', 'value': str(term)}
                
                binding_row[var_name] = term_dict
            bindings.append(binding_row)
        
        return {"results": bindings, "variables": variables}
    def _run_sparql_query(self, query: str) -> Dict[str, Any]:
        """
        Executes a SPARQL query, dispatching to rdflib (local) or SPARQLWrapper (remote).
        Returns a structured dictionary of results.
        """
        print(f"\n🔎 Running SPARQL query... (first 80 chars: {query[:80].replace(chr(10), ' ')}...)")
        
        # --- NEW: Branch for local RDF file (rdflib) ---
        if not self.is_remote:
            if self.graph is None:
                return {"summary_string": "SPARQL query failed: The local RDF graph is not loaded.", "results": [], "row_count": 0, "col_count": 0, "syntax_ok": False, "error_message": "Graph not loaded."}
            
            try:
                qres = self.graph.query(query)
                formatted_results = self._format_rdflib_results(qres)
                bindings = formatted_results["results"]
                summary = f"Query executed successfully on local graph. Found {len(bindings)} results."
                if not bindings:
                    summary = "The query executed successfully on the local graph but returned no results."
                
                return {"summary_string": summary, "results": bindings, "row_count": len(bindings), "col_count": len(formatted_results["variables"]), "syntax_ok": True, "error_message": None}
            except (ParseException, Exception) as e:
                print(f"   -> SPARQL Query (local) Failed: {e}")
                error_msg = f"The query failed to parse with the following error: {str(e)}"
                return {"summary_string": error_msg, "results": [], "row_count": 0, "col_count": 0, "syntax_ok": False, "error_message": str(e)}

        # --- Original logic for remote SPARQL endpoint ---
        else:
            try:
                sparql = SPARQLWrapper(self.sparql_endpoint_url)
                sparql.setQuery(query)
                sparql.setReturnFormat(JSON)
                results_json = sparql.query().convert()
                
                bindings = results_json.get("results", {}).get("bindings", [])
                variables = results_json.get("head", {}).get("vars", [])
                
                summary = "Query executed successfully. Here are the first 10 results:\n" + json.dumps(bindings[:10], indent=2)
                if not bindings:
                    summary = "The query executed successfully but returned no results."

                return {"summary_string": summary, "results": bindings, "row_count": len(bindings), "col_count": len(variables), "syntax_ok": True, "error_message": None}
            except Exception as e:
                print(f"   -> SPARQL Query (remote) Failed: {e}")
                return {"summary_string": f"The query failed to execute with the following error: {str(e)}", "results": [], "row_count": 0, "col_count": 0, "syntax_ok": False, "error_message": str(e)}

    def _get_structured_response(self, messages: List[Dict], response_model) -> Optional[BaseModel]:
        """Generic helper to call the LLM, parse its response, and count tokens."""
        content = ""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0,
            )
            if hasattr(response, "usage") and response.usage:
                self.prompt_tokens += response.usage.prompt_tokens
                self.completion_tokens += response.usage.completion_tokens
                self.total_tokens += response.usage.total_tokens
            content = response.choices[0].message.content
            return response_model.model_validate_json(content)
        except (ValidationError, json.JSONDecodeError) as e:
            print(f"Pydantic/JSON validation failed: {e}")
            print(f"--- Failing LLM Response ---\n{content}\n-----------------------------")
            return None
        except Exception as e:
            print(f"Unexpected error during LLM call: {e}")
            return None

    def refine_and_evaluate_query(self, eval_data: Dict[str, Any], logger: CsvLogger, prefixes: str, knowledge_graph_content: str) -> None:
        """Main loop to refine, evaluate, and log a query."""
        self.prompt_tokens = self.completion_tokens = self.total_tokens = 0
        nl_question = eval_data['question']
        ground_truth_sparql = eval_data.get('ground_truth_sparql')

        print(f"\n🚀 Starting refinement workflow for question: '{nl_question}'")
        system_prompt = (
            f"You are an expert SPARQL developer for Brick Schema and ASHRAE 223p. "
            f"Your job is to write a single, complete SPARQL query to answer the user's request. "
            f"Here is a relevant subgraph for your context:\n\n"
            f"```turtle\n{knowledge_graph_content}\n```\n\n"
            f"If you are unsure about how many projections to return, return more rather than fewer. "
            f"If you are given feedback on a prior attempt, use it to revise and improve your query. "
            f"Respond ONLY with a JSON object containing the key 'sparql_query'."
        )

        query_writer_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"User Question: {nl_question}"}
        ]

        #get the file name from the logger and strip .csv
        log_file_name = os.path.basename(logger.filename).replace('.csv', '')
        print(f"📝 Logging to file: {log_file_name}")
        #save the system prompt as a text file with the log file name and _system_prompt.txt
        #with open(f"{log_file_name}_system_prompt.txt", "w", encoding="utf-8") as f:
        #    f.write(system_prompt)
        
        #save the query writer messages as a text file with the log file name and _query_writer_messages.txt
        #with open(f"{log_file_name}_query_writer_messages.txt", "w", encoding="utf-8") as f:
        #    for msg in query_writer_messages:
        #        f.write(f"{msg['role'].upper()}:\n{msg['content']}\n\n")


        final_generated_query = ""

        for i in range(self.max_iterations):
            print(f"\n--- Iteration {i + 1} ---")
            print("✍️  Calling Query Writer Agent...")
            query_response = self._get_structured_response(query_writer_messages, SparqlQuery)
            
            if not query_response or not query_response.sparql_query:
                print("❌ Query Writer failed to produce a valid query. Aborting iteration.")
                break
            
            final_generated_query = query_response.sparql_query
            print(f"   -> Query received:\n{final_generated_query}")
            query_writer_messages.append({"role": "assistant", "content": json.dumps(query_response.model_dump())})

            results_obj = self._run_sparql_query(final_generated_query)
            print(f"   -> Results Summary: {results_obj['summary_string'][:250]}...")

            print("🧐 Calling Critique Agent...")
            critique_prompt = [
                {"role": "system", "content": "You are an expert in SPARQL especially for Brick Schema and ASHRAE 223p. Your job is to review a SPARQL query and its results based on an original question. Decide if the query is correct or needs improvement. Respond with a JSON object: `{\"decision\": \"FINAL\" | \"IMPROVE\", \"feedback\": \"...your reasoning...\"}`."},
                {"role": "user", "content": f"Original Question: \"{nl_question}\"\n\nSPARQL Query Attempt:\n```sparql\n{final_generated_query}\n```\n\nExecution Results Summary:\n{results_obj['summary_string']}"}
            ]
            critique = self._get_structured_response(critique_prompt, QueryCritique)

            if not critique:
                print("❌ Critique Agent failed. Ending refinement loop.")
                break
            
            print(f"   -> Critique Decision: {critique.decision}")
            print(f"   -> Critique Feedback: {critique.feedback}")

            if critique.decision == "FINAL":
                print("\n✅ Critique Agent approved the query. Refinement complete.")
                break
            
            feedback_for_writer = f"Your last query attempt received the following feedback: '{critique.feedback}'. Please provide a new, improved query that addresses this feedback."
            query_writer_messages.append({"role": "user", "content": feedback_for_writer})
        
        
        if not final_generated_query:
            print("💔 Agentic workflow could not produce a final query.")
            return


        print("\n--- Final Evaluation and Logging ---")
        gen_results_obj = self._run_sparql_query(final_generated_query)
        gt_results_obj = self._run_sparql_query(ground_truth_sparql) if ground_truth_sparql else None
        
        # Initialize metrics to default values
        arity_f1, entity_set_f1, row_matching_f1, exact_match_f1 = 0.0, 0.0, 0.0, 0.0
        less_columns_flag = False
        
        # Calculate metrics only if both ground truth and generated queries are valid
        if gt_results_obj and gt_results_obj["syntax_ok"] and gen_results_obj["syntax_ok"]:
            gold_rows = gt_results_obj["results"]
            pred_rows = gen_results_obj["results"]
            
            arity_f1 = get_arity_matching_f1(final_generated_query, ground_truth_sparql)
            entity_and_row_f1 = get_entity_and_row_matching_f1(gold_rows=gold_rows, pred_rows=pred_rows)
            entity_set_f1 = entity_and_row_f1['entity_set_f1']
            row_matching_f1 = entity_and_row_f1['row_matching_f1']
            exact_match_f1 = get_exact_match_f1(gold_rows=gold_rows, pred_rows=pred_rows)
            
            # Determine if the generated query returned fewer columns than the ground truth
            less_columns_flag = gen_results_obj['col_count'] < gt_results_obj['col_count']
        
        log_entry = {
            **eval_data,
            'model': self.model_name,
            'generated_sparql': final_generated_query,
            'syntax_ok': gen_results_obj['syntax_ok'],
            'returns_results': gen_results_obj['row_count'] > 0,
            'perfect_match': row_matching_f1 == 1.0, # Row Matching F1 is the best indicator for this
            'gt_num_rows': gt_results_obj['row_count'] if gt_results_obj else 0,
            'gt_num_cols': gt_results_obj['col_count'] if gt_results_obj else 0,
            'gen_num_rows': gen_results_obj['row_count'],
            'gen_num_cols': gen_results_obj['col_count'],
            'arity_matching_f1': arity_f1,
            'entity_set_f1': entity_set_f1,
            'row_matching_f1': row_matching_f1,
            'exact_match_f1': exact_match_f1,
            'less_columns_flag': less_columns_flag,
            'prompt_tokens': self.prompt_tokens,
            'completion_tokens': self.completion_tokens,
            'total_tokens': self.total_tokens
        }

        logger.log(log_entry)
        print(f"📊 Log entry saved for query_id: {eval_data['query_id']}")
