import pandas as pd
import re
import numpy as np
import itertools
import time
from typing import Union, List, Dict, Any

# ==============================================================================
#  HELPER FUNCTIONS
# ==============================================================================

def normalize_value(val: Any) -> str:
    """
    Converts a SPARQL result value into a standardized string for reliable comparison.

    SPARQL results can have varied formatting (e.g., "value", "value"^^xsd:string, 
    "value"@en). This function ensures that these variations are treated as 
    identical by performing the following steps:
    1. Converts the value to a string.
    2. Strips leading/trailing whitespace and converts to lowercase.
    3. Removes surrounding quotes.
    4. Strips RDF datatype (e.g., ^^xsd:string) and language tags (e.g., @en).

    Args:
        val: The value from a SPARQL result binding.

    Returns:
        A normalized, canonical string representation of the value.
    """
    if not isinstance(val, str):
        val = str(val)
    val = val.strip().lower()
    if len(val) > 1 and val.startswith('"') and val.endswith('"'):
        val = val[1:-1]
    val = re.sub(r'\^\^.*$', '', val)
    val = re.sub(r'@[a-z]{2}(-[a-z]{2})?$', '', val)
    return val

def row_to_partial_tuple(row: Dict[str, Any], keys_to_include: set) -> tuple:
    """
    Converts a dictionary-based row into a canonical, hashable tuple format.

    This is a crucial step for comparing rows. It creates a sorted tuple of 
    (key, normalized_value) pairs, making the representation independent of the 
    original dictionary's key order. This allows for direct comparison between two
    processed rows to check for equality.

    Args:
        row: A single result row, as a dictionary.
        keys_to_include: A set of keys that should be included in the tuple.

    Returns:
        A sorted tuple representing the row's content, e.g., (('age', '30'), ('name', 'alice')).
    """
    return tuple(sorted(
        (k, normalize_value(v.get('value', ''))) for k, v in row.items() if k in keys_to_include
    ))

def count_sparql_projections(query_str: str) -> int:
    """
    Parses a SPARQL query string to count the number of projected variables.

    This function determines the "arity" (number of columns) of a query by
    counting the variables in the SELECT clause. It is designed to handle 
    complex projections, such as (COUNT(?s) as ?count), not just simple variables.

    Args:
        query_str: The full SPARQL query as a string.

    Returns:
        The integer count of projected variables.
    """
    if not isinstance(query_str, str) or pd.isna(query_str): return 0
    match = re.search(r'SELECT\s+(.+?)\s+WHERE', query_str, re.IGNORECASE | re.DOTALL)
    if not match: return 0
    clause = match.group(1).strip()
    tokens, paren_depth, current_token = [], 0, ""
    for char in clause:
        if char == '(': paren_depth += 1
        elif char == ')': paren_depth -= 1
        if char.isspace() and paren_depth == 0:
            if current_token: tokens.append(current_token); current_token = ""
        else: current_token += char
    if current_token: tokens.append(current_token)
    return len([t for t in tokens if t.upper() != 'DISTINCT'])

def _get_f1_from_counts(gen_count: float, gt_count: float) -> float:
    """
    Calculates the F1 score (Dice Coefficient) between two integer counts.

    This is the core calculation for the arity matching metric, comparing the 
    number of predicted columns to the number of ground truth columns. The formula is:
    F1 = 2 * |Intersection| / (|Set A| + |Set B|)
       = 2 * min(count1, count2) / (count1 + count2)

    Args:
        gen_count: The first count (e.g., number of predicted columns).
        gt_count: The second count (e.g., number of gold columns).

    Returns:
        The F1 score, a float between 0.0 and 1.0.
    """
    if pd.isna(gen_count) or pd.isna(gt_count): return np.nan
    gen_count, gt_count = int(gen_count), int(gt_count)
    if gt_count == 0 and gen_count == 0: return 1.0
    denominator = gen_count + gt_count
    if denominator == 0: return 0.0
    return (2 * min(gen_count, gt_count)) / denominator

def _calculate_scores_for_mapping(gold_rows: List[Dict], pred_rows: List[Dict], column_mapping: Dict[str, str]) -> Dict:
    """
    Calculates column-wise and row-wise F1 scores for a single, fixed column mapping.

    This is the workhorse function called by the permutation search. It assumes a specific
    alignment between predicted and gold columns and computes two scores based on it:
    - Column F1: Averages the F1 score of the sets of unique values for each mapped column pair.
    - Row F1: Renames predicted columns and performs a one-to-one count of identical rows.

    Args:
        gold_rows: The list of ground-truth result rows.
        pred_rows: The list of predicted result rows.
        column_mapping: A dictionary mapping predicted column names to ground-truth column names.

    Returns:
        A dictionary containing the calculated 'column_wise' and 'row_wise' F1 scores.
    """
    if not column_mapping:
        return {"column_wise": {"f1": 0.0}, "row_wise": {"f1": 0.0}}

    # Column-wise F1
    col_precisions, col_recalls = [], []
    for pred_col, gold_col in column_mapping.items():
        gold_values = {normalize_value(r.get(gold_col, {}).get('value', '')) for r in gold_rows}
        pred_values = {normalize_value(r.get(pred_col, {}).get('value', '')) for r in pred_rows}
        tp = len(gold_values.intersection(pred_values))
        col_prec = tp / len(pred_values) if pred_values else 1.0
        col_rec = tp / len(gold_values) if gold_values else 1.0
        col_precisions.append(col_prec); col_recalls.append(col_rec)
    
    avg_precision_col = np.mean(col_precisions) if col_precisions else 0.0
    avg_recall_col = np.mean(col_recalls) if col_recalls else 0.0
    f1_col = 0.0
    if (avg_precision_col + avg_recall_col) > 0:
        f1_col = (2 * avg_precision_col * avg_recall_col) / (avg_precision_col + avg_recall_col)

    # Row-wise F1
    mapped_gold_keys = set(column_mapping.values())
    normalized_pred_rows = [{column_mapping.get(k): v for k, v in r.items() if k in column_mapping} for r in pred_rows]
    matched_gold, matched_pred = set(), set()
    for i, g_row in enumerate(gold_rows):
        for j, norm_p_row in enumerate(normalized_pred_rows):
            if j in matched_pred: continue
            if row_to_partial_tuple(g_row, mapped_gold_keys) == row_to_partial_tuple(norm_p_row, mapped_gold_keys):
                matched_gold.add(i); matched_pred.add(j); break
    
    tp = len(matched_gold); fp = len(pred_rows) - len(matched_pred); fn = len(gold_rows) - len(matched_gold)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1_row = 0.0
    if (precision + recall) > 0:
        f1_row = (2 * precision * recall) / (precision + recall)
        
    return {"column_wise": {"f1": f1_col}, "row_wise": {"f1": f1_row}}

def _find_best_mapping_and_scores(gold_rows: List[Dict], pred_rows: List[Dict], timeout: int = 1200) -> Dict:
    """
    Finds the optimal column alignment by testing all permutations.

    This function automates the column mapping process. It generates every possible
    one-to-one mapping between the predicted and ground-truth columns. For each
    mapping, it calculates row and column F1 scores. It then returns the scores 
    from the single best-performing mapping found. The search stops early if a 
    perfect score (1.0) is found or if the timeout is reached.

    Args:
        gold_rows: The list of ground-truth result rows.
        pred_rows: The list of predicted result rows.
        timeout: The maximum number of seconds to spend searching for the best mapping.

    Returns:
        A dictionary of the best column-wise and row-wise F1 scores found.
    """
    zero_scores = {"column_wise": {"f1": 0.0}, "row_wise": {"f1": 0.0}}
    if not gold_rows or not pred_rows:
        is_perfect = not gold_rows and not pred_rows
        score = 1.0 if is_perfect else 0.0
        return {"column_wise": {"f1": score}, "row_wise": {"f1": score}}

    gold_cols = list(gold_rows[0].keys())
    pred_cols = list(pred_rows[0].keys())
    n_gold, n_pred = len(gold_cols), len(pred_cols)
    
    if n_pred < n_gold:
        return zero_scores

    best_scores = zero_scores
    max_row_f1 = -1.0
    max_col_f1 = -1.0
    
    start_time = time.time()
    # Find the best permutation of pred_cols that matches the gold_cols
    for p_cols_perm in itertools.permutations(pred_cols, n_gold):
        if time.time() - start_time > timeout:
            print(f"⚠️  Alignment search timed out after {timeout} seconds. Returning best result found.")
            break
        
        current_mapping = {p_cols_perm[i]: gold_cols[i] for i in range(n_gold)}
        current_scores = _calculate_scores_for_mapping(gold_rows, pred_rows, current_mapping)
        
        if (current_scores['row_wise']['f1'] > max_row_f1 or
           (current_scores['row_wise']['f1'] == max_row_f1 and
            current_scores['column_wise']['f1'] > max_col_f1)):
            best_scores = current_scores
            max_row_f1 = current_scores['row_wise']['f1']
            max_col_f1 = current_scores['column_wise']['f1']
        
        # If a perfect score is found with any permutation, stop searching.
        if max_row_f1 == 1.0 and max_col_f1 == 1.0:
            break
            
    return best_scores
# ==============================================================================
#  NEW METRIC
# ==============================================================================

def _compute_single_col_f1(pred_values: set, gold_values: set) -> float:
    """
    Computes the standard F1 score (Dice coefficient) between two sets of values.
    """
    if not pred_values and not gold_values:
        return 1.0
    if not pred_values or not gold_values:
        return 0.0
        
    tp = len(pred_values.intersection(gold_values))
    # Standard F1: 2*TP / (len(A) + len(B))
    return (2 * tp) / (len(pred_values) + len(gold_values))

# ==============================================================================
#  NEW PUBLIC METRIC
# ==============================================================================

def get_best_subset_column_f1(gold_rows: List[Dict], pred_rows: List[Dict]) -> float:
    """
    Calculates the 'Best Subset' Column Score (Column Recall).

    Logic:
    1. Compute the F1 score for every possible pair of (pred_col, gold_col).
    2. Find the best alignment.
    3. Normalize by the number of GOLD columns only.
       (This ensures we do not penalize for extra/over-generated columns).

    Example:
        Gold: [Col A, Col B]
        Pred: [Col A, Col B, Col C] (Extra column C)
        
        Matches: (Pred A -> Gold A) = 1.0, (Pred B -> Gold B) = 1.0
        Total Score Sum: 2.0
        Denominator: len(Gold) = 2
        Final Result: 1.0 (Perfect Score)
    """
    # 1. Handle edge cases
    if not gold_rows:
        # If gold is empty, and we predicted nothing, it's perfect (1.0).
        # If we predicted something, it's technically "extra" but since we ignore extra, it's still 1.0?
        # Usually, if gold is empty, the task is to return nothing.
        return 1.0 if not pred_rows else 0.0 
        
    if not pred_rows:
        return 0.0

    gold_cols = list(gold_rows[0].keys())
    pred_cols = list(pred_rows[0].keys())
    
    n_gold = len(gold_cols)
    n_pred = len(pred_cols)
    
    # 2. Pre-extract values into sets
    gold_val_sets = [
        {normalize_value(r.get(c, {}).get('value', '')) for r in gold_rows}
        for c in gold_cols
    ]
    pred_val_sets = [
        {normalize_value(r.get(c, {}).get('value', '')) for r in pred_rows}
        for c in pred_cols
    ]

    # 3. Create Score Matrix (Pred x Gold)
    score_matrix = np.zeros((n_pred, n_gold))
    for i in range(n_pred):
        for j in range(n_gold):
            score_matrix[i][j] = _compute_single_col_f1(pred_val_sets[i], gold_val_sets[j])

    # 4. Find Best Alignment
    max_total_score = 0.0
    
    if n_pred <= n_gold:
        # Fewer predictions than gold (Under-generation): 
        # We try to find which Gold columns we actually found.
        for gold_indices in itertools.permutations(range(n_gold), n_pred):
            current_sum = sum(score_matrix[i][gold_indices[i]] for i in range(n_pred))
            if current_sum > max_total_score:
                max_total_score = current_sum
    else:
        # More predictions than gold (Over-generation):
        # We pick the subset of Predictions that best match the Gold.
        for pred_indices in itertools.permutations(range(n_pred), n_gold):
            current_sum = sum(score_matrix[pred_indices[j]][j] for j in range(n_gold))
            if current_sum > max_total_score:
                max_total_score = current_sum

    # 5. Normalize by Gold Count (Recall-focused)
    # If we matched 2 gold columns perfectly, sum is 2.0. 
    # Divide by n_gold (2) = 1.0.
    return max_total_score / n_gold
# ==============================================================================
#  PUBLIC METRIC FUNCTIONS
# ==============================================================================

def get_arity_matching_f1(generated_output: Union[str, List[Dict]], gold_input: Union[str, List[Dict]]) -> float:
    """
    Calculates the Arity Matching F1 score.

    This metric answers the question: "Did the query return the correct number of columns?"
    It compares the arity (column count) of the predicted result against the 
    ground truth. It can determine the arity from either a SPARQL query string or a list of
    result rows for both the generated and gold inputs.
    """
    gen_count = 0
    if isinstance(generated_output, str):
        gen_count = count_sparql_projections(generated_output)
    elif isinstance(generated_output, list):
        gen_count = len(generated_output[0]) if generated_output else 0
    
    gt_count = 0
    if isinstance(gold_input, str):
        gt_count = count_sparql_projections(gold_input)
    elif isinstance(gold_input, list):
        gt_count = len(gold_input[0]) if gold_input else 0

    return _get_f1_from_counts(gen_count, gt_count)

def get_entity_set_f1(gold_rows: List[Dict], pred_rows: List[Dict]) -> float:
    """
    Calculates the Entity Set F1 (col_f1) after finding the best column alignment.

    This metric answers: "Does the predicted result contain the correct sets of 
    values, ignoring row structure?" It first finds the optimal column alignment, 
    then for each aligned pair, it compares the set of unique values. It's a 
    measure of whether the right entities were retrieved, even if they are not 
    structured into the correct rows.
    """
    best_scores = _find_best_mapping_and_scores(gold_rows, pred_rows)
    return best_scores['column_wise']['f1']

def get_row_matching_f1(gold_rows: List[Dict], pred_rows: List[Dict]) -> float:
    """
    Calculates the Row Matching F1 (row_f1) after finding the best column alignment.

    This metric answers: "After finding the best column alignment, how many of the 
    rows are exactly correct?" It first automates the column mapping and then 
    performs a one-to-one comparison of the complete rows. This is a strong 
    indicator of overall query correctness.
    """
    best_scores = _find_best_mapping_and_scores(gold_rows, pred_rows)
    return best_scores['row_wise']['f1']

def get_exact_match_f1(gold_rows: List[Dict], pred_rows: List[Dict]) -> float:
    """
    Calculates the Exact Match F1 score, requiring an identical column order.

    This is a strict metric that answers: "Is the predicted result identical to the
    ground truth in structure and content?" It requires that the number and order of
    columns be exactly the same, but does NOT require the column *names* to match.
    It then checks for a one-to-one match of row content based on this fixed order.
    No column permutation is performed. A score of 1.0 indicates a perfect
    positional match.
    """
    if not gold_rows and not pred_rows: return 1.0
    if not gold_rows or not pred_rows: return 0.0
    
    gold_cols_list = list(gold_rows[0].keys())
    pred_cols_list = list(pred_rows[0].keys())
    
    # The primary check: The number of columns must be the same.
    # The order is implicitly enforced by the positional mapping created below.
    if len(gold_cols_list) != len(pred_cols_list):
        return 0.0
    
    # Create a mapping based on column *position*, not name.
    positional_mapping = {pred_cols_list[i]: gold_cols_list[i] for i in range(len(gold_cols_list))}
    
    scores = _calculate_scores_for_mapping(gold_rows, pred_rows, positional_mapping)
    return scores['row_wise']['f1']

