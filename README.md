# BuildingQA: A Benchmark for Natural Language Question Answering over Building Knowledge Graphs

This repository contains the **BuildingQA** benchmark, a dataset for Natural Language Question Answering over Building Knowledge Graphs (BKGs).

The benchmark consists of four comprehensive knowledge graphs and a curated set of natural language questions, each paired with a ground-truth SPARQL query. As a baseline for this benchmark, we also provide an implementation of a **ReAct (Reasoning and Acting)** agent designed to translate questions into SPARQL.

---

## The BuildingQA Benchmark Dataset

This benchmark is comprised of two main components:

### 1. Knowledge Graphs
* **Contents:** Four distinct Building Knowledge Graphs (BKGs).
* **Location:** `eval_buildings/`
* **Format:** Turtle (`.ttl`)

### 2. Question-Answer Pairs
* **Contents:** A collection of natural language questions, each manually paired with a ground-truth SPARQL query.
* **Location:** `Benchmark_QA_pairs/`
* **Format:** A separate JSON file is provided for each of the four buildings, containing all of its associated questions and queries.

---

## 📊 Evaluation Metrics

To evaluate performance on the benchmark, this project uses four metrics to assess query correctness, from basic structure to perfect content matching:

| Metric | What it Checks | Strictness | Column Alignment Performed? |
| :--- | :--- | :--- | :--- |
| **Arity Matching F1** | The **number** of columns. | Low | No |
| **Entity Set F1** | Sets of unique **values** within columns. | Medium | **Yes (Finds best match)** |
| **Row-Matching F1** | Row-for-row **content**. | High | **Yes (Finds best match)** |
| **Exact-Match F1** | Column **order** and row-for-row content. | Very High | No (Assumes fixed order) |

**Row-Matching F1** is the primary indicator of a perfectly correct query, as it confirms correct entities and relationships, regardless of column order.

---

## 🤖 Included Baseline: ReAct Agent

We provide a baseline agentic workflow to demonstrate how to interact with the benchmark. This agent uses a two-agent loop to iteratively generate and refine SPARQL queries:

1.  **Query Writer Agent:** Generates and revises SPARQL queries based on a question and graph context.
2.  **Critique Agent:** Evaluates the query's correctness and results. It either approves the query (`FINAL`) or provides specific feedback for improvement (`IMPROVE`).

### Key Agent Features
* **ReAct Framework:** Implements a two-agent loop for query generation and refinement.
* **Structured I/O:** Uses Pydantic models for reliable JSON-based LLM communication.
* **Flexible Data Sources:** Natively supports both remote SPARQL endpoints and the local `.ttl` graph files provided in this benchmark.
* **Context-Aware:** Injects a `num_triples` subset of the knowledge graph into the agent's context.

---

## 🚀 Running the Baseline Agent

### 1. Prerequisites
* Python 3.9+
* Access to an LLM API (e.g., OpenAI)
* A running SPARQL endpoint (like GraphDB) or local `.ttl` graph files.

### 2. Installation
1.  Clone this repository.
2.  Install the required Python packages (e.g., `rdflib`, `sparqlwrapper`, `openai`, `pydantic`).
   ```bash
pip install rdflib sparqlwrapper openai pydantic numpy pandas
```

### 3. Configuration
Open `ReAct_demo.ipynb` and configure the main parameters in the code cells:
1.  **API Key:** Set your `OPENAI_API_KEY` and initialize the `client`.
2.  **Experiment Parameters:** Set variables like `MODEL_NAME`, `BUILDING_NAME`, `num_triples`, `USE_LOCAL_TTL_FILE`, `REMOTE_ENDPOINT_URL`, and `LOG_FILE`.

### 4. Run the Workflow
1.  Open `ReAct_demo.ipynb` in Jupyter.
2.  Execute the cells. The `run_single_question()` function is configured for a quick test.
3.  Execute the `run_all_buildings()` cell to run the full benchmark against all your questions.
4.  Results will be saved to the CSV file specified in your `LOG_FILE` variable.

---

## Citation

If you use this work in your research, please cite the following paper:

```bibtex
@inproceedings{Mulayim2025BuildingQA,
  author = {Mulayim, Ozan Baris and Anwar, Avia and Saka, Mete and Paul, Lazlo and Prakash, Anand Krishnan and Fierro, Gabe and Pritoni, Marco and Berg'{e}s, Mario},
  title = {{BuildingQA: A Benchmark for Natural Language Question Answering over Building Knowledge Graphs}},
  booktitle = {The 12th ACM International Conference on Systems for Energy-Efficient Buildings, Cities, and Transportation (BUILDSYS '25)},
  year = {2025},
  month = nov,
  publisher = {ACM},
  address = {New York, NY, USA},
  pages = {1--11},
  numpages = {11},
  location = {Golden, CO, USA},
  doi = {10.1145/3736425.3770097},
  url = {[https://doi.org/10.1145/3736425.3770097](https://doi.org/10.1145/3736425.3770097)},
  series = {BUILDSYS '25}
}
