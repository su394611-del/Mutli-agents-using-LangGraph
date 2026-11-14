##   Multi-agents using LangGraph
- `langgraph_example_multiagents.py` — one **runnable Python script**, compatible with Windows PowerShell, VS Code, and macOS/Linux terminals.
- `langgraph_example_mutliagents.ipynb` — one **self-contained Jupyter notebook**; every topic includes its code and a runnable example.
- `requirements_langgraph_example.txt` — one dependency list for the offline examples, optional OpenAI examples, and Jupyter.
- `run_langgraph_windows.cmd` — optional Windows double-click launcher (runs all offline examples).
#. It does **not** require PyTorch or a GPU: LangGraph orchestrates nodes; it does not train an LLM. For offline LLM, a consumer-grade single GPU or consumer-grade two GPUs are sufficient.

## Requirements and installation (Windows 11, Anaconda or VS Code terminal)

Recommended: Python 3.11 or 3.12 (3.13 may also be supported depending on dependency wheels).

```powershell
cd C:\HF_Projects\1
py -3.12 -m venv .venv-langgraph
.\.venv-langgraph\Scripts\python.exe -m pip install --upgrade pip
.\.venv-langgraph\Scripts\python.exe -m pip install -r .\requirements_langgraph_example.txt
```

If `py -3.12` is not installed, use `python -m venv .venv-langgraph` with a compatible Python interpreter.

Run the Python executable (script):

```powershell
.\.venv-langgraph\Scripts\python.exe .\langgraph_example_mutliagents.py --list
.\.venv-langgraph\Scripts\python.exe .\langgraph_example_mutliagents --self-test
.\.venv-langgraph\Scripts\python.exe .\langgraph_example_mutliagents --topic conditional_edges
.\.venv-langgraph\Scripts\python.exe .\langgraph_example_mutliagents.py --export-mermaid .\diagrams --json-output .\results.json
```

If your virtual environment is activated, `python` alone is fine.

## Jupyter Notebook in VS Code

1. Open `langgraph_example_mutliagents.ipynb` in VS Code.
2. Select the `.venv-langgraph` Python interpreter as its kernel.
3. Choose **Run All**. All examples run without a paid API or online model calls by default.
4. The two OpenAI examples are defined but are skipped unless you explicitly opt in.

For a notebook opened from another directory, run `%pip install -r /path/to/requirements_langgraph_example.txt` in a cell or install from a terminal first.

## Optional OpenAI examples

These examples invoke real hosted models and may incur API charges. Avoid committing `.env` to GitHub. Create a `.env` file in the project folder:

```dotenv
OPENAI_API_KEY=YOUR_KEY_HERE
OPENAI_MODEL=gpt-5.4-mini
```

Then in PowerShell:

```powershell
.\.venv-langgraph\Scripts\python.exe .\langgraph_example_mutliagents.py --with-llm
```

Or invoke just one:

```powershell
.\.venv-langgraph\Scripts\python.exe .\langgraph_example_mutliagent --topic llm_node
```

In the Jupyter notebook, set `RUN_OPENAI_EXAMPLES=1` in the kernel's environment before executing the cells; do not place real API keys directly into notebook source code. `.env` is loaded only if you enable the examples.

## Topic map

| # | Notebook topic | Python `--topic` |
|---|---|---|
| 1 | Introduction to LangGraph; first node | `greeting` |
| 2 | Nodes and `add_node`: factorial | `factorial` |
| 3 | LLM node (optional API) | `llm_node` |
| 4 | Normal edges | `normal_edges` |
| 5 | Conditional edges and routing | `conditional_edges` |
| 6 | State with `TypedDict` | `typed_dict_state` |
| 7 | State with Pydantic schema and validation | `pydantic_state` |
| 8 | Separate input/output state schemas | `input_output_schemas` |
| 9 | Private/intermediate state schema | `private_schema` |
| 10 | Reducer: overwrite | `reducer_overwrite` |
| 11 | Reducer: numerical addition | `reducer_numeric_add` |
| 12 | Reducer: list append | `reducer_list_append` |
| 13 | Reducer: custom unique-list merge | `custom_reducer` |
| 14 | Built-in `add_messages` reducer | `add_messages` |
| 15 | Built-in `MessagesState` graph, offline | `messages_state` |
| 16 | `MessagesState` with LLM (optional API) | `messages_state_llm` |
| 17 | Invoke a subgraph from a parent node (different schemas) | `subgraph_wrapper` |
| 18 | Add a compiled subgraph directly as a node (shared schema) | `subgraph_direct` |

## Improvements 

- No repeatedly overwritten global `graph`, `graph_builder`, or `GraphState` variables.
- Node functions return **partial state updates**, rather than mutating shared input state dictionaries.
- Pydantic square-output validation is made explicit (input 8 -> output 64 intentionally violates the upper bound of 10).
- `OPENAI_API_KEY` is never assigned `None` and is never printed.
- Both OpenAI examples are opt-in; offline `MessagesState` works without an API key.
- Mermaid diagram source (`.mmd`) can be exported **without** requiring a network-dependent `draw_mermaid_png()` operation.
- Scripts use `argparse` and `if __name__ == '__main__'` to avoid accidental execution when imported.
- Each exercise is named and isolated; a deterministic `--self-test` is included.
- The notebook includes all example definitions and can run on its own; it does not need to import the `.py` file.

## Notes and limitations

- This is a **Python script**, not a standalone binary `.exe`. `run_langgraph_windows.cmd` provides a double-click launcher on Windows if dependencies are installed.
- Python packages must be installed in your environment before running; `.cmd` can find `python` from your PATH.
- The optional OpenAI examples are not offline; an API key and internet access are required.
- Mermaid `.mmd` files can be displayed in VS Code Mermaid extensions or Mermaid Live; image rendering is optional.
  - Corrections and runnable packaging are additional work.
    ## License
    This project is licensed under the Apache License, Version 2.0.© 2026 Sudhanshu Shukla · Licensed under the Apache License, Version 2.0.
