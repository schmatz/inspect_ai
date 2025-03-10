# Inspect AI Codebase Guidelines

## Overview
Inspect is an open-source framework for large language model evaluations created by the UK AI Security Institute. It provides built-in components for:
- Prompt engineering
- Tool usage
- Multi-turn dialog
- Model graded evaluations

Inspect evaluations have three main components:
1. **Datasets**: Collections of labeled samples with inputs and targets
2. **Solvers**: Components that evaluate inputs and produce results (can be chained together)
3. **Scorers**: Evaluate solver outputs using text comparisons, model grading, or custom schemes

## Codebase Structure

### Main Components

- **Dataset** (`src/inspect_ai/dataset/`): Sample definition, data loaders (JSON, CSV, HF)
- **Model** (`src/inspect_ai/model/`): LLM provider interfaces, 20+ supported providers 
- **Solver** (`src/inspect_ai/solver/`): Evaluation strategies (agents, chains, tool users)
- **Scorer** (`src/inspect_ai/scorer/`): Output evaluation (pattern matching, model grading)
- **Tools** (`src/inspect_ai/tool/`): Function-calling interface (bash, python, web browser)
- **Approval** (`src/inspect_ai/approval/`): Human-in-the-loop tool call review

### Supporting Modules

- **CLI** (`src/inspect_ai/_cli/`): Command-line interface
- **Display** (`src/inspect_ai/_display/`): Terminal UI components
- **Eval** (`src/inspect_ai/_eval/`): Core evaluation framework
- **Log** (`src/inspect_ai/log/`): Logging and result storage
- **Util** (`src/inspect_ai/util/`): Utilities including sandboxing
- **View** (`src/inspect_ai/_view/`): Web-based visualization

### Key Implementation Files

- `_eval/eval.py`: Main evaluation entry point
- `_eval/task/task.py`: Task definition and execution
- `model/_model.py`: Core model interface
- `solver/_solver.py`: Solver protocol and decorators
- `scorer/_scorer.py`: Scoring protocol
- `tool/_tool.py`: Tool definition
- `dataset/_dataset.py`: Dataset handling

## Commands
- Install dev: `pip install -e ".[dev]"`
- Lint: `make check` or `ruff check --fix && ruff format`
- Typecheck: `mypy --exclude tests/test_package src tests`
- Run all tests: `pytest`
- Run single test: `pytest tests/path/to/test_file.py::test_function_name`
- Run tests by pattern: `pytest -k "pattern"`
- Test with flags: `pytest --runapi`, `pytest --runslow`

## Important Development Tips

1. ALWAYS run Python related commands in the virtual environment found in .venv. If it doesn't exist, create it.

## Code Style
- **Imports**: Grouped by stdlib, third-party, local; alphabetized within groups
- **Types**: Comprehensive typing required; disallow untyped definitions; strict mypy
- **Naming**: snake_case (variables, functions), PascalCase (classes), _leading_underscore (private)
- **Docstrings**: Google-style docstrings with parameter descriptions
- **Formatting**: Uses ruff-format (based on black), 4-space indentation
- **Error handling**: Specific exception types, context managers, detailed error messages
- **Structure**: Modules prefixed with "_" are internal implementation details

Enforce with: `pre-commit install` to activate automated checks