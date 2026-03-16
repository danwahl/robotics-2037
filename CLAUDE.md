# Robotics 2037

## Project overview

Quantitative forecasting model for AI + robotics capabilities, inspired by "AI 2027". Core question: **When might AI systems design and build physical things with minimal human involvement?**

## Tech stack

- Python 3.13, managed with `uv`
- `squigglepy` (custom fork) for probabilistic modeling
- `pandas`, `matplotlib`, `pyyaml` for data/viz
- `pytest` + `ruff` for dev tooling
- Jupyter notebook (`main.ipynb`) for interactive analysis

## Commands

- `uv run pytest tests/` — run tests
- `uv run ruff check .` — lint
- `uv run jupyter lab` — launch notebook

## Project structure

- `metr.py` — METR model (unified p50/p80 logistic fit with optional ceiling)
- `physical.py` — physical AI model (physical horizon + hardware capability + combined speedup)
- `main.ipynb` — interactive analysis notebook
- `data/benchmark_results_1_0.yaml` — METR v1.0 benchmark data (original)
- `data/benchmark_results_1_1.yaml` — METR v1.1 benchmark data (current, from https://metr.org/time-horizons/)
- `tests/test_metr.py` — 23 tests for METR model
- `tests/test_physical.py` — 16 tests for physical model

## Model structure

### METR node (metr.py)

Log-linear fit to METR benchmark data (SOTA models, 2023+):
- `success_probability(task_min, t_months)` uses sigmoid: `P = 1/(1 + (d/h50)^k)`
- `k` derived from h80/h50 ratio; 4x4 covariance matrix for uncertainty
- Optional logistic ceiling (scalar or distribution); p80 ceiling maintains ratio
- Software speedup = `1/(1 - frac_automated)` where frac is probability-weighted over lognormal task distribution

### Physical AI node (physical.py)

Two sub-models combined multiplicatively:

**PhysicalHorizon** — METR-equivalent for embodied tasks, 2 environment tiers:
- Structured (factory/warehouse): h50 starts ~60 min, doubles every ~8 months
- Unstructured (homes/varied): h50 starts ~4 min, doubles every ~14 months
- Software coupling: fraction of SW speedup exponent transfers to physical improvement

**HardwareCapability** — fraction of tasks that are hardware-feasible:
- Structured starts ~0.75, unstructured ~0.20
- Logistic growth toward 1.0, accelerated by software-driven design improvements

Combined: `physical_speedup = 1/(1 - hw_feasibility * ai_automation_fraction)`

## Conversation history (Chroma MCP)

Two relevant conversations in `claude_conversations` collection:

### "Agentic AI and robotics integration" (71 turns, Jan 2–9 2026)
Main development conversation. Key progression:
1. **Turns 0–15**: Exploration of AI+robotics landscape, causal model sketch
2. **Turns 16–21**: Formalized prediction targets (job equivalents, safety implications), causal model with nodes: METR horizon -> software speedup -> hardware design -> robot capability
3. **Turns 22–25**: Physical design node — hardware capabilities (dexterity, mobility, etc.) + training data improvement rates; two-node decomposition (hardware capability + physical AI horizon)
4. **Turns 26–45**: Implementation — squigglepy setup, METR data parsing, project scaffolding (pyproject.toml, pytest config)
5. **Turns 46–64**: Software speedup model — lognormal task distribution, volume-weighted automation fraction, unified p50/p80 sigmoid model
6. **Turns 65–70**: Logistic ceiling added (exponential early, saturates at ceiling), Gompertz explored and rejected, primer written for design benchmarking side quest

### "Benchmarking AI design capabilities for physical systems" (Jan 9 2026)
Side quest research conversation. Key findings:
- CadQuery is the preferred target for LLM-generated CAD (Python-based)
- Text2CAD (NeurIPS 2024 Spotlight): 170K models, 660K text annotations, eval code available
- ABC dataset: 1M CAD models
- No METR-equivalent for physical design exists yet
- Proposed hierarchy: compiles -> geometry matches -> passes FEA simulation -> manufacturable
- FreeCAD + CalculiX for open-source simulation-in-the-loop validation

## Remaining work

1. **Ceiling for physical horizons** — structured h50 grows very fast; may need logistic ceiling like METR
2. **Uncertainty/Monte Carlo for physical model** — currently deterministic; add sampling like METR model
3. **Deployment/economics** — Wright's Law cost curves, Bass diffusion for adoption
4. **Safety dynamics** — offense/defense balance, open-source lag, diffusion risk
5. **Prediction market operationalization** — map model nodes to measurable/tradeable quantities
6. **CadQueryEval integration** — user has https://danwahl.net/cadqueryeval for LLM design benchmarking

## User preferences

- Prefers step-by-step, building understanding at each stage
- Uses `uv` for all Python commands
- Comfortable with Python, probabilistic modeling, squigglepy
- Wants parsimonious models — simple first, add complexity only when justified
