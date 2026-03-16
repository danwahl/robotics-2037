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

- `metr.py` — core METR model (unified p50/p80 logistic fit with optional ceiling)
- `main.ipynb` — interactive analysis notebook
- `data/benchmark_results.yaml` — METR benchmark data (15+ models, Mar 2024–Nov 2025)
- `tests/test_metr.py` — 23 tests covering fit, sampling, success probability, ceiling behavior

## Current model (METR node)

One unified model fitting both h50 and h80 trends via log-linear regression:
- `success_probability(task_min, t_months)` uses sigmoid: `P = 1/(1 + (d/h50)^k)`
- `k` (steepness) derived from h80/h50 ratio
- 4x4 covariance matrix captures parameter uncertainty
- Optional logistic ceiling (passed at sample time, can be scalar or distribution)
- Ceiling applies to p50; p80 ceiling maintains the exponential-fit ratio

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

## Planned next steps (from conversation)

The conversation left off with the METR node and software speedup model working. Remaining work:
1. **Software speedup integration** — wire speedup model into notebook with squigglepy distributions
2. **Physical design node** — extend METR-style model to physical tasks (slower initial horizon, coupling to software speedup)
3. **Hardware capability node** — model what % of physical tasks robots can do (dexterity, mobility, endurance, etc.)
4. **Deployment/economics** — cost curves, manufacturing scale, regulation
5. **Safety dynamics** — offense/defense balance, open-source lag, diffusion risk
6. **Prediction market operationalization** — map model nodes to measurable/tradeable quantities

## User preferences

- Prefers step-by-step, building understanding at each stage
- Uses `uv` for all Python commands
- Comfortable with Python, probabilistic modeling, squigglepy
- Wants parsimonious models — simple first, add complexity only when justified
