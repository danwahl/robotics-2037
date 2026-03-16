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

## Glossary

- **h50**: task duration (minutes) at which AI has 50% success probability
- **h80**: task duration at which AI has 80% success (= h50 × h80_h50_ratio)
- **k**: sigmoid steepness, derived from h50/h80 gap — controls how sharply success drops with task length
- **METR horizon**: h50 for pure software/digital tasks, fitted from METR benchmark data
- **physical horizon**: h50 for embodied/physical tasks, by environment tier
- **ceiling**: logistic cap on h50 growth (METR: best observed to 30 days; structured: 60 days; unstructured: 18 hr)
- **SW speedup**: `1/(1 - sw_automation_fraction)` — how much faster software work gets done
- **sw_automation_fraction**: volume-weighted fraction of software work AI handles, via sigmoid over lognormal task distribution
- **hardware feasibility (HW)**: fraction of physical tasks robots can physically attempt (dexterity, mobility, battery, etc.) — independent of AI capability
- **ai_automation_fraction**: volume-weighted fraction of physical tasks AI can handle (ignoring hardware), via sigmoid over physical task distribution using physical horizon
- **physical automation fraction**: `hw_feasibility × ai_automation_fraction` — needs both capable hardware AND competent AI
- **physical speedup (Phys-S / Phys-U)**: `1/(1 - physical_automation_fraction)` — combined speedup for physical work
- **sw_coupling** (0.3): how much SW speedup accelerates the physical AI doubling rate
- **sw_design_coupling** (0.05): how much SW speedup accelerates hardware R&D (weaker — atoms are slow)
- **structured**: factory/warehouse — known layout, repetitive tasks, controlled environment
- **unstructured**: homes/construction/varied — novel layouts, diverse tasks, high entropy

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

## Key reference points

- Waymo: 10M sim miles/day (100:1 sim-to-real ratio), World Model built on Genie 3 generates novel scenarios. Classified as "structured" in our 2-tier model despite operating on public roads — the domain is narrow and well-mapped.
- Figure Helix 02: ~4 min autonomous household task (our unstructured anchor)
- Industrial picking (RightHand): 1,200 picks/hr at >99.5% (our structured anchor)

## Remaining work

1. **Deployment/economics** — Wright's Law cost curves, Bass diffusion for adoption
2. **Safety dynamics** — offense/defense balance, open-source lag, diffusion risk
3. **Prediction market operationalization** — map model nodes to measurable/tradeable quantities
4. **CadQueryEval integration** — user has https://danwahl.net/cadqueryeval for LLM design benchmarking

## User preferences

- Prefers step-by-step, building understanding at each stage
- Uses `uv` for all Python commands
- Comfortable with Python, probabilistic modeling, squigglepy
- Wants parsimonious models — simple first, add complexity only when justified
