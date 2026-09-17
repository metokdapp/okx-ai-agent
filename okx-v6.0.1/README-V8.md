# OKX AI Agent V8 — PAPER ONLY

V8 is an in-place upgrade of the recovered V6.0.1 baseline (`c2461c6`) and deliberately preserves its 15-job AUTO flow and compact Telegram report layout.

## V8 changes
- Stronger Technical / Orderbook / Derivatives specialist instructions.
- Adversarial Critic compares specialists, seeks counter-evidence and challenges confidence.
- Master must consume Critic feedback and may use compact prior-cycle memory/reflection as advisory context.
- Structured in-RAM agent memory is recorded after Validator/Risk/PAPER so later cycles see the actual outcome, not only the AI opinion.
- Deterministic Reflection records lessons after each cycle; it cannot change risk or execution.
- Daily loss limit is disabled as requested. Risk per trade remains 1%, max one position, minimum confidence 70%, SL 1.5 ATR, reward/risk 2:1.
- Spot SELL closes an existing PAPER BUY only. SELL with no BUY is blocked; no short is opened.
- Python Validator and Risk Engine remain authoritative. AI cannot change money, position size, risk %, SL/TP, exposure or execution rules.

## Important
Memory/reflection and PAPER account state are still RAM-resident and reset on process restart. This release does not add persistent storage.
