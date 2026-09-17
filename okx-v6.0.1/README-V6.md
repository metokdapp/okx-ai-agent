# OKX AI Agent V6 Modular

V6 is a safe modularization baseline built from the working V5.4 code.

- `main.py`: tiny entrypoint only.
- `runtime_v54.py`: compatibility runtime preserving current behavior while migration proceeds.
- `jobs/job01_...py` through `jobs/job15_...py`: one stable module boundary per job.

Important: this V6 baseline intentionally does **not** rewrite trading logic. It preserves PAPER ONLY behavior and gives each job a separate file boundary. The next refactor can move each implementation out of `runtime_v54.py` one job at a time without changing `main.py`.
