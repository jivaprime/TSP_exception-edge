# TSP Exception-Edge Research

This repository now contains two clearly separated research stages:

1. **Early exploratory solver (2024)** — the original geometry-guided PPO
   implementation and its logs. See [English](README_EN.md) or
   [한국어](README_KR.md).
2. **Rigorous exception-edge study (2026)** — formal definitions, exact-small
   experiments, safe certificates, LIN318 scale-up code, compact audit data,
   and tests. Start at
   [`exception-edge-theory-2026/README.md`](exception-edge-theory-2026/README.md)
   or its
   [한국어 안내](exception-edge-theory-2026/README_KO.md).

The 2026 work is not presented as a retrospective proof of every claim made by
the early solver. It is a new, explicitly auditable theory-and-experiment
layer built around baseline-relative exception edges, cut structure,
Hamiltonian closure, and safe lower-bound certification.

The repository-level license is Apache-2.0. Third-party programs and benchmark
instances are not relicensed by this repository.
