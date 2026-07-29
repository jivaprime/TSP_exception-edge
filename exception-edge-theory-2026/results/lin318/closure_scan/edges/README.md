# Retained forced-closure witnesses

The original LIN318 scan produced 512 verified per-edge result artifacts. This
public tree retains 45 of them:

- all 13 candidates whose safe gain lower bound is positive; and
- the 32 nonpositive/inconclusive candidates used by the published top-32
  anchored probe, ranked by closure upper bound.

Each directory name is the zero-based edge `u_v`; its `result.json` contains
the forced closure tour, cost fields, and independent verification flags. The
complete outcomes for all 512 candidates remain in the parent
`closure_scan.csv`.

The files intentionally remain separate rather than being merged into JSONL.
The reproduction runner indexes them by edge, independently revalidates the
selected witness, and records which artifact supplied each seed. Omitted LKH
work directories and the other 467 per-edge JSON files are not needed for the
published deterministic reproduction.
