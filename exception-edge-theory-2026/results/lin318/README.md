# LIN318 public result inventory

This directory separates primary measurements, negative controls, retained
closure witnesses, and representative tours. Unless a file explicitly says
otherwise, vertex identifiers in machine-readable artifacts are zero-based.

| Path | Role |
|---|---|
| `summary.json` | Machine-readable headline results and interpretation flags |
| `negative_controls/` | The 15-condition zero-base pilot; all conditions ended with `no_cycle_found` |
| `restricted_baseline/` | Matching lower and upper evidence for the weak-Delaunay value 42,231 |
| `closure_scan/closure_scan.csv` | Complete 512-candidate scan table |
| `closure_scan/edges/` | The 45 individual witnesses needed by the deterministic basin reproduction |
| `basin/` | Sanitized intervention tables for the published exploratory stages |
| `tours/` | Representative Hamiltonian tours paired with JSON provenance |

The 1,500-edge graph is an experimental search region consisting of 988
weak-Delaunay edges plus 512 separately generated candidates. It was not
generated from the threshold formula alone.

The 13 threshold-positive rows are safe lower-bound positives. The other 499
rows are inconclusive, not certified negatives. The values 42,118 and 42,108
use verified LKH closure witnesses as seeds and are exploratory results rather
than an LKH-free solver benchmark.

Run `python verify_public_results.py` from the release root for the compact
audit, or add `--full` to reproduce `42,210 -> 42,118 -> 42,108`.
