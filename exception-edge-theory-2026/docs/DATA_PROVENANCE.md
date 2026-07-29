# Data and Artifact Provenance

## Scope

This public folder is an allowlisted extraction from a larger local research
workspace. It contains only material needed to understand, test, or audit the
published theory and results. Unrelated analyses, temporary runs, duplicated
outputs, and local absolute paths were excluded.

## LIN318 benchmark

- Benchmark identity: TSPLIB `lin318`, `EUC_2D`, dimension 318.
- Source problem SHA-256 recorded by the experiment:
  `719d5340b7c550d5508be279320c1611a2877aeb5b36e05c07bd164ff3b7122c`.
- The source `.tsp` file is not redistributed here.
- The known optimum value 42,029 is used only for post-hoc reporting.
- No optimum tour is included in the compact input bundle or read by the
  threshold-to-basin reproduction.

## Candidate snapshots

| Snapshot | Edges | SHA-256 |
|---|---:|---|
| weak-Delaunay baseline edge set | 988 | `bf30c425903939b092212c95ecf7f41468729c574261cb96553ed30574cf25a9` |
| added-edge set | 512 | `5156b1514d5cc9a29f80cdea542200dca778feb556901025a4b4705191bc5293` |
| final candidate CSV | 1,500 | `aa75561a53100148eb69b45fef04bd690e5b0f7a0db9d8a573b85ecb1d3d4a69` |

The archived `static_local_b512` and `static_nearest_b512` candidate CSV files
are byte-identical. The public audit checks this explicitly; they must not be
treated as independent candidate baselines.

## Compact archive

`data/lin318_reproduction_inputs.zip` contains seven allowlisted members:

- weak candidate CSV and its blind seal;
- static-local candidate CSV and seal;
- static-nearest candidate CSV and seal;
- static-shortest saved 2-factor MILP audit.

The original Colab ZIP SHA-256 was
`06ab01ad86cfac20da267336e9f33be088f93a358f1863d117fc739eaa6e0d29`.
It had 401 entries and is not redistributed. The compact archive is a new
curated artifact whose hash is recorded in
`data/lin318_reproduction_manifest.json`.

The path-neutral `evaluation_pilot/benchmark_summary.csv` was retained as
`results/lin318/zero_base_pilot_summary.csv` so the 15 `no_cycle_found`
outcomes can be audited directly. Its SHA-256 is
`262745ca5c5abfeecd6d5ba5d2064552d205818c4889cf1c5e72967d4b0c3dd8`.

## Forced-closure artifacts

The original scan produced 512 independently verified result JSON files. The
Git tree retains:

- the complete 512-row `closure_scan.csv`;
- all 13 safe-positive witness JSON files;
- the 32 inconclusive witness JSON files used by the published top-32 probe.

The retained 45 witnesses are sufficient for the deterministic basin
reproduction. Omitted per-edge logs and LKH work files are not needed by that
runner.

## Restricted weak-Delaunay value

The value 42,231 is supported by:

- a valid integer lower bound from the SEC-MILP process; and
- a separately validated weak-Delaunay Hamiltonian tour of the same cost.

The audit JSON, SEC rounds, summary, and tour witness are under
`results/lin318/restricted_baseline/`.

## Exact-small artifacts

The public folder retains compact group/family summaries and selected figures
from the frozen Stage 2 and Stage 3 validation runs. Large point-level,
pair-level, and repeated development tables were excluded. The selected
tables preserve the reported denominators and allow the headline counts to be
checked without publishing tens of megabytes of redundant rows.

## Personal and secret data audit

The curated folder must contain no personal absolute home-directory paths,
API tokens, private keys, or personal email addresses. A final automated scan
is run before each publication commit.
