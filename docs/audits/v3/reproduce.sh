#!/usr/bin/env bash
# Reproduces the protocol v3 audit analyses (docs/audits/v3/README.md).
#
# Read-only for the repository: every output goes to OUT (first argument, default a new
# temporary directory). Offline, no credentials. Needs `uv sync --locked` first.
# Not part of `make check` or CI: the v3 system is expected to FAIL most adversarial cases;
# this script documents known behaviour and reports drift from the recorded outputs.
set -euo pipefail
ROOT=$(git rev-parse --show-toplevel)
EV="$ROOT/docs/audits/v3/evidence"
OUT="${1:-$(mktemp -d)}"
mkdir -p "$OUT"
PY=(uv run --frozen --offline python -I)
cd "$ROOT"
echo "writing to $OUT"
"${PY[@]}" "$EV/scripts/sanity.py" "$ROOT" > "$OUT/sanity.txt"
"${PY[@]}" "$EV/scripts/recompute.py" "$ROOT" > "$OUT/recompute.txt"
"${PY[@]}" "$EV/scripts/run_cases.py" "$ROOT" "$EV/cases/cases.py" "$OUT/cases-results.json" > "$OUT/cases-summary.txt"
"${PY[@]}" "$EV/scripts/run_cases.py" "$ROOT" "$EV/cases/cases_posthoc.py" "$OUT/cases-results-posthoc.json" > "$OUT/cases-summary-posthoc.txt"
"${PY[@]}" "$EV/scripts/coupling.py" "$EV/cases/paraphrases.json" > "$OUT/coupling.txt"
"${PY[@]}" "$EV/scripts/ablate.py" "$ROOT" > "$OUT/ablate.txt"
"${PY[@]}" "$EV/scripts/metadata_ablation.py" "$ROOT" > "$OUT/metadata-ablation.txt"
"${PY[@]}" "$EV/scripts/robust.py" "$ROOT" > "$OUT/robust.txt"
"${PY[@]}" "$EV/scripts/query_probes.py" "$ROOT" > "$OUT/query-probes.txt"
"${PY[@]}" "$EV/scripts/template_overlap.py" > "$OUT/template-overlap.txt"
status=0
for f in cases-results.json cases-results-posthoc.json; do
  if cmp -s "$EV/outputs/$f" "$OUT/$f"; then echo "identical to recorded: $f"; else echo "DRIFT from recorded: $f"; status=1; fi
done
for f in "$EV"/outputs/*.txt; do
  name=$(basename "$f")
  if [ -f "$OUT/$name" ]; then
    if cmp -s "$f" "$OUT/$name"; then echo "identical to recorded: $name"; else echo "DRIFT from recorded: $name"; status=1; fi
  fi
done
exit $status
