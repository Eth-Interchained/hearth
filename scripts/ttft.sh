#!/usr/bin/env bash
# ttft.sh — measure time-to-first-token the way the operator feels it.
#
#   ./scripts/ttft.sh MODEL [PROMPT_TOKENS] [RUNS]
#
# Three numbers per run, all from llama-server's own `timings` object so there
# is no client-side clock to argue with:
#
#   prompt_ms   prefill — this IS time-to-first-token for a long prompt
#   cached      how many prompt tokens the slot's KV cache already had
#   gateway_ms  the SAME request through hearth's gateway minus straight at the
#               child — hearth's own contribution to TTFT, isolated
#
# Runs the identical prompt RUNS times so the cache-hit case is visible: with
# --cache-reuse set, run 2+ should show `cached` ≈ prompt length and prompt_ms
# collapsing. If run 2 re-prefills everything, the flag is not reaching the
# child (`ps -ww -eo args | grep llama-server` and look for it).
#
# Bash 3.2 + curl + python3 — nothing else.

set -u
MODEL="${1:?usage: ttft.sh MODEL [PROMPT_TOKENS=2000] [RUNS=3]}"
TOKENS="${2:-2000}"
RUNS="${3:-3}"
PORT="${HEARTH_PORT:-11434}"

# Find the child straight from hearth's residency — never guess a port.
ENDPOINT="$(curl -s "http://127.0.0.1:$PORT/residency" \
  | python3 -c 'import sys,json
m=sys.argv[1]
for s in json.load(sys.stdin).get("models",[]):
    if s.get("model")==m and s.get("endpoint"): print(s["endpoint"]); break' "$MODEL")"
[ -n "$ENDPOINT" ] || { echo "ttft: $MODEL has no endpoint in /residency (not ready, or not declared)" >&2; exit 1; }

# A deterministic prompt of roughly TOKENS tokens: numbered lines tokenize at a
# stable ~7 tokens each (measured: 2000 -> 3360 at //4, so //7), so the SAME text is sent every run — that is the point.
PROMPT="$(python3 -c 'import sys
n=int(sys.argv[1])//7
print("Read the following and answer: what is line 7?\n"+"\n".join(f"line {i}: alpha beta gamma" for i in range(n)))' "$TOKENS")"
BODY="$(python3 -c 'import sys,json
print(json.dumps({"prompt":sys.stdin.read(),"n_predict":1,"cache_prompt":True,"temperature":0}))' <<<"$PROMPT")"

timing() { # $1 = base url
  curl -s "$1/completion" -H 'content-type: application/json' -d "$BODY" \
  | python3 -c 'import sys,json
d=json.load(sys.stdin); t=d.get("timings",{})
print(f"{t.get(\"prompt_n\",0)} {t.get(\"prompt_ms\",0):.0f} {t.get(\"cache_n\",0)}")'
}

printf '%s  endpoint %s  ~%s prompt tokens\n' "$MODEL" "$ENDPOINT" "$TOKENS"
printf '%-4s %-9s %-10s %-7s %-11s\n' run direct_ms prompt_n cached gateway_ms
i=1
while [ "$i" -le "$RUNS" ]; do
  read -r pn pms cn <<<"$(timing "http://$ENDPOINT")"
  # Gateway path: wall clock around the same request via 11434, minus the
  # direct prefill just measured. What is left is routing + proxy + queueing.
  t0=$(python3 -c 'import time;print(time.time())')
  curl -s "http://127.0.0.1:$PORT/v1/completions" -H 'content-type: application/json' \
    -d "$(python3 -c 'import sys,json
b=json.loads(sys.stdin.read()); b["model"]=sys.argv[1]; b["max_tokens"]=1; del b["n_predict"]; print(json.dumps(b))' "$MODEL" <<<"$BODY")" >/dev/null
  gw=$(python3 -c 'import sys,time;print(f"{(time.time()-float(sys.argv[1]))*1000-float(sys.argv[2]):.0f}")' "$t0" "$pms")
  printf '%-4s %-9s %-10s %-7s %-11s\n' "$i" "$pms" "$pn" "$cn" "$gw"
  i=$((i+1))
done
