#!/usr/bin/env bash
# Requires curl + jq. Use only a demo student/database. Works with arbitrary field names.
set -euo pipefail
BASE=${KT_BASE_URL:-http://127.0.0.1:8000}
STUDENT=${KT_DEMO_STUDENT:-demo-$(date +%s)}
AUTH="Authorization: Bearer ${KT_API_TOKEN:-development}"
call() { curl --fail-with-body --silent --show-error -H "$AUTH" -H 'Content-Type: application/json' "$@"; }
call "$BASE/health"
INFO=$(call "$BASE/kt/model/info")
NAMES=$(jq -c '.kc_feature_names' <<< "$INFO")
# Missing values are legal; replace VECTOR with actual raw KC JSON for useful forecasts.
VECTOR=${KT_DEMO_VECTOR:-$(jq -c 'map(null)' <<< "$NAMES")}
CANDIDATES=$(jq -cn --argjson v "$VECTOR" '[{candidate_id:"next",kc_vector:$v}]')
PREDICT=$(jq -cn --argjson n "$NAMES" --argjson c "$CANDIDATES" '{kc_feature_names:$n,candidates:$c}')
STATE=$(call "$BASE/kt/students/$STUDENT/state")
REV=$(jq -r '.history_revision' <<< "$STATE")
call -X POST "$BASE/kt/students/$STUDENT/predict" -d "$PREDICT"
ATTEMPT="demo-$(date +%s)-$RANDOM"
OBSERVE=$(jq -cn --argjson n "$NAMES" --argjson v "$VECTOR" --argjson r "$REV" --arg id "$ATTEMPT:0" \
  '{kc_feature_names:$n,expected_revision:$r,event:{event_id:$id,kc_vector:$v,response:1}}')
call -X POST "$BASE/kt/students/$STUDENT/observe" -d "$OBSERVE"
# A complete retry is idempotent, including the original expected_revision.
call -X POST "$BASE/kt/students/$STUDENT/observe" -d "$OBSERVE"
REV=$(call "$BASE/kt/students/$STUDENT/state" | jq -r '.history_revision')
BATCH=$(jq -cn --argjson n "$NAMES" --argjson v "$VECTOR" --argjson r "$REV" --arg a "$ATTEMPT:batch" \
  '{kc_feature_names:$n,expected_revision:$r,attempt_id:$a,events:[
    {event_id:($a+":0"),kc_vector:$v,response:1},{event_id:($a+":1"),kc_vector:$v,response:0}]}')
call -X POST "$BASE/kt/students/$STUDENT/observe-batch" -d "$BATCH"
BRANCH=$(call -X POST "$BASE/kt/branches" -d "$(jq -cn --arg s "$STUDENT" '{student_id:$s}')" | jq -r '.branch_id')
for RESPONSE in 1 0 1; do
  STEP=$(jq -cn --argjson n "$NAMES" --argjson v "$VECTOR" --argjson r "$RESPONSE" \
    '{kc_feature_names:$n,kc_vector:$v,response:$r}')
  call -X POST "$BASE/kt/branches/$BRANCH/step" -d "$STEP"
done
call -X POST "$BASE/kt/branches/$BRANCH/predict" -d "$PREDICT"
call -X DELETE "$BASE/kt/branches/$BRANCH"
ROLLOUT=$(jq -cn --argjson n "$NAMES" --argjson v "$VECTOR" --argjson c "$CANDIDATES" --arg s "$STUDENT" \
  '{student_id:$s,kc_feature_names:$n,mode:"sample",sequence:[{kc_vector:$v},{kc_vector:$v}],
    seed:42,num_trajectories:4,forecast_candidates:$c,include_steps:true,retain_branch:false}')
call -X POST "$BASE/kt/rollouts/evaluate" -d "$ROLLOUT"
FORCED=$(jq -c '.mode="forced" | .num_trajectories=1 | .retain_branch=true | .sequence |= map(.+{response:1})' <<< "$ROLLOUT")
RESULT=$(call -X POST "$BASE/kt/rollouts/evaluate" -d "$FORCED")
printf '%s\n' "$RESULT"
BRANCH=$(jq -r '.branch.branch_id' <<< "$RESULT")
call -X POST "$BASE/kt/branches/$BRANCH/predict" -d "$PREDICT"
call -X DELETE "$BASE/kt/branches/$BRANCH"
call "$BASE/kt/students/$STUDENT/state"
