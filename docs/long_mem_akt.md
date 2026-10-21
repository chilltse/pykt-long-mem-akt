# LongMemAKT implementation and validation

The model uses `pykt.models.akt.AKT` directly for embeddings and local attention.
It does not add an independent PID vector to a KC vector.

## Original model components

AKT's variable `q_data` is the concept/KC ID; `pid_data` is the problem ID.
Its question representation is

`q_t = E_c(c_t) + difficulty(pid_t) * E_c_diff(c_t)`.

With no PID vocabulary, it is simply `E_c(c_t)`. With `separate_qa=False`,
the interaction representation follows the repository implementation:

`v_t = E_c(c_t) + E_r(r_t) + difficulty(pid_t) * (E_c_diff(c_t) + E_r_diff(r_t))`.

`separate_qa=True` uses the existing AKT joint concept/response embedding instead.
The same representations feed local AKT and memory addressing/writing. There is
no second question embedding table for long memory. Question/addressing inputs
never contain response labels. This version accepts the existing integer KC/PID
schema; arbitrary structured feature encoders are not implemented.

`AKT.encode_embeddings` extracts the existing code for reuse. `AKT.reset` now
zeros only PID difficulties; the previous shape-based reset accidentally zeroed
unrelated tensors when their first dimension equalled `num_pid + 1`. Attention
allocates tensors on the input's device instead of a global CUDA/CPU choice.

Local predictions reuse AKT's interaction encoder, question encoder, retriever,
learned `gammas`, and attention-derived distance decay. No extra input shift or
absolute position embedding is applied to local AKT. Its first retriever row has
zero historical attention, but its residual retains the current question. Valid
tokens are gathered per sample before calling original AKT, then scattered back;
padding cannot change its distance calculation. This prioritizes correctness
over batched throughput. The independent write encoder uses positions and causal
attention including the diagonal, after outcomes are observed.

Repository DKVMN receives concept IDs from the standard model dispatch. Its
`k_emb_layer` and joint `v_emb_layer` distinguish the query from the observed
interaction; neither implies that a second independent PID vector is required.
LongMemAKT follows its learned shared keys, trainable initial values, weighted
read, and sigmoid-erase/tanh-add structure and initialization. Addressing remains
projected/scaled as required by the guide. Writing remains the guide's slot-summary
block update, **not** original DKVMN's per-interaction update.

## Block order and shapes

For batch B, block length T, dimension D, slots S, value dimension V:

1. Encode questions/interactions `[B,T,D]` with original AKT embeddings.
2. Compute AKT short states `[B,T,D]` and addresses `[B,T,S]`.
3. Read the immutable entry values `[B,S,V]` to get `[B,T,V]`.
4. Fuse projected long/short states with a scalar or vector gate and predict.
5. Independently encode completed interactions and aggregate `[B,S,V]` summaries
   plus `[B,S]` soft counts, then apply one block write.

Training groups default to two blocks. Within a group the next prediction loss
trains the preceding Writer and write encoder. At the last block, save the
pre-write values and summary without writing. After the optimizer step, detach
both; the next group executes this pending write with current Writer parameters.
That boundary loss trains the Writer but not the previous group's detached
encoder. New students start from a differentiable copy of `memory_init`; the
initial parameter is not detached. Runtime storage is independent per student.
The final training block's write is unnecessary without a future target; evaluation
does perform it to produce the final state. States reset every student/epoch.

Time stamps are loaded in float64 and differenced before casting. `time_unit=1000`
means input milliseconds / 1000 = seconds. Missing/negative/nonmonotonic timestamps
use event distance and are counted separately in diagnostics. `time_weight=0`
disables weighting even when event fallback is forced.

## Evaluation and integration

The dedicated dataset reads the existing original student split CSVs and respects
fold assignments. It rejects duplicate student rows rather than inventing a block
order. The runner explicitly requires one student per batch; different students
are never flattened together.

Default `score_protocol=legacy` reconstructs the **concept-level nonwindowed**
selection from the data config's `maxlen` and `min_seq_len`: segment starts and
too-short tails are not scored. Memory blocks are independent of that selection;
all permitted interactions still supply history. `score_protocol=student` is an
explicit different protocol: score all allowed interactions except the student's
first, including later block starts and short tails. This is appropriate for
investigating memory at boundaries, but scores cannot be compared as if the target
set were identical. Both protocols emit predictions for every interaction with a
`scored` column. Existing `selectmasks`, when present, are also respected.

There is no window loader or window metric for this model. Question-level
multi-concept fusion evaluation is not implemented; these results must not be
reported as the existing question-level/window benchmark. Original folds and
binary labels are unchanged. The standard predictor accepts `--student_id` to
select a single student's complete held-out trajectory.

For an offline final short block, write it once at sequence end. If pausing an
online stream mid-block, retain its entry memory and buffered interactions; do
not finalize it or reset its position until full or explicitly ended. The current
runner handles complete offline sequences; persisted online sessions are not
provided.

## Configuration and commands

Defaults: `block_size=200`, `tbptt_blocks=2`, `num_slots=32`, `d_model=128`,
`num_attn_heads=8`, `n_blocks=2`, `d_ff=256`, `l2=1e-5`, `ablation=gate`,
`write_encoding=causal`, `score_protocol=legacy`.

`short` uses the original AKT output head on the unmodified AKT short states and
is numerically identical given the same parameters. `long` uses only the long
branch plus current question. `concat` actually concatenates question, long and
short vectors into a `3*D` prediction head. `gate` uses the dynamic mixture.
`write_encoding=independent` disables the write sequence encoder and position
embedding; `time_weight=0` disables temporal weighting.

From the project root, with the pykt environment active:

```bash
python -m pytest -q tests/test_long_mem_akt.py
cd examples
PYTHONPATH=.. python wandb_long_mem_akt_train.py --dataset_name assist2009 --fold 0 --num_epochs 1
PYTHONPATH=.. python wandb_long_mem_akt_train.py --dataset_name assist2009 --ablation short --num_epochs 1
PYTHONPATH=.. python wandb_long_mem_akt_train.py --dataset_name assist2009 --time_weight 0 --write_encoding independent --num_epochs 1
PYTHONPATH=.. python wandb_predict.py --save_dir saved_model/ACTUAL_RUN_DIRECTORY --student_id ACTUAL_UID --use_wandb 0
```

Training writes `long_mem_metrics.jsonl`: AUC, log loss, Brier, gate quantiles,
slot access fractions, memory update norm, exclusive Writer/write-encoder gradient
norms, interactions/sec and CUDA peak allocation. The recurring-KC subset uses
exact KC ID equality: first occurrence in the current block, but seen in an earlier
block. Subset counts and Brier scores are reported; an empty subset has no score.
The default legacy mask normally excludes block starts when block/data lengths
match; use explicit student scoring to investigate those positions. Diagnostics
are not calibrated mastery or causal branch-contribution estimates.

The corrected architecture is incompatible with earlier provisional LongMemAKT
checkpoints; retrain them. Synthetic tests verify numerical behavior and real
training/prediction entrypoints, not improved accuracy on a real dataset. Only
CPU tests have been run; production GPU cost and task performance remain unmeasured.
