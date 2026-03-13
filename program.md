# autoresearch-sdxl: Autonomous Parallel LoRA Training Loop

You are an autonomous researcher. Your goal: maximize CLIP image-image similarity between generated images and a set of reference images by tuning SDXL LoRA training hyperparameters on Modal.

## How It Works

You write `batch.yaml` with 3-5 experiment variants → they run in parallel on separate A100 GPUs → you read all scores at once → you keep the best → you write the next batch → loop.

This is **3-5x faster** than testing one config at a time.

## Files You Read

- `results.tsv` — all past experiments and scores
- `reasoning.md` — your research journal (last ~10 entries)
- `config.yaml` — current best hyperparameter configuration (baseline for next batch)
- `batch.yaml` — current batch of experiments

## Files You Write

- `batch.yaml` — define 3-5 experiments to run in parallel
- `config.yaml` — update with the best config after each batch
- `reasoning.md` — append your hypothesis for this batch
- `results.tsv` — record all experiment results

## config.yaml Search Space

| Parameter | Type | Description |
|-----------|------|-------------|
| rank | int | LoRA rank (4–128). Controls adapter capacity. |
| alpha | int | LoRA alpha. Usually equals rank. |
| lr | float | Learning rate for optimizer. |
| train_batch_size | int | Training batch size. |
| gradient_accumulation_steps | int | Effective batch = batch_size × this. |
| max_train_steps | int | Number of training iterations. |
| lr_scheduler | string | Schedule: cosine, linear, constant. |
| lr_warmup_steps | int | Warmup steps before full LR. |
| guidance | float | Classifier-free guidance scale for eval images. |
| trigger_word | string | Token that activates the LoRA style (e.g., "cybrn"). |
| caption_template | string | Template for training captions. Use {trigger} placeholder. |

## batch.yaml Format

Each experiment inherits from `config.yaml` and overrides only what changes.
Every entry MUST have a unique `tag`.

```yaml
experiments:
  - tag: rank8
    rank: 8
    alpha: 8
  - tag: rank16
    rank: 16
    alpha: 16
  - tag: rank32
    rank: 32
    alpha: 32
```

## Setup (one-time, before the loop)

1. Create branch: `git checkout -b autoresearch-sdxl/<tag>`
2. Verify `prepare.py` has been run (reference embeddings on Modal Volume)
3. Initialize `results.tsv` with header:
   ```
   commit	clip_centroid	clip_nn	stddev	neg_ctrl	vram_gb	train_sec	status	description
   ```
4. Run baseline single experiment:
   ```bash
   modal run train.py > run.log 2>&1
   ```
5. Record baseline in `results.tsv` and `reasoning.md`
6. Update `config.yaml` with baseline results

## Loop

```
LOOP FOREVER:

1. READ STATE
   • results.tsv (all past experiments)
   • reasoning.md (recent entries — last ~10)
   • config.yaml (current best config)

2. REASON + PROPOSE BATCH
   • What has worked? What hasn't?
   • What are 3-5 high-leverage things to try in parallel?
   • Design experiments that explore DIFFERENT dimensions
     (e.g., one rank change, one LR change, one steps change)
     not variations of the same parameter
   • Write hypothesis in reasoning.md (append, keep entries short)

3. WRITE batch.yaml
   • 3-5 experiments, each overriding config.yaml
   • Give each a clear tag describing the change
   • Stage + commit:
     git add batch.yaml reasoning.md
     git commit -m "batch: <description>"

4. RUN
   modal run train.py --batch > run.log 2>&1

5. READ RESULTS
   grep "best_clip_centroid:" run.log
   grep "BATCH RESULTS" -A 20 run.log
   If empty → tail -50 run.log (crash or timeout)

6. DECIDE
   Look at the BEST result from the batch.
   Compare to the current best in results.tsv.
   Minimum delta: 0.005 to count as improvement.

   If best improved (delta >= 0.005):
     → Update config.yaml with the winning config
     → Log ALL results to results.tsv (winner: keep, others: discard)
     → git add config.yaml results.tsv
     → git commit -m "results: keep <winning tag> — <description>"

   If no experiment improved:
     → Log all to results.tsv (all: discard)
     → config.yaml stays unchanged
     → git add results.tsv
     → git commit -m "results: discard batch — <description>"

   If crash/timeout on some:
     → Log crashes to results.tsv (status: crash)
     → Still evaluate the ones that succeeded
     → git add results.tsv
     → git commit -m "results: partial batch — <description>"

7. GOTO 1
```

## Batch Design Principles

**Maximize information per batch:**
- Don't test 5 values of the same parameter — that's a grid sweep, not research
- Test across different dimensions: one rank change + one LR change + one steps change + one caption change
- Include one "safe" experiment (small change) and one "bold" experiment (large change)
- If a direction shows promise, next batch explores that neighborhood

**Good batch example:**
```yaml
experiments:
  - tag: rank32           # test higher capacity
    rank: 32
    alpha: 32
  - tag: lr_half          # test lower learning rate
    lr: 5e-5
  - tag: steps_1500       # test more training
    max_train_steps: 1500
  - tag: caption_detail   # test richer captions
    caption_template: "a detailed {trigger} style painting, renaissance oil painting with cyberpunk neon elements, "
```

**Bad batch example (wastes GPUs):**
```yaml
experiments:
  - tag: rank8
    rank: 8
  - tag: rank12
    rank: 12
  - tag: rank16
    rank: 16
  - tag: rank24
    rank: 24
```

## Expected Output Format (batch mode)

```
======================================================================
BATCH RESULTS — 4 experiments
======================================================================

  [rank32] clip_centroid=0.862  nn=0.831  neg=0.305  rank=32  lr=0.0001  steps=1000  train=920s
  [lr_half] clip_centroid=0.845  nn=0.818  neg=0.310  rank=16  lr=5e-05  steps=1000  train=910s
  [steps_1500] clip_centroid=0.871  nn=0.840  neg=0.312  rank=16  lr=0.0001  steps=1500  train=1380s
  [caption_detail] clip_centroid=0.858  nn=0.825  neg=0.302  rank=16  lr=0.0001  steps=1000  train=925s

──────────────────────────────────────────────────────────────────────
BEST: [steps_1500] clip_sim_centroid=0.871000
  rank=16, alpha=16, lr=0.0001, steps=1500
──────────────────────────────────────────────────────────────────────

best_tag: steps_1500
best_clip_centroid: 0.871000
best_config: {"rank": 16, "alpha": 16, "lr": 0.0001, "max_train_steps": 1500, ...}
```

## results.tsv Schema

```
commit	clip_centroid	clip_nn	stddev	neg_ctrl	vram_gb	train_sec	status	description
a1b2c3d	0.847	0.812	0.018	0.312	63.5	900	keep	baseline (default config)
e4f5g6h	0.862	0.831	0.015	0.305	63.5	920	keep	batch1: rank32 (winner)
i7j8k9l	0.845	0.818	0.022	0.310	63.5	910	discard	batch1: lr_half
m1n2o3p	0.871	0.840	0.012	0.312	63.5	1380	keep	batch2: steps_1500 (winner)
```

## Fallback to Single Mode

If you want to test ONE specific hypothesis (e.g., verify a fix after a crash), use single mode:
```bash
# Edit config.yaml directly
modal run train.py > run.log 2>&1
```

## Crash Recovery

If starting a new session, read `results.tsv` and `git log` to determine the last completed batch. Resume from step 1.

## NEVER STOP

Once the loop begins, do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human may be asleep or away. You are autonomous. If you run out of ideas, re-read reasoning.md, try combining near-misses, try more radical changes. The loop runs until the human interrupts you.

## Strategy Guidance (suggestions, not rules)

- **Batch 1 (broad sweep):** Rank (8, 16, 32, 64) — find the right capacity.
- **Batch 2 (refine winner + explore):** Refine best rank ± neighbors, try different LR, try more steps.
- **Batch 3-4:** Explore LR scheduler, warmup, alpha ratio around the winning config.
- **Batch 5+:** Caption strategies, guidance scale, combinations. Captions are often the biggest lever.
- **Later batches:** Fine-grained refinement around the best known config.

Adapt based on results. If rank barely matters but LR is highly sensitive, spend more batches on LR. If captions dominate everything, pivot early.

## Cadence

~15-20 min per batch (all experiments run in parallel). ~3-4 batches/hour. Each batch tests 3-5 configs. **~12-20 experiments/hour.**

~100-150 experiments overnight (8 hours).

## Cost

~$3.74/hr per A100. With 4 parallel GPUs: ~$15/hr during batch execution.
Overnight (8 hours, ~70% GPU utilization): ~$80-100.
