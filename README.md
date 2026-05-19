# SmolVLA Evaluation Pipeline

End-to-end offline evaluation pipeline for **SmolVLA** (450M-parameter vision-language-action model) on the SO-100 robot manipulation dataset. Establishes a calibrated zero-shot baseline (mean L1 = 22.28) on 4,032 held-out frames, after diagnosing and correcting a normalization mismatch in the released checkpoint.

[Project portfolio →](https://bhalerao-ameya.vercel.app)

---

## TL;DR

| Metric | Value |
|---|---|
| Model | `lerobot/smolvla_base` (450M params) |
| Dataset | `lerobot/svla_so100_pickplace` (50 episodes, 19,631 frames) |
| Test split | Episodes 40-49 (4,032 frames) |
| Naive baseline (units mismatched) | mean L1 = 77.8 |
| **Calibrated baseline (units corrected)** | **mean L1 = 22.3** |
| Reduction from normalization fix | **65%** |

---

## What this project shows

1. **End-to-end VLA evaluation pipeline** — load SmolVLA, run inference against a real robot demonstration dataset, compute per-frame action prediction error.
2. **A subtle ML bug found and fixed** — the released SmolVLA checkpoint outputs actions in normalized space, but the SO-100 demonstration data is stored in raw joint-degree units. Naive comparison gives a misleading mean L1 of 77.8. Computing dataset-specific normalization stats from the train split (episodes 0-39) and un-normalizing predictions yields an honest baseline of 22.3.
3. **Episode-level train/test split** — splitting by episode rather than by frame prevents within-episode information leakage, a common methodological mistake in robotics ML.

---

## The normalization story (why the bullet exists)

The first run of the eval gave a mean L1 of **77.8** with a suspicious per-dimension pattern:

```
dim 0:  22.2     dim 3:  62.5
dim 1: 149.2     dim 4:  75.9
dim 2: 147.5     dim 5:   9.5
```

A random model would have produced roughly uniform error across dimensions. A ~15× spread between dim 1 (149.2) and dim 5 (9.5) signals a *structural* mismatch, not a learning failure.

Dumping predictions and ground truth side by side revealed it:

- **Ground truth** lived in raw joint-degree space (dim 1: 129-177°)
- **Predictions** lived in normalized space (dim 1: -0.23 to 1.57)

SmolVLA's released base checkpoint does not ship with dataset-specific normalization statistics. Computing mean/std on the train split (episodes 0-39) and un-normalizing predictions at eval time fixed the issue:

```python
ACTION_MEAN = train_actions.mean(axis=0)  # per-dim mean from train split
ACTION_STD  = train_actions.std(axis=0) + 1e-6
pred_unnorm = pred_norm * ACTION_STD + ACTION_MEAN
```

After the fix, mean L1 dropped to 22.3 and per-dimension error spread shrank to ~4× (9.1 to 38.5), consistent with a model that has structure but isn't yet fine-tuned on this dataset.

This bug would have made any downstream fine-tuning comparison meaningless, so identifying it early was important.

---

## Reproduce

**Requirements:** Google Colab Pro/Pro+ (any GPU ≥ 16GB) or local machine with similar specs. HuggingFace account.

```bash
pip install "lerobot[smolvla,peft]" pandas pyarrow tqdm
```

1. Set `HF_TOKEN` in Colab Secrets (or HuggingFace login locally)
2. Open `notebooks/week1_pipeline_setup.ipynb` — runs the full pipeline on one sample
3. Open `notebooks/week2_baseline_evaluation.ipynb` — runs the full 4,032-frame eval and saves predictions

Results are written to `/content/drive/MyDrive/vla-project/outputs/` by default (configurable).

---

## Repo structure

```
.
├── notebooks/
│   ├── week1_pipeline_setup.ipynb        # Load model, dataset, run one forward pass
│   └── week2_baseline_evaluation.ipynb   # Full test-split eval with normalization fix
├── outputs/
│   ├── predictions.parquet                # 4,032 (prediction, gt, per-frame L1) rows
│   ├── summary.json                       # Aggregate metrics
│   └── error_distribution.png             # Histogram + per-dim bar chart
└── README.md
```

---

## Methodology notes

- **Split policy.** Episodes 0-39 train (reserved for fine-tuning experiments), episodes 40-49 test. Episode-level prevents within-episode leakage.
- **Why L1 not success rate.** No simulator was available, so the standard offline behavior-cloning metric (L1 between predicted and demonstrated action) was used. L1 correlates with task success but is not equivalent — this caveat is appropriate to mention in any downstream claims.
- **Input adaptation.** SmolVLA was trained for 3-camera setups; SO-100 has 2 cameras (top, wrist). The policy's `empty_cameras` config handles the missing third camera by zero-padding. Images are resized from 480×640 to 256×256 to match the policy's expected input shape.
- **Action chunking.** SmolVLA predicts chunks of 50 actions; `policy.reset()` is called between frames to force a fresh prediction for each observation rather than serving cached actions from a previous chunk.

---

## Limitations and honest framing

- This is an **evaluation pipeline**, not a fine-tuning project. The baseline was established but no successful fine-tuning run was completed.
- Action chunking issues were encountered when attempting LoRA fine-tuning with the PEFT wrapper around SmolVLA's `forward()` — the released model's training path is not designed for direct invocation from outside `lerobot-train`. Documented but not solved.
- The L1 metric is a proxy. A true success-rate evaluation would require either a simulator (LIBERO/MetaWorld) or real SO-100 hardware.

---

## References

- **SmolVLA paper.** Shukor et al., 2025. [arXiv:2506.01844](https://arxiv.org/abs/2506.01844)
- **LeRobot library.** Cadene et al., 2024. [github.com/huggingface/lerobot](https://github.com/huggingface/lerobot)
- **Dataset.** [`lerobot/svla_so100_pickplace`](https://huggingface.co/datasets/lerobot/svla_so100_pickplace) — 50 episodes of SO-100 pick-and-place demonstrations

---

## Contact

Ameya Bhalerao — [portfolio](https://bhalerao-ameya.vercel.app)