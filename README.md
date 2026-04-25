# Proto-Adapter + CoOp Hybrid for Few-Shot Image Classification

A hybrid few-shot classifier built on a frozen CLIP ViT-B/16 backbone. It combines [CoOp](https://arxiv.org/abs/2109.01134) (Context Optimization — learnable prompt vectors) with a training-free [proto-adapter](https://arxiv.org/abs/2111.03930) (mean-feature class prototypes), and routes between them at inference time using a confidence-based switch. The goal is to specialize on seen classes without giving up CLIP's zero-shot generalization to unseen ones.

Originally built as a deep-learning course project with **Christian Li Sivertsen** and now being polished and extended.

## Results

Evaluated on the Flowers102 base/novel split (51 base classes, 51 novel classes, 10-shot). Hybrid numbers are reported as **mean ± std over three seeds**.

| Model                              | Base Acc          | Novel Acc       | Harmonic Mean    |
| ---------------------------------- | ----------------- | --------------- | ---------------- |
| Zero-shot CLIP (baseline)          | 71.33%            | **78.24%**      | 74.63%           |
| **CoOp + Proto-Adapter (ours)**    | **91.47% ± 1.61** | 75.83% ± 0.77   | **82.92% ± 1.10** |

A **+20.14** point gain on base classes for a **−2.41** point cost on novel classes — an **8.29** point improvement on the harmonic mean used by CoOp/CoCoOp to compare base/novel methods.

## How it works

At inference, every test image is scored by both branches:

1. **CoOp branch.** Replaces the hand-crafted prompt `"a photo of a {class}"` with a learnable context vector trained on the base classes. The CLIP text encoder produces class embeddings from this learned prompt, which are then matched against image features.
2. **Proto-adapter branch.** For each base class, the L2-normalized mean image feature over the few-shot training images becomes the class prototype. Classification is nearest-prototype.

The two branches are mixed by a weight `alpha`. A **confidence-based switch** then decides whether to trust the mix or fall back to the CoOp prompt alone: if the prototype softmax confidence is below `conf_threshold`, the model uses the CoOp branch only. This keeps the proto-adapter — which has no prototypes for novel classes — out of decisions where it is unlikely to help.

## Hyperparameter selection

Hyperparameters are picked by a two-stage search on the **validation split**, not the test split:

- **Stage 1** is a random search over training-time hyperparameters (`n_ctx`, `lr`, `epochs`). Each draw runs CoOp training once.
- **Stage 2** is a dense grid sweep over inference-time hyperparameters (`alpha`, `conf_threshold`) on top of each Stage-1 artifact. No retraining.

The winner is the configuration with the highest **harmonic mean of base and novel val accuracy**, then re-trained across multiple seeds for the final test report.

## Setup

```bash
git clone https://github.com/HunterBaker01/proto_coop_hybrid.git
cd proto_coop_hybrid
pip install -r requirements.txt
```

CUDA is recommended; the pipeline runs on CPU but CoOp training will be slow.

## Running

Open `proto_coop_hybrid.ipynb` and execute top-to-bottom. The notebook will:

1. Download Flowers102 to `./data/` via `torchvision`.
2. Load CLIP ViT-B/16 (downloaded automatically on first run).
3. Run the two-stage hyperparameter search and the final multi-seed evaluation.
4. Persist per-run summaries to `plots/`.

To override defaults (e.g. number of Stage-1 draws, seed list, alpha/conf grids), edit the call to `run_experiments(...)` in the runner cell.

## Repository layout

```
proto_coop_hybrid.ipynb   # full pipeline + report
requirements.txt          # pinned dependency ranges
```

A modular Python-package version is planned but not yet checked in.

## References

1. K. Zhou, J. Yang, C. C. Loy, Z. Liu. *Learning to Prompt for Vision-Language Models.* IJCV 2022. [arXiv:2109.01134](https://arxiv.org/abs/2109.01134)
2. R. Zhang et al. *Tip-Adapter: Training-free CLIP-Adapter for Better Vision-Language Modeling.* ECCV 2022. [arXiv:2111.03930](https://arxiv.org/abs/2111.03930)
3. A. Radford et al. *Learning Transferable Visual Models From Natural Language Supervision (CLIP).* ICML 2021. [arXiv:2103.00020](https://arxiv.org/abs/2103.00020)

## Acknowledgements

Joint work with **Christian Li Sivertsen**, originally completed for a deep-learning course.
