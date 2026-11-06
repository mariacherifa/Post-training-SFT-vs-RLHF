# LLM Post-Training from Scratch: TinyGPT, SFT, Reward Modeling & PPO

This project provides an educational implementation of a **decoder-only language model post-training pipeline**, built primarily with PyTorch. It builds on my [TinyGPT implementation from scratch](https://github.com/mariacherifa/Tiny_GPT), a character-level language model, and explores three main stages of post-training: **supervised fine-tuning (SFT)** on preferred responses, **reward modeling** using human preference pairs, and **reinforcement learning from human feedback (RLHF)** using Proximal Policy Optimization (PPO).

The goal is to gain a deeper understanding of these algorithms by implementing their core components, testing their mathematical correctness, and conducting controlled experiments to compare SFT and PPO-based RLHF. The focus is on understanding how post-training methods work and affect model behavior, rather than building a competitive general-purpose chatbot.

## 1. Research questions
The experiments performed try to answer the following questions: 

1. How does **SFT** change a pretrained language model?
2. Can a **reward model** learn to distinguish preferred from rejected responses?
3. Does **PPO-based RLHF** improve the model compared to SFT?
4. How does the KL coefficient \(\beta\) affect reward and how far the model moves from SFT reference policy?
5. How do learning rate, PPO updates, and random seeds affect training?

## 2. Pipeline

```text
TinyStories text
      |
      v
TinyGPT pretraining (character-level decoder-only Transformer)
      |
      v
Context extension (block size) (128 -> 2048) + vocabulary extension
      |
      v
Adapted pretrained TinyGPT checkpoint
      |
      +--------------------------+
      |                          |
      v                          v
SHP preferred responses      SHP preference pairs
      |                          |
      v                          v
Supervised fine-tuning       Reward-model training
      |                          |
      v                          v
SFT policy                 Frozen reward model
      |
      +---------------------+
      |                     |
      v                     v
Trainable PPO policy    Frozen SFT reference policy
      |                     |
      +----------+----------+
                 |
                 v
      Sample responses to SHP prompts
                 |
                 v
      Token-level KL penalty + terminal reward
                 |
                 v
      Returns, advantages, clipped PPO loss
                 |
                 v
      Updated PPO policy + trained value model
                 |
                 v
      Held-out evaluation: TinyGPT vs SFT vs PPO
```

For the controlled PPO experiments, **each run starts from the same SFT checkpoint and the same frozen reward model**; only the intended experimental parameters change.

## 3. Implemented components

### A. TinyGPT backbone

A small decoder-only Transformer implemented in the companion `Tiny_GPT_github` project and imported as an editable Python package. The implementation contains token/position embeddings, causal self-attention, multi-head attention, Transformer blocks, next-token cross-entropy training, and autoregressive sampling.

The original experiment used a 128-character context and a vocabulary of 90 characters. The adaptation pipeline extends the context to **2048 positions** and expands the vocabulary to **243 characters** while intending to preserve existing embeddings and weights. The continuation run shown during development used two epochs of 20 optimization steps each; that was a **pipeline smoke test**, not extensive long-context pretraining.

### B. Supervised fine-tuning (SFT)

Using a chosen response from a Stanford Human Preferences (SHP) pair, construct an instruction/response sequence. Optimize the language model using a **response-only loss mask** so prompt tokens are excluded from the SFT objective:

\[
\mathcal L_{\mathrm{SFT}}(\theta)
= -\frac{\sum_{b,t} m_{b,t}\log\pi_\theta(y_{b,t}\mid y_{b,<t})}
        {\sum_{b,t}m_{b,t}}.
\]

The implementation provides `SFTDataset`, `SFTLoss`, and `TrainSFT`, with separate train/validation loops and saved SFT checkpoints.

### C. Preference reward model

Each SHP example supplies an instruction, a preferred (`chosen`) response, and a non-preferred (`rejected`) response. The reward model produces a scalar score for each complete sequence. It is trained with a pairwise Bradley–Terry objective:

\[
\mathcal L_{\mathrm{RM}}(\phi)
= -\mathbb E\left[\log\sigma\big(R_\phi(x,y^+)-R_\phi(x,y^-)\big)\right].
\]

`RewardModel` uses a Transformer backbone and a scalar reward head; `RewardLoss` and `TrainReward` implement the optimization. A separate evaluation script measures held-out preference accuracy, pairwise loss, and reward margins.

### D. PPO-based RLHF

PPO uses four models:

| Model | Initialization | Updated during PPO? |
|---|---|---|
| Policy \(\pi_\theta\) | SFT checkpoint | Yes |
| Reference \(\pi_{\mathrm{ref}}\) | Frozen SFT checkpoint | No |
| Reward model \(R_\phi\) | Trained preference checkpoint | No |
| Value model \(V_\psi\) | Initialized for PPO | Yes |

For each sampled response token, the rollout stores \(\log\pi_{\mathrm{old}}(y_t\mid s_t)\). The sampled KL-penalty reward is

\[
r_t^{\mathrm{KL}}=-\beta\left(\log\pi_{\mathrm{old}}(y_t\mid s_t)-\log\pi_{\mathrm{ref}}(y_t\mid s_t)\right).
\]

The reward model adds a **terminal sequence reward** at the final generated token:

\[
r_t=r_t^{\mathrm{KL}}+\mathbf 1_{\{t=T\}}R_\phi(x,y).
\]

The current implementation uses discounted Monte Carlo returns and advantages:

\[
G_t=\sum_{k=t}^{T}\gamma^{k-t}r_k,
\qquad A_t=G_t-V_\psi(s_t).
\]

The clipped policy loss is minimized:

\[
\mathcal L_{\mathrm{PPO}}(\theta)=-\mathbb E_t\left[
\min\left(\rho_t(\theta)A_t,
\operatorname{clip}(\rho_t(\theta),1-\epsilon,1+\epsilon)A_t\right)
\right],
\qquad
\rho_t(\theta)=\exp\left(\log\pi_\theta-\log\pi_{\mathrm{old}}\right).
\]

The value model is fitted to fixed rollout returns with mean squared error. The policy and value model have separate optimizers, and multiple optimization passes can be performed over one rollout. **Generalized advantage estimation (GAE) is not part of the current training objective.**

## 4. Dataset

**Source:** [Stanford Human Preferences (SHP)](https://huggingface.co/datasets/stanfordnlp/SHP).

The data preparation script maps `history` to the instruction, orders `human_ref_A` / `human_ref_B` according to the preference label, and retains useful metadata such as `post_id` and `domain`. The initial sampling script requests up to **3,000 training, 500 validation, and 500 test pairs**, before removing invalid or incompatible examples.

Three task-specific views are prepared:

```python
# SFT: accepted response only
{"instruction": "...", "response": "..."}

# Reward modeling: preferred / non-preferred response
{"instruction": "...", "accepted": "...", "rejected": "..."}

# PPO: instruction only
{"instruction": "..."}
```

Because the model is **character-level**, compatibility filtering checks every formatted character against the checkpoint's `stoi` mapping. It also checks the sequence context budget. PPO prompts must leave room for generated tokens. Reward-model scoring requires that both chosen and rejected sequences fit the reward model's context size. The train, validation and test splits are kept separate; the test split is reserved for final evaluation.

## 5. Repository structure

The key directories and script names are:

```text
post-training-LLM/
├── scripts/
│   ├── dataset_analysis.py
│   ├── prepare_experiments.py
│   ├── extend_tinygpt.py
│   ├── run_pretraining.py
│   ├── run_sft.py
│   ├── run_reward.py
│   ├── evaluate_reward.py
│   ├── run_ppo.py
│   ├── evaluate_models.py
│   ├── aggregate_experiments.py
│   └── run_comparison_suite.py
├── src/
│   ├── data/
│   │   ├── SFT_dataset.py
│   │   ├── PreferenceDataSet.py
│   │   └── utils.py
│   ├── models/
│   │   ├── policy.py
│   │   ├── reference_policies_rlhf.py
│   │   ├── reward_model.py
│   │   ├── value_model.py
│   │   ├── rollout.py
│   │   └── advantages.py
│   ├── losses/
│   │   ├── sft_loss.py
│   │   ├── reward_loss.py
│   │   └── ppo_loss.py
│   └── trainers/
│       ├── training_sft.py
│       ├── train_reward.py
│       └── ppo_training.py
├── tests/
├── data/                 # locally prepared SHP examples
├── checkpoints/          # saved model weights and tokenizer mappings
└── experiments/          # configs, histories, figures, comparisons
```

`TinyGPT` itself is imported from [TinyGPT implementation from scratch](https://github.com/mariacherifa/Tiny_GPT), so the two repositories must both be available locally.

## 6. Installation

Use a Python environment with PyTorch and the scientific Python dependencies. For example, from the `post-training-LLM` repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch numpy matplotlib datasets pytest

# Install the companion TinyGPT package (adjust the relative path):
python -m pip install -e ../Tiny_GPT_github
```

The scripts choose Apple's MPS backend when available in supported parts of the pipeline; CPU is also supported. Training cost grows substantially with the context length due to attention's approximately quadratic sequence-length scaling.

## 7. Running the pipeline

Run commands from the repository root. **Audit the code and checkpoints described in §10 before reproducing training or interpreting results.**

### Inspect / prepare SHP

```bash
python -m scripts.dataset_analysis --block-size 2048 \
  --checkpoint checkpoints/tinygpt_2048_pretrained.pt
python -m scripts.prepare_experiments
```

### Train the models

```bash
# Adapt / continue pretraining TinyGPT
python -m scripts.run_pretraining

# Train supervised fine-tuning policy
python -m scripts.run_sft

# Train reward model separately; do not retrain between PPO beta runs
python -m scripts.run_reward

# Evaluate reward on validation preferences
python -m scripts.evaluate_reward --split validation

# Run one PPO configuration
python -m scripts.run_ppo --beta 0.05 --seed 42
```

These commands describe the intended entry points; the uploaded versions of several scripts need the corrections noted below before the workflow is reproducible end to end.

### Evaluate existing checkpoints

If `checkpoints/tinygpt_ppo.pt` was saved outside the new per-experiment directory structure, pass it explicitly:

```bash
python -m scripts.evaluate_models \
  --ppo-checkpoint checkpoints/tinygpt_ppo.pt \
  --split validation \
  --num-prompts 12 \
  --max-new-tokens 24
```

The evaluator also discovers checkpoints in `experiments/ppo_beta_*/tinygpt_ppo.pt` automatically. It uses identical prompts and generation settings across models and writes scores, generations, summaries, and blind SFT-versus-PPO pairs.

### Run a small controlled KL sweep

```bash
python -m scripts.run_comparison_suite \
  --train \
  --betas 0.01,0.05,0.1 \
  --seeds 42 \
  --ppo-epochs 2 \
  --train-examples 10 \
  --val-examples 4 \
  --eval-prompts 12 \
  --max-new-tokens 16
```

This is a **debug sweep**. Increase data and evaluation budgets for a scientific comparison. To use all compatible PPO prompts in the supplied split, pass `--train-examples 0 --val-examples 0`. For repeated runs, consider `--seeds 42,123,777` while fixing all other settings.

To reuse already saved `ppo_beta_*` runs without training:

```bash
python -m scripts.run_comparison_suite --split validation
```

### Aggregate manually

```bash
python -m scripts.aggregate_experiments \
  --evaluation-dir experiments/model_evaluation/<RUN_DIRECTORY>
```

The aggregator creates experiment tables, learning curves, reward-versus-KL plots, and a Markdown comparison report. It checks for inconsistent experimental settings before making controlled comparisons.

## 8. Evaluation: what we measure

| Metric | Interpretation | Limitation |
|---|---|---|
| Reward preference accuracy | Fraction of held-out chosen/rejected pairs correctly ranked | Tests ranking on SHP pairs, not necessarily generated responses |
| Pairwise reward loss / margin | Reward-model ranking confidence and direction | Raw score scale depends on training |
| Learned reward on generations | Reward model's score of each model's response | PPO may exploit reward-model mistakes |
| Sampled log-ratio to SFT | Approximate reference KL on sampled tokens | Not the exact KL; estimates can be negative |
| PPO policy loss | Clipped optimization diagnostic | Not a monotonic language-quality metric |
| Value loss | Critic fit to target returns | Low value loss does not guarantee good responses |
| Clip fraction | Fraction of sampled token ratios outside `[1-ε, 1+ε]` | Zero can be normal with small PPO updates |
| Accepted-response CE | Conditional cross-entropy of held-out accepted SHP responses | Lower is better predictive likelihood, not proof of preference |
| Blind human preference | Human comparisons of SFT and PPO responses on same prompts | Requires genuine blinded annotation |

The current generation implementation samples a **fixed number of character tokens**, without an end-of-sequence stopping rule. Response length may therefore be constant, and should not be interpreted as a quality metric yet.

For blind scoring, fill the `winner` column (`A`, `B`, or `tie`) in a copy of `blind_pairs.csv`, without exposing `blind_keys.csv` to annotators, then run:

```bash
python -m scripts.aggregate_experiments \
  --evaluation-dir experiments/model_evaluation/<RUN_DIRECTORY> \
  --judgments path/to/judged_pairs.csv
```

## 9. Experimental plan

1. **Baseline:** Compare adapted pretrained TinyGPT (M0) with SFT (M1) on identical held-out prompts.
2. **Reward quality:** Train the reward model on more than the initial debugging subset; evaluate accuracy and reward margins before trusting PPO scores.
3. **PPO comparison:** Train separate policies from **the same SFT checkpoint** using \(\beta\in\{0,0.01,0.05,0.1,0.5\}\), holding prompts, reward model, steps, learning rate and clipping fixed.
4. **Repeatability:** Repeat the useful configurations across multiple seeds; report mean and variability.
5. **Independent quality assessment:** Compare pretrained/SFT/PPO generations and blind human win rates, in addition to learned reward and accepted-response CE.
6. **Final test:** Choose settings on validation, then evaluate only the selected settings on the held-out test split.

A **continued-SFT control** with a comparable additional training budget is a useful optional baseline to separate PPO's effect from the benefit of additional optimization.

