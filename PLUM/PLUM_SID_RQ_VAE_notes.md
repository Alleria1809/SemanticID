# PLUM SID Construction — Technical Notes

**Paper:** [PLUM: Adapting Pre-trained Language Models for Industrial-scale Generative Recommendations](https://arxiv.org/abs/2510.07784), §2.1.

## 1. SID Model: Inputs and Outputs

- **Input:** pre-extracted item content embeddings (e.g., video and text), **not** a ground-truth SID.
- **Architecture:** modality-specific **DNN encoders** → concatenate/project to `z ∈ R^d` → **residual quantizer** → modality-specific decoders. A projection head supports contrastive training.
- Upstream content feature extractors can be frozen; **the SID model's DNN encoders are trainable**.
- **Outputs:** an SID `(k_1, ..., k_L)`, quantized vector `z_q`, and reconstructed input embeddings. No SID labels are required.

## 2. How Residual Quantization Works

Each level has a codebook `C_l ∈ R^(K_l × d)` containing `K_l` trainable **d-dimensional** codewords. Different levels can have different `K_l`, but the codeword dimension matches `z`.

Starting from `r_0 = z`, repeat for levels `l = 1, ..., L`:

\[
k_l=\arg\min_k\|r_{l-1}-C_l[k]\|_2^2,\quad
 e_l=C_l[k_l],\quad r_l=r_{l-1}-e_l.
\]

**Nearest codeword → subtract → repeat on residual.** The SID consists of selected indices; the quantized vector is `z_q = Σ_l e_l ≈ z`. This is *greedy* quantization, not joint combinatorial search.

**Example:** `z=[2.6,1.4]`; selected vectors `[2,1]`, `[0.5,0]`, `[0.1,0.3]` give residuals `[0.6,0.4] → [0.1,0.4] → [0,0.1]` and `z_q=[2.6,1.3]`.

**PLUM extensions:** multi-resolution codebooks with `K_l = 2048/2^(l−1)`; **progressive masking** randomly selects a prefix length `r` and forms `z_q` from only the first `r` selected codewords during training. The paper does not specify the sampling distribution or whether the depth is per-item/per-batch, and its mask inequality has an indexing ambiguity.

## 3. Three Training Losses

\[
\mathcal L=\mathcal L_{\mathrm{recon}}+\mathcal L_{\mathrm{rq}}+\mathcal L_{\mathrm{con}}.
\]

| Loss | Requires a co-watch pair? | Purpose |
|---|---|---|
| **Reconstruction** | No | Reconstruct each original content embedding from `z_q`: `Σ_m ||x_m − x̂_m||²`. |
| **RQ / VQ** | No | Codebook loss moves selected codewords toward the residual; **commitment loss trains the SID encoder** to approach selected codewords. |
| **Behavior contrastive** | **Yes** | Makes representations of co-watched items similar relative to in-batch comparison videos. |

### RQ loss: codebook vs. commitment

The standard, implementation-oriented loss for quantization level `l` is:

\[
\mathcal L_{\mathrm{rq}}^{(l)}=
\underbrace{\|\operatorname{sg}(u_l)-e_l\|_2^2}_{\text{Codebook loss}}
+\underbrace{\beta\|u_l-\operatorname{sg}(e_l)\|_2^2}_{\text{Commitment loss}},
\qquad u_l=r_{l-1}.
\]

Here `u_l` is the residual **before** level `l` quantization, `e_l = C_l[k_l]` is the selected codeword, and `sg(·)` means **stop-gradient**.

| Term | Gradient flows to | Effect |
|---|---|---|
| **Codebook** `||sg(u_l) − e_l||²` | Selected codeword `e_l` (a trainable codebook parameter) | Moves `e_l` toward the encoder's residual `u_l`. |
| **Commitment** `β||u_l − sg(e_l)||²` | SID encoder / upstream residual-generating path | Encourages the encoder output to stay close to its chosen codeword; `e_l` is detached **in this term**. |

**How PLUM can update codebook weights without a rotation trick:**

1. Forward: find `k_l = argmin_k ||u_l − C_l[k]||²` and gather `e_l = C_l[k_l]`.
2. Backward: although the **index selection** (`argmin`) is non-differentiable, `e_l = C_l[k_l]` is differentiable **with respect to the selected codeword values**. The **codebook loss directly supplies their gradients**:

   \[
   \frac{\partial\mathcal L_{\mathrm{codebook}}^{(l)}}{\partial e_l}
   =2(e_l-\operatorname{sg}(u_l)).
   \]

3. Optimizer step: SGD/Adam updates the selected rows of each codebook, moving them toward the assigned residuals. Unselected codewords get no gradient from this term in that step (one reason codebook collapse can occur).

For example, if `u_l = [2.6, 1.4]` and the selected codeword is `e_l = [2, 1]`, its codebook-loss gradient is `[-1.2, -0.8]`; gradient descent therefore **increases** both coordinates of `e_l` toward the residual.

**STE is a different mechanism:** a standard RQ-VAE uses the **straight-through estimator** to propagate reconstruction (and, when routed through quantization, contrastive) gradients to the SID encoder despite discrete index selection. A typical surrogate is `q_st = u + (q − u).detach()`, whose **forward value is `q` but whose gradient w.r.t. `u` is the identity**. The codebook weights do **not** require STE or a rotation trick to receive the gradients from the explicit codebook-loss term. PLUM specifies the codebook/commitment objective but does **not** document its exact autograd/STE implementation; STE is a standard reproduction choice, not a confirmed PLUM detail.

**Notation caveat:** PLUM's printed Eq. (2) uses `r_l` inside both RQ terms **and** defines `r_l = r_{l-1} − e_l`. The standard quantization objective compares a selected codeword against the **pre-quantization** residual `r_{l-1}` (`u_l` above). The printed indexing should be checked rather than copied literally.

**Batch example:** if `(A, B)` is a co-watch positive pair, each item gets its own reconstruction and RQ losses:

- `L_A = L_recon(A) + L_rq(A) + L_con(A → B)`
- `L_B = L_recon(B) + L_rq(B) + L_con(B → A)`

`A → B` means A is the anchor and B the positive; `B → A` reverses the roles. Both can be evaluated **in one shared-model batch and one optimizer step**, using other batch items as comparisons. Exact loss reduction is an implementation choice. **Paper caveat:** PLUM's printed contrastive objective lacks the `log` of standard InfoNCE, does not explicitly exclude the anchor from its denominator, and has an `N_b`/`2N_b` notation inconsistency. Standard bidirectional InfoNCE is an alternative implementation, not an exact transcription.

## 4. Initialization, Collapse, and RQ-KMeans

- **Random initialization:** initialize each codebook's `K_l × d` vectors randomly.
- **K-means initialization:** cluster `z` to initialize level 1, then cluster each level's residuals to initialize later codebooks. Codewords begin near the actual data distribution.
- **Why it helps:** codewords never selected by nearest-neighbor assignment receive no normal codebook-loss updates and can remain *dead*. K-means improves initial coverage but **cannot guarantee** no later collapse.
- **PLUM does not disclose its codebook initialization method.** Monitor code utilization and assignment perplexity. *Codebook collapse* (unused codes) differs from *SID collision* (different items sharing an SID).

| RQ-KMeans | RQ-VAE + K-means initialization |
|---|---|
| K-means **learns** codebooks at successive residual levels, without gradient-trained encoder/decoder. | K-means only **initializes** codebooks; encoder, codebooks, and decoder are then trained with gradient-based losses. |

## 5. What to Keep After Training

- **Existing items:** store their SID index sequences for downstream LLM training and SID-to-item lookup.
- **New/cold-start items:** retain the trained SID **encoders/projection and codebooks** to assign SIDs from content alone. The decoders and contrastive head are not needed at inference.
- Codeword indices are **categorical labels**: the numerical gap between IDs does not measure semantic distance.

**Takeaway:** RQ-VAE learns discrete content-based item representations; PLUM adds co-watch supervision through the training loss rather than requiring behavior features to generate each SID.
