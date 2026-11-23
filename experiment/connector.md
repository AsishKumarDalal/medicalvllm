# Connector architecture — Perceiver resampler

Detailed design of the bridge between the frozen **BiomedCLIP ViT-B/16**
(196 patch tokens × 768-d) and the **minillm GPT** (768-d token embeddings).
See [vllm.md](vllm.md) for the full project plan.

- **Input:** `patches [B, 196, 768]` — patch tokens, CLS dropped
- **Output:** `visual_prefix [B, 64, 768]` — same width as minillm embeddings
- **Trainable params:** **14.82M** (2 resampler blocks, 4× FFN)
- **Trained in:** Stage 1 (LLM frozen) and Stage 2 (LLM unfrozen); tower always frozen

![Connector architecture](connector_architecture.png)

*Editable source: [`make_connector_diagram.py`](make_connector_diagram.py) —
rerun `python experiment/make_connector_diagram.py` after design changes.*

**Prior art:** this is the **Perceiver Resampler** from
[Flamingo (Alayrac et al., 2022)](https://arxiv.org/abs/2204.14198),
closely related to [BLIP-2's Q-Former](https://arxiv.org/abs/2301.12597).
The same connector is used in the medical VLM
[Med-Flamingo](https://arxiv.org/abs/2307.15189). Full citations, design-rule
paper (PaLM2-VAdapter) and medical precedents: **[references.md](references.md)**.

---

## 1. Where the connector sits

```mermaid
flowchart LR
    IMG["Medical image<br/>B × 3 × 224 × 224"] --> TOWER["BiomedCLIP ViT-B/16<br/>frozen · fp16 · no_grad"]
    TOWER --> CLS["CLS token<br/>dropped"]
    TOWER --> PATCH["Patch tokens<br/>B × 196 × 768"]
    PATCH --> CONN["Connector<br/>Perceiver resampler"]
    Q["Learned queries<br/>64 × 768"] --> CONN
    CONN --> PREFIX["Visual prefix<br/>B × 64 × 768"]

    TXT["Token ids<br/>B × T"] --> EMB["tok_emb<br/>B × T × 768"]
    PREFIX --> CAT["Concatenate on seq dim<br/>B × (64+T) × 768"]
    EMB --> CAT
    CAT --> POS["+ pos_emb[0 : 64+T]"]
    POS --> GPT["minillm GPT<br/>9 blocks · causal"]
    GPT --> LOGITS["Logits<br/>B × (64+T) × 50257"]

    classDef frozen fill:#e8e8e8,stroke:#888,stroke-dasharray:5 5,color:#444
    classDef train fill:#dbeafe,stroke:#2563eb,color:#1e3a8a
    classDef data fill:#f5f5f5,stroke:#999,color:#333
    class TOWER,CLS frozen
    class CONN train
    class IMG,PATCH,Q,PREFIX,TXT,EMB,CAT,POS,GPT,LOGITS data
```

The connector never changes token ids or vocabulary — it only manufactures
`K = 64` vectors that occupy absolute positions `0..63`; text starts at
position 64.

---

## 2. Connector internals

```mermaid
flowchart TB
    P["patches<br/>B × 196 × 768"] --> LNKV0["ln_kv"]
    LNKV0 --> KV0["K = V<br/>B × 196 × 768"]

    Q0["queries<br/>64 × 768<br/>learned parameter"] --> EXP["expand over batch<br/>B × 64 × 768"]

    EXP --> BLOCK1["Resampler block 1"]
    KV0 --> BLOCK1
    BLOCK1 --> BLOCK2["Resampler block 2"]
    KV0 --> BLOCK2
    BLOCK2 --> LNOUT["LayerNorm"]
    LNOUT --> PROJ["Linear 768 → 768"]
    PROJ --> GATE["× learnable gate<br/>init 0.1"]
    GATE --> OUT["visual prefix<br/>B × 64 × 768"]

    classDef params fill:#dbeafe,stroke:#2563eb,color:#1e3a8a
    classDef ops fill:#f8fafc,stroke:#64748b,color:#0f172a
    class Q0,GATE params
    class P,KV0,OUT ops
```

Note: K/V are projected from the same patch sequence for both blocks in this
diagram; the reference implementation normalizes patches inside each block
(`ln_kv` per block). Functionally equivalent, ~1.5K params difference.

---

## 3. One resampler block (pre-norm residual)

```mermaid
flowchart TB
    Q["q<br/>B × 64 × 768"] --> LNQ["LayerNorm 768"]
    KV["patch features<br/>B × 196 × 768"] --> LNKV["LayerNorm 768"]

    LNQ --> QP["Q projection<br/>768 → 768"]
    LNKV --> KP["K projection<br/>768 → 768"]
    LNKV --> VP["V projection<br/>768 → 768"]

    QP --> ATTN["Scaled dot-product attention<br/>12 heads · head_dim 64<br/>no causal mask · over 196 patches"]
    KP --> ATTN
    VP --> ATTN
    ATTN --> OP["Output projection<br/>768 → 768"]
    OP --> R1(("+"))
    Q --> R1
    R1 --> H["h<br/>B × 64 × 768"]

    H --> LNF["LayerNorm 768"]
    LNF --> F1["Linear 768 → 3072"]
    F1 --> GELU["GELU"]
    GELU --> F2["Linear 3072 → 768"]
    F2 --> R2(("+"))
    H --> R2
    R2 --> OUT["q′<br/>B × 64 × 768"]

    classDef norm fill:#fef3c7,stroke:#d97706,color:#78350f
    classDef attn fill:#dbeafe,stroke:#2563eb,color:#1e3a8a
    classDef ff fill:#dcfce7,stroke:#16a34a,color:#14532d
    class LNQ,LNKV,LNF norm
    class ATTN,Q,OP,KP,VP attn
    class F1,F2,GELU ff
```

Key properties:

1. **Queries are independent** — every one of the 64 slots cross-attends to
   the 196 patches; there is no self-attention among slots. Cheap, and
   duplication is harmless at this scale.
2. **No causal mask** — the image is fully visible; only the GPT is causal.
3. **Pre-norm residual** — stable at LR 1e-4 with no warmup tricks beyond
   the configured 200 steps.
4. **Cross-attention shapes** — `Q: [B, 64, 768]`, `K/V: [B, 196, 768]`,
   attention matrix `[B, 12, 64, 196]` (small: ~12K scores per image).

---

## 4. Exact tensor flow

```mermaid
flowchart LR
    A["pixel_values<br/>B × 3 × 224 × 224"] --> B["ViT patch embed<br/>14 × 14 grid"]
    B --> C["12 transformer blocks<br/>frozen"]
    C --> D["final LayerNorm<br/>B × 197 × 768"]
    D --> E["drop CLS<br/>B × 196 × 768"]
    E --> F["resampler ×2<br/>B × 64 × 768"]
    F --> G["gate × proj<br/>B × 64 × 768"]
    G --> H["concat with tok_emb<br/>B × (64+T) × 768"]
    H --> I["+ pos_emb<br/>positions 0..63+T"]
    I --> J["GPT blocks 0..8<br/>causal"]
    J --> K["final_norm + out_head<br/>B × (64+T) × 50257"]
    K --> L["CE on answer tokens<br/>labels -100 elsewhere"]

    style C fill:#e8e8e8,stroke:#888,stroke-dasharray:5 5
    style F fill:#dbeafe,stroke:#2563eb
    style G fill:#dbeafe,stroke:#2563eb
```

Sequence budget with `context_length = 1024`:

| Segment | Positions | Count |
| --- | --- | --- |
| Visual prefix | `0 … 63` | 64 |
| Prompt / question | `64 … 64+T-1` | up to ~800 |
| Answer + EOT | after prompt | ~100–150 |
| **Hard limit** | `0 … 1023` | 1024 |

If `64 + T + max_new_tokens > 1024`, image generation raises instead of
silently sliding the window (which would drop the image).

---

## 5. Training / gradient flow

```mermaid
flowchart LR
    subgraph S1["Stage 1 — alignment"]
        direction LR
        I1["image"] --> T1["ViT 🔒"] --> C1["connector 🔓"] --> L1["GPT 🔒"] --> O1["caption loss"]
    end

    subgraph S2["Stage 2 — medical VQA SFT"]
        direction LR
        I2["image"] --> T2["ViT 🔒"] --> C2["connector 🔓"] --> L2["GPT 🔓"] --> O2["answer loss"]
    end

    classDef locked fill:#e8e8e8,stroke:#888,stroke-dasharray:5 5
    classDef open fill:#dbeafe,stroke:#2563eb
    class T1,T2,L1 locked
    class C1,C2,L2 open
```

| Stage | Tower | Connector | GPT | Purpose |
| --- | --- | --- | --- | --- |
| 1 | frozen | **train** | frozen | Learn to speak the decoder's embedding space |
| 2 | frozen | **train** | **train** | Learn medical QA behavior on top of aligned prefix |
| 3 (opt) | top 2 blocks | train | train | Visual specialization at LR 1e-6 |

The tower forward runs under `torch.no_grad()` in **all** stages, so it adds
no activation memory for backward and no optimizer state.

---

## 6. Forward pass (sequence)

```mermaid
sequenceDiagram
    autonumber
    participant DL as DataLoader
    participant VT as BiomedCLIP tower
    participant CN as Connector
    participant GPT as minillm GPT
    participant LOSS as Loss

    DL->>VT: pixel_values [B,3,224,224]
    Note over VT: eval() · fp16 · no_grad
    VT-->>CN: patch tokens [B,196,768]
    CN->>CN: expand 64 learned queries [B,64,768]
    CN->>CN: block 1: cross-attend → residual → FFN → residual
    CN->>CN: block 2: cross-attend → residual → FFN → residual
    CN->>CN: LayerNorm → Linear → × gate
    CN-->>GPT: visual prefix [B,64,768]
    DL->>GPT: input_ids [B,T], labels [B,T]
    GPT->>GPT: tok_emb → concat → pos_emb → 9 blocks → out_head
    GPT->>LOSS: logits [B,64+T,50257]
    Note over LOSS: labels = [-100 × 64 | text labels]<br/>CE only on answer + EOT
    LOSS-->>CN: gradients flow only into connector
```

---

## 7. Parameter count (K=64, depth=2, 4× FFN)

| Module | Shape | Params |
| --- | --- | ---: |
| Learned queries | `64 × 768` | 49,152 |
| Per block: `ln_q` + `ln_kv` | 2 × LayerNorm(768) | 3,072 |
| Per block: cross-attn `in_proj` | `768 × 2304` + bias | 1,771,776 |
| Per block: cross-attn `out_proj` | `768 × 768` + bias | 590,592 |
| Per block: `ln_ff` | LayerNorm(768) | 1,536 |
| Per block: FFN `768→3072→768` | 2 linears | 4,722,432 |
| **Per block subtotal** | | **7,089,408** |
| **2 blocks** | | **14,178,816** |
| Final `LayerNorm` | 768 | 1,536 |
| Output projection | `768 × 768` + bias | 590,592 |
| Output gate | scalar | 1 |
| **Total connector** | | **14,820,097 (~14.8M)** |

For comparison, the frozen tower is 86M and the decoder is 102.5M, so the
connector is ~7.7% of the trainable stack in Stage 2 and the only trainable
module in Stage 1 (AdamW state ≈ 178MB fp32).

---

## 8. Reference implementation

```python
class PerceiverResampler(nn.Module):
    """196 patch tokens -> K fixed visual tokens in decoder embedding space."""

    def __init__(self, dim=768, num_queries=64, depth=2, heads=12, ff_mult=4):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(num_queries, dim) * 0.02)
        self.blocks = nn.ModuleList([
            nn.ModuleDict(dict(
                ln_q=nn.LayerNorm(dim),
                ln_kv=nn.LayerNorm(dim),
                attn=nn.MultiheadAttention(dim, heads, batch_first=True),
                ln_ff=nn.LayerNorm(dim),
                ff=nn.Sequential(
                    nn.Linear(dim, ff_mult * dim),
                    nn.GELU(),
                    nn.Linear(ff_mult * dim, dim),
                ),
            ))
            for _ in range(depth)
        ])
        self.ln_out = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, dim)
        self.gate = nn.Parameter(torch.tensor(0.1))

    def forward(self, patches):                       # [B, 196, 768]
        b = patches.size(0)
        q = self.queries.unsqueeze(0).expand(b, -1, -1)
        for blk in self.blocks:
            kv = blk["ln_kv"](patches)
            q = q + blk["attn"](blk["ln_q"](q), kv, kv,
                                need_weights=False)[0]
            q = q + blk["ff"](blk["ln_ff"](q))
        return self.gate * self.proj(self.ln_out(q))  # [B, 64, 768]
```

Design details worth preserving:

- **Query init** `N(0, 0.02)` matches GPT-2-style init; large queries cause
  early attention saturation.
- **Gate init 0.1** keeps the prefix small relative to trained token
  embeddings at step 0. The gate is a *learned scalar*, so Stage 1 can raise
  it if the signal is useful.
- **`need_weights=False`** avoids materializing the 12K-score matrix for
  logging (attention weights are not used).
- **Projection into 768-d is direct** — no bottleneck. A 256-d intermediate
  is only worth it if connector memory ever matters, which it does not here.
- **No dropout inside the connector** by default; Stage 2 GPT dropout (0.1)
  already regularizes, and the connector sees each image once per epoch.

---

## 9. Alternative connector (ablation baseline)

```mermaid
flowchart LR
    P["patches<br/>B × 196 × 768"] --> R["reshape to grid<br/>B × 768 × 14 × 14"]
    R --> AP["2 × 2 average pool<br/>B × 768 × 7 × 7"]
    AP --> FL["flatten + transpose<br/>B × 49 × 768"]
    FL --> L1["Linear 768 → 768"]
    L1 --> G["GELU"]
    G --> L2["Linear 768 → 768"]
    L2 --> O["visual tokens<br/>B × 49 × 768"]

    classDef train fill:#dbeafe,stroke:#2563eb
    class L1,L2 train
```

| | Perceiver (default) | Pooled MLP |
| --- | --- | --- |
| Visual tokens | 64, fixed | 49, tied to 14×14 grid |
| Params | 14.82M | 1.18M |
| Selection | learned queries choose regions | uniform 2×2 pooling |
| Encoder swap | any patch count | needs grid reshape |
| Role | main model | cheap baseline / sanity check |

Both write into the same 768-d space, so the GPT wrapper and data pipeline
are identical — switch with `connector: perceiver | pooled_mlp` in
`configs/vision.yaml`.

---

## 10. Failure modes and guards

| Failure | Symptom | Guard |
| --- | --- | --- |
| Prefix too loud at init | loss spikes / NaN in first 100 steps | gate 0.1 + connector-only Stage 1 |
| Queries collapse to one vector | captions ignore details, low diversity | monitor pairwise cosine of queries; add diversity loss only if observed |
| Patch features unnormalized | attention logits explode | `ln_kv` before every attention |
| Connector overfits ROCO style | generic "radiograph shows…" captions | Stage 2 mix + 10% text replay; early stop on VQA dev |
| Tower accidentally trained | VRAM jump, degraded encoder | `requires_grad_(False)` + `eval()` + `no_grad()` + test asserting bit-identical tower weights after a step |
| Sequence overflow | image silently dropped | hard error when `K + T + max_new > context_length` |

---

## 11. Tests specific to the connector

Per `agent.md`, tiny CPU model + fake random tower, no HF downloads:

1. output shape `[B, 64, 768]` for arbitrary `B`;
2. gate scales output linearly (`gate=0` ⇒ zero prefix);
3. gradients reach `queries`, `proj`, block params;
4. frozen fake tower params are bit-identical after `backward()`;
5. `K` is configurable (32 / 64 / 128) and output respects it;
6. full wrapper: prefix positions have `labels == -100`;
7. KV-cached generation with prefix matches uncached logits (≤ 1e-4).
