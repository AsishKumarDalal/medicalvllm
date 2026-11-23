# References — connector architecture and medical VLM prior art

Short answer: the connector is **not a new invention**. It is the
**Perceiver Resampler** from Flamingo, closely related to BLIP-2's Q-Former,
and already used in a medical VLM (Med-Flamingo). Our contribution is the
adaptation to a 100M `minillm` decoder + BiomedCLIP + medical data mix, not
the connector itself.

---

## 1. Direct sources for the connector

| Our design choice | Paper | What it establishes |
| --- | --- | --- |
| Learned queries cross-attending to patch tokens, fixed K outputs | **Flamingo** — Alayrac et al., NeurIPS 2022, [arXiv:2204.14198](https://arxiv.org/abs/2204.14198) | Introduces the **Perceiver Resampler**: "a predefined number of latent input queries" → fixed **64 visual tokens**, frozen vision encoder. This is the exact template. |
| Latent-query cross-attention as a general architecture | **Perceiver** — Jaegle et al., ICML 2021, [arXiv:2103.03206](https://arxiv.org/abs/2103.03206); **Perceiver IO** — Jaegle et al., ICLR 2022, [arXiv:2107.14795](https://arxiv.org/abs/2107.14795) | Origin of learned latents attending to a large input array; the resampler is a 2-block Perceiver. |
| Learned queries + frozen encoder + frozen LLM, two-stage training | **BLIP-2** — Li et al., ICML 2023, [arXiv:2301.12597](https://arxiv.org/abs/2301.12597) | **Q-Former**: 32 learned queries, cross-attention to frozen image features, then generative tuning with a frozen LLM. Closest training recipe (Stage 1 alignment → Stage 2 generative). |
| Simpler projector alternative (linear / 2-layer MLP) | **LLaVA** — Liu et al., NeurIPS 2023, [arXiv:2304.08485](https://arxiv.org/abs/2304.08485); **LLaVA-1.5** — Liu et al., CVPR 2024, [arXiv:2310.03744](https://arxiv.org/abs/2310.03744) | Establishes the two-stage recipe: alignment on image-caption pairs, then visual instruction tuning; projector can be as simple as one linear layer. |
| Linear projection on top of a Q-Former | **MiniGPT-4** — Zhu et al., 2023, [arXiv:2304.10592](https://arxiv.org/abs/2304.10592) | Same frozen-encoder/frozen-LLM pattern with a very small trainable bridge. |
| Instruction-aware querying | **InstructBLIP** — Dai et al., NeurIPS 2023, [arXiv:2305.06500](https://arxiv.org/abs/2305.06500) | Q-Former conditioned on the instruction; motivates our Stage-2 unfreeze if fixed queries underfit. |
| Resampler design rules (separate LNs, FFN, moderate width) | **PaLM2-VAdapter** — 2024, [arXiv:2402.10896](https://arxiv.org/abs/2402.10896) | Empirical rules for Perceiver-resampler adapters: LayerNorms applied **separately** to queries and cross-modal K/V, FFN improves stability, moderate query width. Matches our block design. |
| Connector ablation (MLP vs resampler vs pooling) | **What matters when building vision-language models?** (Idefics2) — Laurençon et al., NeurIPS 2024, [arXiv:2405.02246](https://arxiv.org/abs/2405.02246) | Systematic study of connector choices, image resolution, and data; useful for our K/encoder ablations. |
| Inductive bias in visual projectors | **Honeybee** — Cha et al., NeurIPS 2023, [arXiv:2312.06742](https://arxiv.org/abs/2312.06742) | Compares linear/MLP/convolutional projectors; supports keeping the resampler as default and MLP as cheap baseline. |

---

## 2. Medical VLM prior art using this pattern

| Model | Vision encoder | Connector | LLM | Training | Relevance |
| --- | --- | --- | --- | --- | --- |
| **Med-Flamingo** — Moor et al., ML4H 2023, [arXiv:2307.15189](https://arxiv.org/abs/2307.15189) | CLIP ViT-L/14, frozen | **Perceiver resampler** (Flamingo-style) | LLaMA-7B, frozen + gated x-attn | Continued pretraining on medical image-text | **Closest precedent**: medical domain + exactly this connector + frozen encoder. |
| **LLaVA-Med** — Li et al., NeurIPS 2023 D&B, [arXiv:2306.00890](https://arxiv.org/abs/2306.00890) | CLIP ViT-L/14, frozen | linear projection | Vicuna-13B | Stage 1 biomedical alignment on PMC figure-caption, Stage 2 GPT-4 instruction data | Validates our **two-stage curriculum** and caption→VQA data order, at 7B–13B scale. |
| **BiomedCLIP** — Zhang et al., [arXiv:2303.00915](https://arxiv.org/abs/2303.00915) | ViT-B/16 + PubMedBERT, PMC-15M | — (contrastive encoder) | — | 15M image-text pairs | Our frozen encoder; supplies the 196×768 patch features. |
| **MedGemma** — Sellergren et al., 2025, [arXiv:2507.05201](https://arxiv.org/abs/2507.05201) | MedSigLIP-400M, medically tuned | projector into Gemma 3 | Gemma 3 4B/27B | Medical multimodal tuning | Modern evidence that a medical-domain encoder + projector is the right shape; much larger scale. |
| **HuatuoGPT-Vision / PubMedVision** — Chen et al., 2024, [arXiv:2406.19280](https://arxiv.org/abs/2406.19280) | CLIP-L/336 | 2-layer MLP | LLaMA-3-8B / Yi-34B | 1.3M PubMed VQA, LLaVA-1.5 recipe | Data source for Stage 2 scale-up; shows MLP projector is competitive with enough data. |
| **MIMO** — Chen et al., CVPR 2025 | medical encoder | resampler + cross-attention | LLM | referring/grounding medical VQA | Recent medical VLM using a resampler-style bridge. |
| **CheXagent** — Chen et al., 2024, [arXiv:2401.12208](https://arxiv.org/abs/2401.12208) | chest X-ray foundation encoder | projector | LLM | chest X-ray instruction tuning | Specialty (CXR) comparison point. |

**Positioning of our work:** the connector and two-stage recipe are
established; what differs is the **constrained decoder** (102.5M, 9 layers,
1024 context), the **BiomedCLIP 768-d alignment** (no dimension adapter
needed), and the **64-token absolute-position prefix** rather than gated
cross-attention layers inside the LM. That makes it a small-scale,
reproducible medical VLM study rather than a new architecture.

---

## 3. Dataset papers

| Dataset | Paper | Scale used here |
| --- | --- | --- |
| **ROCOv2** | Rückert et al., 2024, [arXiv:2405.10004](https://arxiv.org/abs/2405.10004) | 79,789 radiology images; ~60k train pairs (Stage 1) |
| **PMC-VQA** | Zhang et al., 2023, [arXiv:2305.10415](https://arxiv.org/abs/2305.10415) | 227k QA / 149k images (Stage 2) |
| **SLAKE** | Liu et al., ISBI 2021, [arXiv:2102.09542](https://arxiv.org/abs/2102.09542) | 642 images / 14k QA (train + eval) |
| **VQA-RAD** | Lau et al., *Scientific Data* 5:180251, 2018, [doi:10.1038/sdata.2018.251](https://doi.org/10.1038/sdata.2018.251) | 315 images / 3,515 QA (eval + small train) |
| **PathVQA** | He et al., 2020, [arXiv:2003.10286](https://arxiv.org/abs/2003.10286) | 4,998 images / 32,799 QA (Stage 2 pathology) |
| **PubMedVision** | Chen et al., 2024, [arXiv:2406.19280](https://arxiv.org/abs/2406.19280) | 1.3M medical VQA (optional scale-up) |

---

## 4. BibTeX

```bibtex
@inproceedings{alayrac2022flamingo,
  title     = {Flamingo: a Visual Language Model for Few-Shot Learning},
  author    = {Alayrac, Jean-Baptiste and Donahue, Jeff and Luc, Pauline and Miech, Antoine and Barr, Iain and Hasson, Yana and Lenc, Karel and Mensch, Arthur and Millican, Katie and Reynolds, Malcolm and others},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2022},
  eprint    = {2204.14198},
  archivePrefix = {arXiv}
}

@inproceedings{jaegle2021perceiver,
  title     = {Perceiver: General Perception with Iterative Attention},
  author    = {Jaegle, Andrew and Gimeno, Felix and Brock, Andrew and Vinyals, Oriol and Zisserman, Andrew and Carreira, Joao},
  booktitle = {International Conference on Machine Learning},
  year      = {2021},
  eprint    = {2103.03206},
  archivePrefix = {arXiv}
}

@inproceedings{li2023blip2,
  title     = {{BLIP-2}: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models},
  author    = {Li, Junnan and Li, Dongxu and Savarese, Silvio and Hoi, Steven},
  booktitle = {International Conference on Machine Learning},
  year      = {2023},
  eprint    = {2301.12597},
  archivePrefix = {arXiv}
}

@inproceedings{liu2023llava,
  title     = {Visual Instruction Tuning},
  author    = {Liu, Haotian and Li, Chunyuan and Wu, Qingyang and Lee, Yong Jae},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2023},
  eprint    = {2304.08485},
  archivePrefix = {arXiv}
}

@inproceedings{liu2024llava15,
  title     = {Improved Baselines with Visual Instruction Tuning},
  author    = {Liu, Haotian and Li, Chunyuan and Li, Yuheng and Lee, Yong Jae},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year      = {2024},
  eprint    = {2310.03744},
  archivePrefix = {arXiv}
}

@article{moor2023medflamingo,
  title     = {Med-Flamingo: a Multimodal Medical Few-shot Learner},
  author    = {Moor, Michael and Huang, Qian and Wu, Shirley and Yasunaga, Michihiro and Dalmia, Yash and Leskovec, Jure and Zakka, Cyril and Reis, Eduardo Pontes and Rajpurkar, Pranav},
  journal   = {Proceedings of Machine Learning Research},
  volume    = {225},
  pages     = {353--367},
  year      = {2023},
  eprint    = {2307.15189},
  archivePrefix = {arXiv}
}

@inproceedings{li2023llavamed,
  title     = {{LLaVA-Med}: Training a Large Language-and-Vision Assistant for Biomedicine in One Day},
  author    = {Li, Chunyuan and Wong, Cliff and Zhang, Sheng and Usuyama, Naoto and Liu, Haotian and Yang, Jianwei and Naumann, Tristan and Poon, Hoifung and Gao, Jianfeng},
  booktitle = {Advances in Neural Information Processing Systems, Datasets and Benchmarks Track},
  year      = {2023},
  eprint    = {2306.00890},
  archivePrefix = {arXiv}
}

@article{zhang2023biomedclip,
  title     = {{BiomedCLIP}: a Multimodal Biomedical Foundation Model Pretrained from Fifteen Million Scientific Image-Text Pairs},
  author    = {Zhang, Sheng and Xu, Yanbo and Usuyama, Naoto and Xu, Hanwen and Bagga, Jaspreet and Tinn, Robert and Preston, Sam and Rao, Rajesh and Wei, Mu and Valluri, Naveen and others},
  journal   = {arXiv preprint arXiv:2303.00915},
  year      = {2023}
}

@article{sellergren2025medgemma,
  title     = {{MedGemma} Technical Report},
  author    = {Sellergren, Andrew and Kazemzadeh, Sahar and Jaroensri, Tiam and Kiraly, Atilla and Traverse, Madeleine and Kohlberger, Timo and others},
  journal   = {arXiv preprint arXiv:2507.05201},
  year      = {2025}
}

@inproceedings{zhang2023pmcvqa,
  title     = {{PMC-VQA}: Visual Instruction Tuning for Medical Visual Question Answering},
  author    = {Zhang, Xiaoman and Wu, Chaoyi and Zhao, Ziheng and Lin, Weixiong and Zhang, Ya and Wang, Yanfeng and Xie, Weidi},
  booktitle = {International Conference on Medical Image Computing and Computer-Assisted Intervention},
  year      = {2023},
  eprint    = {2305.10415},
  archivePrefix = {arXiv}
}

@article{ruckert2024rocov2,
  title     = {{ROCOv2}: Radiology Objects in COntext Version 2, an Updated Multimodal Image Dataset},
  author    = {R{\"u}ckert, Johannes and Bloch, Louise and Br{\"u}ngel, Raphael and Idrissi-Yaghir, Ahmad and Sch{\"a}fer, Henning and Schmidt, Cynthia S. and Koitka, Sven and Pelka, Obioma and Ben Abacha, Asma and Garc{\'i}a Seco de Herrera, Alba and others},
  journal   = {arXiv preprint arXiv:2405.10004},
  year      = {2024}
}

@article{lau2018vqarad,
  title     = {A Dataset of Clinically Generated Visual Questions and Answers about Radiology Images},
  author    = {Lau, Jason J. and Gayen, Soumya and Ben Abacha, Asma and Demner-Fushman, Dina},
  journal   = {Scientific Data},
  volume    = {5},
  pages     = {180251},
  year      = {2018},
  doi       = {10.1038/sdata.2018.251}
}

@inproceedings{liu2021slake,
  title     = {{SLAKE}: A Semantically-Labeled Knowledge-Enhanced Dataset for Medical Visual Question Answering},
  author    = {Liu, Bo and Zhan, Li-Ming and Xu, Li and Ma, Lin and Yang, Yan and Wu, Xiao-Ming},
  booktitle = {IEEE International Symposium on Biomedical Imaging},
  pages     = {1650--1654},
  year      = {2021},
  eprint    = {2102.09542},
  archivePrefix = {arXiv}
}

@article{he2020pathvqa,
  title     = {{PathVQA}: 30000+ Questions for Medical Visual Question Answering},
  author    = {He, Xuehai and Zhang, Yichen and Mou, Luntian and Xing, Eric and Xie, Pengtao},
  journal   = {arXiv preprint arXiv:2003.10286},
  year      = {2020}
}

@article{chen2024huatuogptvision,
  title     = {{HuatuoGPT-Vision}, Towards Injecting Medical Visual Knowledge into Multimodal {LLMs} at Scale},
  author    = {Chen, Junying and Ouyang, Ruyi and Gao, Anningzhe and Chen, Shunian and Chen, Guiming Hardy and Wang, Xidong and Zhang, Ruifei and Cai, Zhenyang and Ji, Ke and Yu, Guangjun and Wan, Xiang and Wang, Benyou},
  journal   = {arXiv preprint arXiv:2406.19280},
  year      = {2024}
}
```
