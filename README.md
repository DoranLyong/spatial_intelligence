<div align="center">

# 🌏 Oh My World Intelligence

[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome) [![Papers](https://img.shields.io/badge/Archived_Papers-31-8A2BE2.svg)](#taxonomy--placement-rules) [![Taxonomy](https://img.shields.io/badge/Categories-7-blue.svg)](#taxonomy--placement-rules) [![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](#contributing)

**📜 A curated archive of World Intelligence research — from spatial perception to physical interaction and world models.**<br>

**perceive → reconstruct → generate → act → predict/simulate**

</div>

---

## 🚩 News & Updates

- 🌍 **[2026-07] World Models category launched** — Category 7 opens with 11 archived papers spanning the field's history, from *World Models* (Ha & Schmidhuber, 2018) and *Genie* (ICML 2024) to *V-JEPA 2*, *DreamerV3* (Nature 2025), and the 2026 wave (*AdaJEPA*, *Vid2World*, *Nano World Models*, *Reasoning Visual World*).
- 🔁 **[2026-07] Repository renamed** — `oh-my-spatial-intelligence` → `oh-my-world-intelligence`. Spatial intelligence (categories 1–5) knows the world's *current* state; Physical AI (category 6) *acts* on it; World Models (category 7) predict its *next* state.
- 🧭 **[2026-07] Taxonomy v2** — 7-category system with a single placement question, three boundary rules, and a case-law FAQ for ambiguous papers.
- 🤝 **[Ongoing]** Contributions welcome — see [Contributing](#contributing).

---

## Overview

- 🎯 [Aim of the Project](#aim-of-the-project)
- 🧭 [Taxonomy & Placement Rules](#taxonomy--placement-rules)
- 🖼️ [1. Rendering & Representation](#1-rendering--representation)
- 📐 [2. Geometry & Structure](#2-geometry--structure)
- ✨ [3. Generative 3D](#3-generative-3d)
- 👁️ [4. Perception & Understanding](#4-perception--understanding)
- 🧊 [5. 6D Pose Estimation](#5-6d-pose-estimation)
- 🦾 [6. Physical AI & Interaction](#6-physical-ai--interaction)
- 🌍 [7. World Models](#7-world-models)
- ⚖️ [Boundary-Case FAQ](#boundary-case-faq)
- 🏷️ [World Model Tag System](#world-model-tag-system)
- 📚 [Reference Taxonomies](#reference-taxonomies)
- 🗂️ [Repository Layout](#repository-layout)
- 🤝 [Contributing](#contributing)

---

## Aim of the Project

World intelligence research is scattered across communities — neural rendering, 3D geometry, perception, robotics, and the rapidly exploding world-model literature. This repository aims to:

- 🗄️ **Archive, not just link** — each paper ships with its PDF, a runnable official/reference implementation, and a standardized concept mind-map study note in a single self-contained folder.
- 🧭 **Place every paper unambiguously** — one placement question ("*what is the PRIMARY optimization objective?*"), three boundary rules, and a case-law FAQ resolve even hybrid papers mechanically.
- 🌉 **Bridge spatial intelligence and world models** — the taxonomy makes the pipeline explicit: perceiving and reconstructing the current world state (1–5) feeds acting on it (6) and predicting its future (7).
- 📈 **Track the 2026 world-model wave** — category 7 is organized by the *representation space* in which the future is rolled out (latent / generative video / explicit 3D / multimodal), the axis along which the community actually divides.

---

## Taxonomy & Placement Rules

Every paper is placed by a single question: **"What is the PRIMARY optimization objective?"**

| Optimization target | Category |
|---|---|
| Photometric consistency | [1. Rendering & Representation](#1-rendering--representation) |
| Correspondence & global alignment | [2. Geometry & Structure](#2-geometry--structure) |
| Static 3D generative prior | [3. Generative 3D](#3-generative-3d) |
| Semantics & feature extraction | [4. Perception & Understanding](#4-perception--understanding) |
| SE(3) pose registration | [5. 6D Pose Estimation](#5-6d-pose-estimation) |
| Physical properties, action & interaction | [6. Physical AI & Interaction](#6-physical-ai--interaction) |
| **Prediction of unobserved future states** | [**7. World Models**](#7-world-models) |

**Boundary rules:**

1. **Reconstruction vs. prediction** — a paper with a time axis (4D) that *reconstructs observed intervals* goes to 2; one that *predicts unobserved futures* goes to 7. (e.g., UFO-4D → 2, LeWorldModel → 7)
2. **Static generation vs. future rollout** — static 3D content generation goes to 3; action/time-conditioned future rollout goes to 7. (e.g., GaussianGPT → 3)
3. **Learning vs. consuming a world model** — papers that *learn* a world model go to 7; papers that build the physical substrate (assets, environments) or *consume* a world model to produce actions/policies go to 6. (e.g., PhysForge → 6)

Ambiguous cases are settled by precedent — see the [Boundary-Case FAQ](#boundary-case-faq).

---

## 1. Rendering & Representation

> *How do we represent the world so it can be rendered?* — **Photometric Consistency & Neural Rendering**

Optimized for how closely a rendering matches input images, not for geometric accuracy: a lumpy mesh is fine if the rendered image looks right. From the world-model perspective, this line owns the *explicit-3D* side of the "which space should hold world state — pixels, latents, or explicit 3D?" debate.

**📦 Archived**

- **INR Dictionaries**, "A Structured Dictionary Perspective on Implicit Neural Representations" (CVPR 2022). [![arXiv](https://img.shields.io/badge/arXiv-2112.01917-b31b1b.svg)](https://arxiv.org/abs/2112.01917) [![Archive](https://img.shields.io/badge/Archive-Local-8A2BE2.svg)](Rendering_and_Representation/2022_inr_dictionaries/)

**🔭 Also notable:** NeRF · 3D Gaussian Splatting · Instant-NGP

---

## 2. Geometry & Structure

> *How do we recover shape?* — **Correspondence, Global Alignment & Test-Time Training**

Maps pixels directly to 3D coordinates and optimizes metric accuracy of the recovered structure. Time-axis papers stay here as long as they *reconstruct observed intervals* (boundary rule 1). The test-time-training memory mechanisms of TTT3R/LoGeR connect directly to online adaptation and state maintenance in world models (cf. AdaJEPA in 7-1).

**📦 Archived**

- **Test3R**, "Test3R: Learning to Reconstruct 3D at Test Time" (2025). [![arXiv](https://img.shields.io/badge/arXiv-2506.13750-b31b1b.svg)](https://arxiv.org/abs/2506.13750) [![Archive](https://img.shields.io/badge/Archive-Local-8A2BE2.svg)](Geometry_and_Structure/2025_Test3R/) [![Note](https://img.shields.io/badge/Study-Note-e6b800.svg)](Geometry_and_Structure/2025_Test3R/study_note.md)
- **CoMe**, "CoMe: Confidence-Based Mesh Extraction from 3D Gaussians" (2026). [![arXiv](https://img.shields.io/badge/arXiv-2603.24725-b31b1b.svg)](https://arxiv.org/abs/2603.24725) [![Archive](https://img.shields.io/badge/Archive-Local-8A2BE2.svg)](Geometry_and_Structure/2026_CoMe/) [![Note](https://img.shields.io/badge/Study-Note-e6b800.svg)](Geometry_and_Structure/2026_CoMe/study_note.md)
- **LoGeR**, "LoGeR: Long-Context Geometric Reconstruction with Hybrid Memory" (2026). [![arXiv](https://img.shields.io/badge/arXiv-2603.03269-b31b1b.svg)](https://arxiv.org/abs/2603.03269) [![Archive](https://img.shields.io/badge/Archive-Local-8A2BE2.svg)](Geometry_and_Structure/2026_LoGeR/) [![Note](https://img.shields.io/badge/Study-Note-e6b800.svg)](Geometry_and_Structure/2026_LoGeR/study_note.md)
- **TTT3R**, "TTT3R: 3D Reconstruction as Test-Time Training" (ICLR 2026). [![arXiv](https://img.shields.io/badge/arXiv-2509.26645-b31b1b.svg)](https://arxiv.org/abs/2509.26645) [![Archive](https://img.shields.io/badge/Archive-Local-8A2BE2.svg)](Geometry_and_Structure/2026_TTT3R/) [![Note](https://img.shields.io/badge/Study-Note-e6b800.svg)](Geometry_and_Structure/2026_TTT3R/study_note.md)
- **UFO-4D**, "UFO-4D: Unposed Feedforward 4D Reconstruction from Two Images" (ICLR 2026). [![arXiv](https://img.shields.io/badge/arXiv-2602.24290-b31b1b.svg)](https://arxiv.org/abs/2602.24290) [![Archive](https://img.shields.io/badge/Archive-Local-8A2BE2.svg)](Geometry_and_Structure/2026_UFO-4D/) [![Note](https://img.shields.io/badge/Study-Note-e6b800.svg)](Geometry_and_Structure/2026_UFO-4D/study_note.md)

**🔭 Also notable:** DUSt3R · MVSNet · COLMAP

---

## 3. Generative 3D

> *How do we imagine what we cannot see?* — **Hallucination & Learned Priors**

Fills unobserved regions by *imagining* from learned data distributions rather than computing them. Static 3D generation stays here even when autoregressive (boundary rule 2) — GaussianGPT is flagged as a *promotion candidate* to category 7 the moment action conditioning is added.

**📦 Archived**

- [⭐️] **SAM 3D Objects**, "SAM 3D: 3Dfy Anything in Images" (Meta, 2025). [![arXiv](https://img.shields.io/badge/arXiv-2511.16624-b31b1b.svg)](https://arxiv.org/abs/2511.16624) [![Archive](https://img.shields.io/badge/Archive-Local-8A2BE2.svg)](Generative_3D/2025_SAM3D-objects/) [![Note](https://img.shields.io/badge/Study-Note-e6b800.svg)](Generative_3D/2025_SAM3D-objects/study_note.md)
- **GaussianGPT**, "GaussianGPT: Towards Autoregressive 3D Gaussian Scene Generation" (2026). [![arXiv](https://img.shields.io/badge/arXiv-2603.26661-b31b1b.svg)](https://arxiv.org/abs/2603.26661) [![Archive](https://img.shields.io/badge/Archive-Local-8A2BE2.svg)](Generative_3D/2026_GaussianGPT/) [![Note](https://img.shields.io/badge/Study-Note-e6b800.svg)](Generative_3D/2026_GaussianGPT/study_note.md)

**🔭 Also notable:** DreamFusion · Magic3D

---

## 4. Perception & Understanding

> *What is it, and what does it mean?* — **Semantics & Feature Extraction**

Identifies semantic classes, parts, and features beyond geometry. From the world-model perspective this category is the *upstream supplier of observation encoders* — the quality of a V-JEPA-style world model is bounded by its encoder.

### 4-0. Universal Encoders (`00_UniversalEncoders/`)

Task-agnostic visual representations via self-supervised pretraining or multi-teacher distillation.

- [⭐️] **I-JEPA**, "Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture" (CVPR 2023) — *the architectural origin of the JEPA world-model lineage (7-1); predicts spatially masked blocks of a static image, not future states — see [FAQ](#boundary-case-faq)*. [![arXiv](https://img.shields.io/badge/arXiv-2301.08243-b31b1b.svg)](https://arxiv.org/abs/2301.08243) [![Code](https://img.shields.io/badge/Code-GitHub-green)](https://github.com/facebookresearch/ijepa) [![Archive](https://img.shields.io/badge/Archive-Local-8A2BE2.svg)](Perception/00_UniversalEncoders/2023_IJEPA/)
- **Perception Encoder**, "Perception Encoder: The best visual embeddings are not at the output of the network" (Meta, 2025). [![arXiv](https://img.shields.io/badge/arXiv-2504.13181-b31b1b.svg)](https://arxiv.org/abs/2504.13181) [![Archive](https://img.shields.io/badge/Archive-Local-8A2BE2.svg)](Perception/00_UniversalEncoders/2026_PerceptionEncoder/) [![Note](https://img.shields.io/badge/Study-Note-e6b800.svg)](Perception/00_UniversalEncoders/2026_PerceptionEncoder/study_note.md)
- **EUPE**, "Efficient Universal Perception Encoder" (2026). [![arXiv](https://img.shields.io/badge/arXiv-2603.22387-b31b1b.svg)](https://arxiv.org/abs/2603.22387) [![Archive](https://img.shields.io/badge/Archive-Local-8A2BE2.svg)](Perception/00_UniversalEncoders/2026_EUPE/) [![Note](https://img.shields.io/badge/Study-Note-e6b800.svg)](Perception/00_UniversalEncoders/2026_EUPE/study_note.md)
- **NAVER DIVINE**, "UNIC / DUNE — universal encoder distillation" (NAVER Labs, ECCV 2024 / CVPR 2025). [![arXiv](https://img.shields.io/badge/arXiv-2408.05088-b31b1b.svg)](https://arxiv.org/abs/2408.05088) [![arXiv](https://img.shields.io/badge/arXiv-2503.14405-b31b1b.svg)](https://arxiv.org/abs/2503.14405) [![Archive](https://img.shields.io/badge/Archive-Local-8A2BE2.svg)](Perception/00_UniversalEncoders/NAVER_DIVINE/)

### 4-1. Point Cloud Analysis (`point-cloud-analysis/`)

Backbone networks learning high-dimensional features for classification and segmentation of point clouds.

- **PointMamba**, "PointMamba: A Simple State Space Model for Point Cloud Analysis" (NeurIPS 2024). [![arXiv](https://img.shields.io/badge/arXiv-2402.10739-b31b1b.svg)](https://arxiv.org/abs/2402.10739) [![Archive](https://img.shields.io/badge/Archive-Local-8A2BE2.svg)](Perception/point-cloud-analysis/2024_PointMamba/)
- **PointGST**, "PointGST: Parameter-Efficient Fine-Tuning in the Spectral Domain" (TPAMI 2025). [![arXiv](https://img.shields.io/badge/arXiv-2410.08114-b31b1b.svg)](https://arxiv.org/abs/2410.08114) [![Archive](https://img.shields.io/badge/Archive-Local-8A2BE2.svg)](Perception/point-cloud-analysis/2025_PointGST/)

### 4-2. 3D Scene Graphs *(planned)*

Hierarchical relations and attributes over 3D maps. *Also notable:* Open-Vocabulary 3D Scene Graphs · Hydra · S-Graphs

### 4-3. Open-World Segmentation (`03_OpenWorld_Segmentation/`)

Transfers SAM-family 2D segmentation knowledge to 3D (Gaussians) and video for promptable segmentation of arbitrary objects.

- **SegAnyGAussians (SAGA)**, "Segment Any 3D Gaussians" (AAAI 2025). [![arXiv](https://img.shields.io/badge/arXiv-2312.00860-b31b1b.svg)](https://arxiv.org/abs/2312.00860) [![Archive](https://img.shields.io/badge/Archive-Local-8A2BE2.svg)](Perception/03_OpenWorld_Segmentation/2025_SegAnyGAussians/) [![Note](https://img.shields.io/badge/Study-Note-e6b800.svg)](Perception/03_OpenWorld_Segmentation/2025_SegAnyGAussians/study_note.md)
- **X2SAM**, "X2SAM: Any Segmentation in Images and Videos" (2026). [![arXiv](https://img.shields.io/badge/arXiv-2605.00891-b31b1b.svg)](https://arxiv.org/abs/2605.00891) [![Archive](https://img.shields.io/badge/Archive-Local-8A2BE2.svg)](Perception/03_OpenWorld_Segmentation/2026_X2SAM/)

---

## 5. 6D Pose Estimation

> *Where is the object, and in what pose?* — **Pose Matching & Registration**

Estimates the SE(3) transform $(R, t)$ between an observation and a known reference under a rigid-body assumption. Frame-to-frame pose *tracking* is a time-series state estimate that feeds manipulation (6) and world-model state initialization (7).

**📦 Archived**

- [⭐️] **FoundationPose**, "FoundationPose: Unified 6D Pose Estimation and Tracking of Novel Objects" (CVPR 2024). [![arXiv](https://img.shields.io/badge/arXiv-2312.08344-b31b1b.svg)](https://arxiv.org/abs/2312.08344) [![Code](https://img.shields.io/badge/Code-GitHub-green)](https://github.com/NVlabs/FoundationPose) [![Archive](https://img.shields.io/badge/Archive-Local-8A2BE2.svg)](6DoF_Pose/2024_FoundationPose/)
- **Any6D**, "Any6D: Model-free 6D Pose Estimation of Novel Objects" (CVPR 2025). [![arXiv](https://img.shields.io/badge/arXiv-2503.18673-b31b1b.svg)](https://arxiv.org/abs/2503.18673) [![Archive](https://img.shields.io/badge/Archive-Local-8A2BE2.svg)](6DoF_Pose/2025_Any6D/)

**🔭 Also notable:** MegaPose · PVNet

---

## 6. Physical AI & Interaction

> *How do we act on the world?* — **Action-Oriented Perception & Physics**

Beyond seeing: infer the physical/functional properties needed for interaction, build the physics substrate world models run on, or output actions directly. Boundary with 7: *learn* a predictive model → 7; *build or touch* the world → 6 (boundary rule 3).

> **Note:** the former sub-category *6-1 Dynamic 4D Reconstruction* was dissolved — applying the placement question consistently, observed-interval dynamic reconstruction (4DGS, D-NeRF, UFO-4D) belongs to category 2 and future-dynamics prediction to category 7.

### 6-2. Physics-based Vision & Sim-Ready Assets (`02_Physics-based/`)

Inverse-physics estimation of latent parameters (mass, friction, elasticity, articulation) and generation of simulation-ready assets.

- **PhysForge**, "PhysForge: Generating Physics-Grounded 3D Assets for Interaction" (2026). [![arXiv](https://img.shields.io/badge/arXiv-2605.05163-b31b1b.svg)](https://arxiv.org/abs/2605.05163) [![Archive](https://img.shields.io/badge/Archive-Local-8A2BE2.svg)](Physical_AI/02_Physics-based/2026_PhysForge/) [![Note](https://img.shields.io/badge/Study-Note-e6b800.svg)](Physical_AI/02_Physics-based/2026_PhysForge/study_note.md)

*Also notable:* PAC-NeRF · Diff-Physics · PhyGround (physics-fidelity benchmark)

### 6-3. Affordance Learning *(planned)*

Maps functional possibilities — where and how an object can be manipulated — onto 3D space. *Also notable:* Contact-GraspNet · Affordance Diffusion · Where2Act

### 6-4. Manipulation & Policy Learning (`04_PolicyLearning/`)

Visuomotor policies and action representations: papers whose objective is the policy itself, without learning a world model.

- **B-spline Policy**, "B-spline Policy: Accelerating Manipulation Policies via B-spline Action Representations" (2026). [![arXiv](https://img.shields.io/badge/arXiv-2607.09648-b31b1b.svg)](https://arxiv.org/abs/2607.09648) [![Website](https://img.shields.io/badge/Website-Link-blue)](https://B-spline-policy.github.io) [![Archive](https://img.shields.io/badge/Archive-Local-8A2BE2.svg)](Physical_AI/04_PolicyLearning/2026_BsplinePolicy/)

*Also notable:* Diffusion Policy · ACT · OpenVLA · π0

---

## 7. World Models

> *How do we predict the world's next state?* — **Future-State Prediction & Action-Conditioned Rollout**

Papers whose primary objective is learning the transition function $p(s_{t+1} \mid s_t, a_t)$ — predicting states *not yet observed*, whatever space the state lives in. Sub-categories follow the one mutually exclusive axis the community actually divides along: **the representation space of the rollout**. Function (Renderer/Simulator/Planner), conditioning, domain, and physics grounding are multi-valued attributes handled as [tags](#world-model-tag-system), not directories.

### 7-1. Latent Prediction (`01_LatentPrediction/`)

Future rollouts happen *in latent space* — no pixels are generated during planning or imagination, which is why latent planners are fast (V-JEPA 2 plans ~15× faster than pixel-generative baselines). Includes both reconstruction-free JEPA models and RSSM/Dreamer models with auxiliary decoders: the criterion is *where the rollout lives*, not whether a reconstruction loss exists.

- [⭐️] **World Models**, "World Models" (Ha & Schmidhuber, NeurIPS 2018) — *the founding paper: policies trained inside the VAE+MDN-RNN latent dream*. [![arXiv](https://img.shields.io/badge/arXiv-1803.10122-b31b1b.svg)](https://arxiv.org/abs/1803.10122) [![Website](https://img.shields.io/badge/Website-Link-blue)](https://worldmodels.github.io) [![Archive](https://img.shields.io/badge/Archive-Local-8A2BE2.svg)](World_Models/01_LatentPrediction/2018_WorldModels/)
- [⭐️] **DreamerV3**, "Mastering Diverse Control Tasks through World Models" (Nature 640, 2025) — *one fixed configuration masters 150+ tasks; first to mine Minecraft diamonds from scratch*. [![Nature](https://img.shields.io/badge/Nature-2025-006c66.svg)](https://doi.org/10.1038/s41586-025-08744-2) [![arXiv](https://img.shields.io/badge/arXiv-2301.04104-b31b1b.svg)](https://arxiv.org/abs/2301.04104) [![Archive](https://img.shields.io/badge/Archive-Local-8A2BE2.svg)](World_Models/01_LatentPrediction/2025_DreamerV3/)
- [⭐️] **V-JEPA 2**, "V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning" (Meta, 2025) — *1M hours of video pretraining + 62 h of robot data → zero-shot latent-MPC manipulation*. [![arXiv](https://img.shields.io/badge/arXiv-2506.09985-b31b1b.svg)](https://arxiv.org/abs/2506.09985) [![Code](https://img.shields.io/badge/Code-GitHub-green)](https://github.com/facebookresearch/vjepa2) [![Archive](https://img.shields.io/badge/Archive-Local-8A2BE2.svg)](World_Models/01_LatentPrediction/2025_VJEPA2/)
- **LeWorldModel**, "LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architectures" (2026). [![arXiv](https://img.shields.io/badge/arXiv-2603.19312-b31b1b.svg)](https://arxiv.org/abs/2603.19312) [![Archive](https://img.shields.io/badge/Archive-Local-8A2BE2.svg)](World_Models/01_LatentPrediction/2026_LeWorldModel/) [![Note](https://img.shields.io/badge/Study-Note-e6b800.svg)](World_Models/01_LatentPrediction/2026_LeWorldModel/study_note.md)
- **Physical Representation Learning**, "Representation Learning for Spatiotemporal Physical Systems" (2026) — *latent prediction beats pixel reconstruction for physical inference on PDE systems*. [![arXiv](https://img.shields.io/badge/arXiv-2603.13227-b31b1b.svg)](https://arxiv.org/abs/2603.13227) [![Archive](https://img.shields.io/badge/Archive-Local-8A2BE2.svg)](World_Models/01_LatentPrediction/2026_Physical_Representation_Learning/) [![Note](https://img.shields.io/badge/Study-Note-e6b800.svg)](World_Models/01_LatentPrediction/2026_Physical_Representation_Learning/study_note.md)
- **AdaJEPA**, "AdaJEPA: An Adaptive Latent World Model" (NYU, 2026) — *test-time adaptation of a JEPA world model inside the MPC loop; cross-references the TTT line in category 2*. [![arXiv](https://img.shields.io/badge/arXiv-2606.32026-b31b1b.svg)](https://arxiv.org/abs/2606.32026) [![Archive](https://img.shields.io/badge/Archive-Local-8A2BE2.svg)](World_Models/01_LatentPrediction/2026_AdaJEPA/)

*Also notable:* DINO-world · Dreamer 4

### 7-2. Generative Video & World Simulators (`02_GenerativeVideo/`)

Future frames are generated directly by diffusion/autoregressive transformers. Visual fidelity is the strength; *action conditioning* is the practical line between a video generator and a world model.

- [⭐️] **Genie**, "Genie: Generative Interactive Environments" (DeepMind, ICML 2024) — *first generative interactive environment; unsupervised latent actions from video-only data*. [![arXiv](https://img.shields.io/badge/arXiv-2402.15391-b31b1b.svg)](https://arxiv.org/abs/2402.15391) [![Archive](https://img.shields.io/badge/Archive-Local-8A2BE2.svg)](World_Models/02_GenerativeVideo/2024_Genie/)
- **Unified World Models**, "Unified World Models: Coupling Video and Action Diffusion for Pretraining on Large Robotic Datasets" (UW/TRI, 2025) — *independent per-modality diffusion timesteps let one model act as policy / forward / inverse dynamics / video predictor*. [![arXiv](https://img.shields.io/badge/arXiv-2504.02792-b31b1b.svg)](https://arxiv.org/abs/2504.02792) [![Website](https://img.shields.io/badge/Website-Link-blue)](https://weirdlabuw.github.io/uwm/) [![Archive](https://img.shields.io/badge/Archive-Local-8A2BE2.svg)](World_Models/02_GenerativeVideo/2025_UnifiedWorldModels/)
- **Nano World Models**, "Nano World Models: A Minimalist Implementation of Future Video Prediction" (2026) — *diffusion-forcing research substrate for controlled world-model studies*. [![arXiv](https://img.shields.io/badge/arXiv-2605.23993-b31b1b.svg)](https://arxiv.org/abs/2605.23993) [![Archive](https://img.shields.io/badge/Archive-Local-8A2BE2.svg)](World_Models/02_GenerativeVideo/2026_NanoWorldModels/)
- **Vid2World**, "Vid2World: Crafting Video Diffusion Models to Interactive World Models" (ICLR 2026) — *causalization + causal action guidance turn pre-trained video diffusion into interactive world models*. [![arXiv](https://img.shields.io/badge/arXiv-2505.14357-b31b1b.svg)](https://arxiv.org/abs/2505.14357) [![Website](https://img.shields.io/badge/Website-Link-blue)](https://knightnemo.github.io/vid2world/) [![Archive](https://img.shields.io/badge/Archive-Local-8A2BE2.svg)](World_Models/02_GenerativeVideo/2026_Vid2World/)

*Also notable:* Sora 2 · Genie 3 · NVIDIA Cosmos · GAIA-2/GAIA-3 · Vista · PAN

### 7-3. Explicit 3D & Spatial World Models *(planned)*

Future/unobserved states output as explicit 3D structure (Gaussians, meshes) maintaining a persistent, physically explorable world state. *Also notable:* Marble (World Labs)

### 7-4. Multimodal & Reasoning World Models (`04_MultimodalReasoning/`)

World state modeled in *interleaved verbal + visual* representations: the world model lives inside the chain-of-thought of an LLM/UMM, with visual generation serving as world simulation for reasoning.

- **Reasoning Visual World**, "Visual Generation Unlocks Human-Like Reasoning through Multimodal World Models" (Tsinghua/ByteDance, 2026) — *formalizes the visual-superiority hypothesis; VisWorld-Eval suite*. [![arXiv](https://img.shields.io/badge/arXiv-2601.19834-b31b1b.svg)](https://arxiv.org/abs/2601.19834) [![Website](https://img.shields.io/badge/Website-Link-blue)](https://thuml.github.io/Reasoning-Visual-World) [![Archive](https://img.shields.io/badge/Archive-Local-8A2BE2.svg)](World_Models/04_MultimodalReasoning/2026_Reasoning_Visual_World/)

*Also notable:* PAN · Cosmos 3 (Reason) · ThinkMorph

---

## Boundary-Case FAQ

Placement case law — every ruling applies the single question "*what is the PRIMARY optimization objective?*".

| Case | Ruling | Rationale |
|---|---|---|
| UFO-4D is 4D — why not 7? | **2** | Reconstructs + interpolates *observed* frames; no future rollout (rule 1) |
| GaussianGPT is autoregressive — why not 7? | **3** | Autoregressive *rollout* but static scene generation, no time dynamics or actions (rule 2); promotion candidate if actions are added |
| PhysForge — why not 7? | **6-2** | Generates physics-attached sim-ready assets; does not predict futures (rule 3) |
| Physical Representation Learning — why not 6-2? | **7-1** | Latent-space future prediction (JEPA) *is* the training objective |
| PhyWorld-style physics-aligned video generation? | **7-2** + physics tag | Objective is physics-aligned *future video generation*; pure fidelity benchmarks/evaluation go to 6-2 |
| AdaJEPA-style TTT of world models? | **7-1** | The object being adapted is the latent world model itself; the TTT mechanism cross-references Test3R/TTT3R (category 2) |
| Reasoning Visual World — why not 7-2? | **7-4** | Not video rollout: world modeling inside interleaved visual-verbal CoT — a multimodal representation space |
| Joint policy + world-model training (UWM-style)? | **7-2** + Planner tag | The joint loss weights future-observation prediction equally — *learning*, not consuming (rule 3); a policy paper consuming a frozen world model goes to 6 |
| DreamerV3 has a reconstruction decoder — why 7-1? | **7-1** | The 7-1 criterion is the *rollout space*, not the presence of a reconstruction loss: RSSM imagination is fully latent; the decoder is an auxiliary representation signal |
| I-JEPA is "JEPA" — why not 7-1? | **4-0** | JEPA is an *architecture* (embedding-space prediction), not a world model per se. Spatial masks of a static image → representation learning (4-0); temporal future states → world model (7-1) |

---

## World Model Tag System

Category-7 papers carry these tags in their project README / study note (multi-valued attributes → tags, not directories):

| Tag axis | Values |
|---|---|
| Function (Fei-Fei Li taxonomy) | Renderer / Simulator / Planner |
| Conditioning | action / text / multimodal |
| Domain | driving / robotics / game / general |
| Physics grounding | implicit (data statistics) / explicit (physics alignment, simulator coupling) |

---

## Reference Taxonomies

Category 7's sub-structure aligns with the 2025–2026 community-standard frameworks (kept as references, not directories):

- Fei-Fei Li & World Labs, **"A Functional Taxonomy of World Models"** (2026) — Renderer / Simulator / Planner
- Ding et al., **"Understanding World or Predicting Future? A Comprehensive Survey of World Models"** (ACM Computing Surveys, 2025) — understanding (internal representation) vs. prediction (future states). [![arXiv](https://img.shields.io/badge/arXiv-2411.14499-b31b1b.svg)](https://arxiv.org/abs/2411.14499)
- NTU, **"World Model for Robot Learning: A Comprehensive Survey"** (2026) — policy-coupling architecture / functional purpose / generative capability. [![arXiv](https://img.shields.io/badge/arXiv-2605.00080-b31b1b.svg)](https://arxiv.org/abs/2605.00080)

---

## Repository Layout

```
<Category>/<NN_SubCategory>/<YYYY_PaperName>/
├── YYYY_PaperName.pdf     # the paper itself
├── README.md              # summary, links, BibTeX, placement notes
├── study_note.md          # standardized concept mind-map (via /paper-to-note)
└── <implementation>/      # official or reference code, self-contained env
```

- Projects are named `YYYY_PaperName/` by publication year. Planned sub-directories are created only when their first paper arrives — no empty scaffolding.
- Known naming-rule exceptions: `Perception/point-cloud-analysis/` (historical, no numeric prefix) and `Perception/00_UniversalEncoders/2026_PerceptionEncoder/` (published 2025; rename pending).

## Contributing

1. **Place it** — answer "*what is the PRIMARY optimization objective?*", apply the three [boundary rules](#taxonomy--placement-rules), and check the [FAQ](#boundary-case-faq) for precedent.
2. **Name it** — `YYYY_PaperName/` under the correct (sub-)category directory.
3. **Fill it** — paper PDF + `README.md` with summary, links, and BibTeX; category-7 papers also record their [tags](#world-model-tag-system).
4. **Note it** — generate a `study_note.md` (concept mind-map format).
5. **Sync it** — update this README's section list and the taxonomy tables in `CLAUDE.md` in the same commit.
