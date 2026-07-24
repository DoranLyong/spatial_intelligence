# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a **curated taxonomy of world intelligence research** (renamed from oh-my-spatial-intelligence) — spatial perception, 3D computer vision, physical AI, and world models — organized by category. Each subdirectory contains an official or reference implementation of a published paper. The repository is research-focused — it collects, adapts, and experiments with state-of-the-art methods rather than shipping a unified product.

The taxonomy has 7 categories (directories map to these):

| Category | Directory | Sub-categories | Examples |
|---|---|---|---|
| 1. Rendering & Representation | `Rendering_and_Representation/` | — | NeRF, 3DGS, INR Dictionaries |
| 2. Geometry & Structure | `Geometry_and_Structure/` | — | DUSt3R, Test3R, CoMe, UFO-4D, LoGeR, TTT3R |
| 3. Generative 3D | `Generative_3D/` | — | SAM 3D, GaussianGPT, DreamFusion |
| 4. Perception & Understanding | `Perception/` | 4-0 Universal Encoders (`00_UniversalEncoders/`), 4-1 Point Cloud Analysis (`point-cloud-analysis/`), 4-2 Scene Graphs *(planned)*, 4-3 Open-World Segmentation (`03_OpenWorld_Segmentation/`) | I-JEPA, UNIC, DUNE, EUPE, Perception Encoder, PointMamba, SAGA, X2SAM |
| 5. 6D Pose Estimation | `6DoF_Pose/` | — | FoundationPose, Any6D |
| 6. Physical AI & Interaction | `Physical_AI/` | 6-2 Physics-based & Sim-Ready Assets (`02_Physics-based/`), 6-3 Affordance *(planned)*, 6-4 Manipulation & Policy Learning (`04_PolicyLearning/`); former 6-1 Dynamic 4D was dissolved (reconstruction → cat. 2, prediction → cat. 7) | PhysForge, B-spline Policy, Contact-GraspNet |
| 7. World Models | `World_Models/` | 7-1 Latent Prediction / JEPA (`01_LatentPrediction/`), 7-2 Generative Video (`02_GenerativeVideo/`), 7-3 Explicit 3D *(planned)*, 7-4 Multimodal & Reasoning (`04_MultimodalReasoning/`) | World Models (2018), Genie, V-JEPA 2, DreamerV3, LeWorldModel, Physical Representation Learning, AdaJEPA, Unified World Models, Nano World Models, Vid2World, Reasoning Visual World |

The root `README.md` is an English awesome-list-style archive index (global audience) defining the full taxonomy with rationale per category, plus boundary rules and a Boundary-Case FAQ for ambiguous cases (e.g. 4D reconstruction vs. future prediction, static generation vs. rollout, model learning vs. model consumption). Paper entries link to the local archive folder, study note, and arXiv/website/code where available.

## Architecture

- **No shared codebase.** Each project (e.g. `6DoF_Pose/2024_FoundationPose/`) is self-contained with its own dependencies, configs, and entry points.
- Projects are named `YYYY_ProjectName/` by publication year. Known legacy exceptions: `Perception/00_UniversalEncoders/2026_PerceptionEncoder/` (published 2025) and the un-numbered sub-category directory `Perception/point-cloud-analysis/`.
- Common internal layout per project: `models/`, `datasets/`, `cfgs/` (YAML configs), `demo/`, `requirements.txt`, conda `environments/` YAML.
- Some projects include C++/CUDA extensions built via CMake + pybind11 (FoundationPose) or compiled in-place (CroCo RoPE kernels in Test3R).

## Study Notes

Each studied paper has a `study_note.md` in its directory, created via the `/paper-to-note` skill. These follow a standardized concept mind-map format with 4-facet breakdowns (Definition, Properties, Application, Links), key equations, tables, and cross-references.

| Paper | Location |
|---|---|
| CoMe (2026) | `Geometry_and_Structure/2026_CoMe/study_note.md` |
| GaussianGPT (2026) | `Generative_3D/2026_GaussianGPT/study_note.md` |
| Test3R (2025) | `Geometry_and_Structure/2025_Test3R/study_note.md` |
| SAM 3D (2025) | `Generative_3D/2025_SAM3D-objects/study_note.md` |
| UFO-4D (2026) | `Geometry_and_Structure/2026_UFO-4D/study_note.md` |
| LoGeR (2026) | `Geometry_and_Structure/2026_LoGeR/study_note.md` |
| TTT3R (2026) | `Geometry_and_Structure/2026_TTT3R/study_note.md` |
| EUPE (2026) | `Perception/00_UniversalEncoders/2026_EUPE/study_note.md` |
| Perception Encoder (2025) | `Perception/00_UniversalEncoders/2026_PerceptionEncoder/study_note.md` |
| SegAnyGAussians (2025) | `Perception/03_OpenWorld_Segmentation/2025_SegAnyGAussians/study_note.md` |
| LeWorldModel (2026) | `World_Models/01_LatentPrediction/2026_LeWorldModel/study_note.md` |
| Physical Representation Learning (2026) | `World_Models/01_LatentPrediction/2026_Physical_Representation_Learning/study_note.md` |
| PhysForge (2026) | `Physical_AI/02_Physics-based/2026_PhysForge/study_note.md` |

## Environment & Dependencies

- **Python 3.9–3.12** across projects.
- **PyTorch 2.7.0 + CUDA 12.8** is the standardized stack.
- Package managers: pip, conda/mamba. Each project has its own `requirements.txt` and/or conda YAML.
- Docker support exists for FoundationPose.

## Per-Project Setup

Each project must be set up independently. The general pattern:

```bash
cd <Category>/<YYYY_ProjectName>
# Create env from conda YAML if present:
mamba env create -f environments/<env>.yml
# Or install via pip:
pip install -r requirements.txt
```

Check each project's README.md for specific instructions — dataset downloads, model checkpoint links, and build steps vary significantly.

## Adding a New Paper

1. Place it under the correct taxonomy category directory — decide by asking "What is the PRIMARY optimization objective?" and consult the boundary rules and Boundary-Case FAQ in the root `README.md` for ambiguous cases.
2. Name the directory `YYYY_PaperName/`.
3. Include the paper PDF and a `README.md` with setup/citation.
4. Use `/paper-to-note` to generate a `study_note.md`.
5. For World Models papers (category 7), record the tags from the World Model Tag System (function: Renderer/Simulator/Planner; conditioning; domain; physics grounding) in the project README or study note.
6. Add the paper as an entry in the root `README.md` section list (archive/note/arXiv badges) in the same commit.
7. Planned sub-directories are created when their first paper arrives — never pre-create empty scaffolding.
