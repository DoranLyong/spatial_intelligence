## Project Conventions

- Each paper implementation lives under its taxonomy category directory, named `YYYY_PaperName/`.
- Categories: `Rendering_and_Representation/`, `Geometry_and_Structure/`, `Generative_3D/`, `Perception/`, `6DoF_Pose/`, `Physical_AI/`, `World_Models/`.
- Category 4 (Perception) has sub-categories: 4-0 Universal Encoders (`00_UniversalEncoders/`), 4-1 Point Cloud Analysis (`point-cloud-analysis/`), 4-2 Scene Graphs *(planned)*, 4-3 Open-World Segmentation (`03_OpenWorld_Segmentation/`).
- Category 6 (Physical AI) has sub-categories: 6-2 Physics-based & Sim-Ready Assets (`02_Physics-based/`), 6-3 Affordance *(planned)*, 6-4 Manipulation & Policy Learning (`04_PolicyLearning/`). Former 6-1 Dynamic 4D was dissolved: observed-interval reconstruction goes to category 2, future prediction to category 7.
- Category 7 (World Models) has sub-categories by representation space: 7-1 Latent Prediction / JEPA (`01_LatentPrediction/`), 7-2 Generative Video (`02_GenerativeVideo/`), 7-3 Explicit 3D *(planned)*, 7-4 Multimodal & Reasoning (`04_MultimodalReasoning/`). Function/conditioning/domain/physics-grounding are tags (README 부록 C), not directories.
- To decide where a paper belongs, ask: "What is the PRIMARY optimization objective?" Boundary rules: observed-interval reconstruction → 2 vs. future prediction → 7; static 3D generation → 3 vs. action/time-conditioned rollout → 7; learning a world model → 7 vs. consuming one or building physics assets → 6. See README 부록 B for precedents.
- Planned sub-directories are created only when their first paper arrives — no empty scaffolding.
- Every new project must include its own `README.md` with setup instructions, usage, and BibTeX citation.
- Use `/paper-to-note` to create a `study_note.md` for each studied paper.
- The standardized stack is PyTorch 2.7.0 + CUDA 12.8. Prefer mamba for environment creation.
- The root `README.md` is an English awesome-list-style archive index (the repo targets a global audience; the earlier Korean taxonomy lives in git history). When adding a paper, add its entry to the matching README section with Archive/Note badges (plus arXiv/website/code badges when known) in the same commit.
- Per-paper READMEs and study notes may remain Korean or English; new ones for the global archive should prefer English.
