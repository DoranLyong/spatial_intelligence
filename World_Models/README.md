# 7. World Models

세계의 다음 상태를 어떻게 예측할 것인가? — 전이 함수 $p(s_{t+1} \mid s_t, a_t)$의 학습, 즉 **미관측 미래 상태의 예측**이 PRIMARY optimization objective인 논문들. 2026년 신설 카테고리.

하위 분류는 상호 배타적인 **표현 공간 축**만 디렉토리로 나눈다. 기능(Renderer/Simulator/Planner)·조건화·도메인·물리 근거성은 태그로 관리한다 — 경계 판정 FAQ(부록 B)와 태그 체계(부록 C)는 루트 [README.md](../README.md) 참조.

## 하위 분류

| 하위 분류 | 디렉토리 | 보유 논문 | 외부 예시 |
|---|---|---|---|
| 7-1 Latent Prediction (JEPA·RSSM 계열) | `01_LatentPrediction/` | World Models (2018, Ha & Schmidhuber), V-JEPA 2 (2025), DreamerV3 (2025), LeWorldModel (2026), Physical Representation Learning (2026), AdaJEPA (2026) | DINO-world, Dreamer 4 |
| 7-2 Generative Video & World Simulators | `02_GenerativeVideo/` | Genie (2024), Unified World Models (2025), Nano World Models (2026), Vid2World (2026) | Sora 2, Genie 3, Cosmos, GAIA-2/3 |
| 7-3 Explicit 3D & Spatial | *(계획)* | — | Marble (World Labs) |
| 7-4 Multimodal & Reasoning | `04_MultimodalReasoning/` | Reasoning Visual World (2026) | PAN, Cosmos 3 (Reason), ThinkMorph |

## 인접 카테고리와의 경계

- 관측 구간의 (동적) 재구성 → 카테고리 2 (예: UFO-4D)
- 시간 동역학·행동 조건 없는 정적 3D 생성 → 카테고리 3 (예: GaussianGPT)
- 물리 속성 부착 sim-ready 에셋 생성, world model 소비(정책 산출) → 카테고리 6 (예: PhysForge)
