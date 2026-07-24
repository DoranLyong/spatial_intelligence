# World Models (Ha & Schmidhuber, 2018)

**분류:** 7-1 Latent Prediction · **태그:** 기능=Simulator/Planner, 조건화=action, 도메인=game, 물리 근거성=implicit

Google Brain/NNAISENSE/IDSIA (David Ha, Jürgen Schmidhuber). 현대 world model 분야에 **이름을 준 원조 논문**. 에이전트를 세 모듈로 분해한다: **V**(VAE — 관측 프레임을 저차원 잠재 벡터 $z$로 압축), **M**(MDN-RNN — 행동 조건부 다음 잠재 상태의 확률 밀도 $P(z_{t+1} \mid a_t, z_t, h_t)$를 예측, temperature $\tau$로 불확실성 제어), **C**(잠재 특징 $[z_t, h_t]$에서 행동을 내는 의도적으로 최소화된 선형 컨트롤러, CMA-ES로 학습). CarRacing에서 SOTA를 달성했고, VizDoom에서는 에이전트를 M이 생성한 **환각 꿈(hallucinated dream) 안에서만 학습**시켜 실제 환경으로 전이하는 것을 최초로 시연 — Dreamer 계열 상상 기반 RL의 직계 조상.

- **arXiv:** https://arxiv.org/abs/1803.10122 (NeurIPS 2018에 "Recurrent World Models Facilitate Policy Evolution"으로 게재)
- **Interactive:** https://worldmodels.github.io
- **PDF:** `2018_WorldModels.pdf`
- **판정 메모:** DreamerV3 판례 적용 — VAE 재구성은 표현 학습용 보조 신호이고 롤아웃(꿈)은 전적으로 잠재 공간에서 일어남 → 7-1. 소형 컨트롤러 동시 학습은 경계 규칙 3의 '학습' 측. 계보: MDN-RNN → RSSM(DreamerV3, 같은 7-1) → 현대 잠재 world model
- **Study note:** `/paper-to-note`로 생성 예정

## Citation

```bibtex
@article{ha2018worldmodels,
  title   = {World Models},
  author  = {Ha, David and Schmidhuber, J{\"u}rgen},
  journal = {arXiv preprint arXiv:1803.10122},
  year    = {2018},
  note    = {NeurIPS 2018 version: "Recurrent World Models Facilitate Policy Evolution"}
}
```
