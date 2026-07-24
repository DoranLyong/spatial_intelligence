# Visual Generation Unlocks Human-Like Reasoning through Multimodal World Models

**분류:** 7-4 Multimodal & Reasoning World Models · **태그:** 기능=Simulator/Planner, 조건화=multimodal, 도메인=general, 물리 근거성=implicit

Tsinghua/ByteDance Seed (Jialong Wu, Xiaoying Zhang, Mingsheng Long 외). 시각 생성이 언제·어떻게 추론에 도움이 되는지를 **world model 관점에서 다룬 최초의 원리적 연구**. 태스크의 세계를 multi-observable MDP(MOMDP)로 정식화하고, world model의 두 원자 능력(world reconstruction / world simulation)을 정의한 뒤, 언어 CoT만 쓰는 implicit/verbal world modeling과 시각 생성을 교차하는 visual world modeling을 구분한다. **Visual superiority hypothesis**: 물리 세계에 근거한 태스크에서는 시각 생성이 더 자연스러운 world model이 된다. 이를 검증하는 7-태스크 평가 스위트 **VisWorld-Eval**을 구축하고, SOTA UMM(BAGEL)에서 interleaved visual-verbal CoT가 verbal CoT를 크게 능가함(단, 시각적 world modeling이 불필요한 미로·Sokoban류에서는 이득 없음)을 보였다.

- **arXiv:** https://arxiv.org/abs/2601.19834
- **Project:** https://thuml.github.io/Reasoning-Visual-World
- **PDF:** `2026_Reasoning_Visual_World.pdf`
- **교차 참조:** 7-2의 Vid2World (같은 THUML 그룹 — 픽셀 공간 world model)와 표현 공간 축에서 대비되는 짝
- **Study note:** `/paper-to-note`로 생성 예정

## Citation

```bibtex
@article{wu2026visualreasoning,
  title   = {Visual Generation Unlocks Human-Like Reasoning through Multimodal World Models},
  author  = {Wu, Jialong and Zhang, Xiaoying and Yuan, Hongyi and Zhang, Xiangcheng and Huang, Tianhao and He, Changjing and Deng, Chaoyi and Zhang, Renrui and Wu, Youbin and Long, Mingsheng},
  journal = {arXiv preprint arXiv:2601.19834},
  year    = {2026}
}
```
