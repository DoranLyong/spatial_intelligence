# Vid2World: Crafting Video Diffusion Models to Interactive World Models

**분류:** 7-2 Generative Video & World Simulators · **태그:** 기능=Renderer/Simulator, 조건화=action, 도메인=robotics/game/general, 물리 근거성=implicit

Tsinghua/Chongqing (Siqiao Huang, Jialong Wu, Qixing Zhou, Shangchen Miao, Mingsheng Long). ICLR 2026. 인터넷 스케일 action-free 비디오로 사전학습된 **full-sequence 비디오 디퓨전 모델을 인터랙티브 world model로 전환**하는 일반적 방법. 두 가지 핵심 기법: (1) **video diffusion causalization** — 양방향 temporal attention/convolution을 causal 구조로 가중치 이식하여 자기회귀 롤아웃을 가능하게 하고, (2) **causal action guidance** — 프레임 단위 행동 신호를 주입해 반사실적(counterfactual) 행동 제어를 학습. 로봇 조작·3D 게임 시뮬레이션·open-world 내비게이션에서 기존 전이 방법과 SOTA world model 대비 우위.

- **arXiv:** https://arxiv.org/abs/2505.14357 (ICLR 2026)
- **Project:** https://knightnemo.github.io/vid2world/
- **PDF:** `2026_Vid2World.pdf`
- **Study note:** `/paper-to-note`로 생성 예정

## Citation

```bibtex
@inproceedings{huang2026vid2world,
  title     = {Vid2World: Crafting Video Diffusion Models to Interactive World Models},
  author    = {Huang, Siqiao and Wu, Jialong and Zhou, Qixing and Miao, Shangchen and Long, Mingsheng},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026}
}
```
