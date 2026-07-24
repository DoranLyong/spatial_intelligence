# AdaJEPA: An Adaptive Latent World Model

**분류:** 7-1 Latent Prediction (JEPA 계열) · **태그:** 기능=Planner, 조건화=action, 도메인=robotics, 물리 근거성=implicit

NYU (Ying Wang, Oumayma Bounou, Yann LeCun, Mengye Ren). 사전학습된 JEPA latent world model을 MPC 폐루프 안에서 **test-time adaptation**하는 방법. 계획한 행동을 실행한 뒤 실제로 관측된 다음 상태 전이를 self-supervised 신호로 사용해, MPC replanning 스텝마다 gradient 1 step으로 인코더·예측기의 마지막 레이어만 재보정한다. 분포 이동(형상·시각·동역학·레이아웃) 하에서 goal-reaching 계획 성공률을 크게 개선.

- **arXiv:** https://arxiv.org/abs/2606.32026
- **PDF:** `2026_AdaJEPA.pdf`
- **교차 참조:** 카테고리 2의 test-time training 계열 (Test3R, TTT3R) — 동일한 TTT 메커니즘을 정적 기하 재구성이 아닌 latent world model에 적용
- **Study note:** `/paper-to-note`로 생성 예정

## Citation

```bibtex
@article{wang2026adajepa,
  title   = {AdaJEPA: An Adaptive Latent World Model},
  author  = {Wang, Ying and Bounou, Oumayma and LeCun, Yann and Ren, Mengye},
  journal = {arXiv preprint arXiv:2606.32026},
  year    = {2026}
}
```
