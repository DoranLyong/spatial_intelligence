# Unified World Models: Coupling Video and Action Diffusion for Pretraining on Large Robotic Datasets

**분류:** 7-2 Generative Video & World Simulators · **태그:** 기능=Planner/Simulator, 조건화=action, 도메인=robotics, 물리 근거성=implicit

UW/Toyota Research Institute (Chuning Zhu, Raymond Yu, Siyuan Feng, Benjamin Burchfiel, Paarth Shah, Abhishek Gupta). 행동 디퓨전과 비디오(다음 관측) 디퓨전을 단일 디퓨전 트랜스포머에 통합하되, **모달리티별 diffusion timestep을 독립적으로 제어**하는 프레임워크. timestep을 T(완전 노이즈)로 고정하면 해당 변수가 주변화(marginalization)되고 0으로 고정하면 조건화(conditioning)되므로, 한 모델이 추론 시점에 policy $p(a|o)$ / forward dynamics $p(o'|o,a)$ / inverse dynamics $p(a|o,o')$ / video prediction $p(o'|o)$을 모두 수행한다. 대규모 로봇 데이터 + action-free 비디오 co-training으로 모방학습 대비 더 일반화되고 강건한 정책을 획득 — 모방학습과 world modeling의 통일.

- **arXiv:** https://arxiv.org/abs/2504.02792
- **Project:** https://weirdlabuw.github.io/uwm/
- **PDF:** `2025_UnifiedWorldModels.pdf`
- **경계 판정 메모:** 목적은 정책 개선이지만 primary objective가 행동+미래 관측의 **결합 디퓨전 손실**로 미래 상태 예측을 대등하게 포함 — 경계 규칙 3에서 world model '소비'가 아닌 '학습'이므로 6번이 아닌 7-2 (루트 README 부록 B 참조)
- **Study note:** `/paper-to-note`로 생성 예정

## Citation

```bibtex
@article{zhu2025unified,
  title   = {Unified World Models: Coupling Video and Action Diffusion for Pretraining on Large Robotic Datasets},
  author  = {Zhu, Chuning and Yu, Raymond and Feng, Siyuan and Burchfiel, Benjamin and Shah, Paarth and Gupta, Abhishek},
  journal = {arXiv preprint arXiv:2504.02792},
  year    = {2025}
}
```
