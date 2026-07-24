# V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning

**분류:** 7-1 Latent Prediction · **태그:** 기능=Planner, 조건화=action, 도메인=robotics/general, 물리 근거성=implicit

Meta FAIR (Assran, Bardes, Fan, Garrido, ..., LeCun, Rabbat, Ballas). 잠재 예측형 world model의 대표작. 2단계 학습: (1) **1M 시간+ 인터넷 비디오**로 mask-denoising 특징 예측(표현 공간, 픽셀 재구성 없음) 사전학습해 최대 1B 인코더 획득 — 모션 이해(SSv2 77.3), 행동 예측(EK-100 39.7 recall@5, +44% SOTA), LLM 정렬 video QA(PerceptionTest 84.0)까지 달성. (2) 인코더를 동결하고 **62시간 무라벨 로봇 데이터(Droid)**만으로 행동 조건부 next-frame 표현 예측 world model **V-JEPA 2-AC**(300M, block-causal attention)를 후학습. 잠재 공간 MPC 계획으로 Franka 팔에서 신규 환경 **제로샷** prehensile 조작(Grasp, Pick-and-Place)을 태스크별 학습·보상 없이 수행.

- **arXiv:** https://arxiv.org/abs/2506.09985 (2025)
- **Code:** https://github.com/facebookresearch/vjepa2
- **PDF:** `2025_VJEPA2.pdf`
- **교차 참조:** I-JEPA (4-0, `Perception/00_UniversalEncoders/2023_IJEPA/`) — 공간 마스킹 표현 학습에서 시간 축 행동 조건 동역학으로 확장된 계보의 기원. 같은 7-1의 LeWorldModel(end-to-end JEPA), AdaJEPA(V-JEPA류의 test-time adaptation)와 직접 연결
- **Study note:** `/paper-to-note`로 생성 예정

## Citation

```bibtex
@article{assran2025vjepa2,
  title   = {V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning},
  author  = {Assran, Mahmoud and Bardes, Adrien and Fan, David and Garrido, Quentin and Howes, Russell and Komeili, Mojtaba and Muckley, Matthew and Rizvi, Ammar and Roberts, Claire and Sinha, Koustuv and Zholus, Artem and Arnaud, Sergio and Gejji, Abha and Martin, Ada and Hogan, Francois Robert and Dugas, Daniel and Bojanowski, Piotr and Khalidov, Vasil and Labatut, Patrick and Massa, Francisco and Szafraniec, Marc and Krishnakumar, Kapil and Li, Yong and Ma, Xiaodong and Chandar, Sarath and Meier, Franziska and LeCun, Yann and Rabbat, Michael and Ballas, Nicolas},
  journal = {arXiv preprint arXiv:2506.09985},
  year    = {2025}
}
```
