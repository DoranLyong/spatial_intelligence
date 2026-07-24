# DreamerV3: Mastering Diverse Control Tasks through World Models

**분류:** 7-1 Latent Prediction · **태그:** 기능=Planner, 조건화=action, 도메인=robotics/game/general, 물리 근거성=implicit

Google DeepMind/Toronto (Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, Timothy Lillicrap). **Nature 640, 2025.** Dreamer 계보의 3세대. RSSM(recurrent state-space model) world model — 인코더 $z_t \sim q_\phi(z_t|h_t,x_t)$, 시퀀스 모델 $h_t = f_\phi(h_{t-1},z_{t-1},a_{t-1})$, dynamics 예측기 $\hat z_t \sim p_\phi(\hat z_t|h_t)$, reward/continue 예측기, 보조 재구성 디코더 — 를 학습하고, actor-critic을 **순수 잠재 공간 상상 롤아웃**(T=16)에서 학습한다. Symlog 변환·free bits·percentile return 정규화 등 강건화 기법으로 **고정 하이퍼파라미터 단일 설정**이 150+ 태스크(Control Suite, Atari, ProcGen, DMLab, Minecraft)에서 전문 알고리즘들을 능가하며, 인간 데이터·커리큘럼 없이 Minecraft 다이아몬드를 최초로 획득했다.

- **DOI:** https://doi.org/10.1038/s41586-025-08744-2 (Nature, open access)
- **PDF:** `2025_DreamerV3.pdf`
- **판정 메모:** 보조 재구성 디코더가 있지만(JEPA와의 차이) 미래 롤아웃은 전적으로 잠재 공간에서 일어남 — 7-1의 판정 기준. 정책을 함께 학습하나 world model '학습'이 본체(UWM 판례, 경계 규칙 3)
- **Study note:** `/paper-to-note`로 생성 예정

## Citation

```bibtex
@article{hafner2025dreamerv3,
  title   = {Mastering diverse control tasks through world models},
  author  = {Hafner, Danijar and Pasukonis, Jurgis and Ba, Jimmy and Lillicrap, Timothy},
  journal = {Nature},
  volume  = {640},
  year    = {2025},
  doi     = {10.1038/s41586-025-08744-2}
}
```
