# B-spline Policy: Accelerating Manipulation Policies via B-spline Action Representations

**분류:** 6-4 Manipulation & Policy Learning

Harvard/MIT/UT Austin (Xiaoshen Han, Haoyu Xiong, Haonan Chen, Chaoqi Liu, Antonio Torralba, Yuke Zhu, Yilun Du). 로봇 조작 정책의 실행 속도를 높이기 위한 **연속 행동 표현**. 이산 action chunk 대신 정책이 B-spline 파라미터(knot 벡터 + control point)를 직접 출력하여 $\mathbf{a}(u)=\sum_i N_{i,p}(u)\,c_i$의 연속 궤적을 만든다. (1) temporal rescaling으로 재학습 없이 임의 속도·제어 주파수 실행, (2) 추론 시 세그먼트 정렬로 chunk 경계 불연속 제거, (3) FITPACK 기반 adaptive knot 배치로 시연 궤적을 고곡률 구간에 밀도를 몰아주며 근사. Diffusion Policy·ACT 백본에 plug-and-play로 통합되어 성공률을 유지하면서 태스크 완료 시간을 대폭 단축.

- **arXiv:** https://arxiv.org/abs/2607.09648 (2026)
- **Project:** https://B-spline-policy.github.io
- **PDF:** `2026_BsplinePolicy.pdf`
- **판정 메모:** 정책(행동 궤적 예측) 손실만 최적화하고 world model 학습이 없음 — 경계 규칙 3의 명백한 6번. 원본 파일명의 '2025'는 오기(arXiv 2607 = 2026년 7월)로 수납 시 교정
- **Study note:** `/paper-to-note`로 생성 예정

## Citation

```bibtex
@article{han2026bspline,
  title   = {B-spline Policy: Accelerating Manipulation Policies via B-spline Action Representations},
  author  = {Han, Xiaoshen and Xiong, Haoyu and Chen, Haonan and Liu, Chaoqi and Torralba, Antonio and Zhu, Yuke and Du, Yilun},
  journal = {arXiv preprint arXiv:2607.09648},
  year    = {2026}
}
```
