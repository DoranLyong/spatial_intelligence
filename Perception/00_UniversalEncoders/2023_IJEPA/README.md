# I-JEPA: Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture

**분류:** 4-0 Universal Encoders

Meta AI/FAIR (Mahmoud Assran, Quentin Duval, Ishan Misra, Piotr Bojanowski, Pascal Vincent, Michael Rabbat, Yann LeCun, Nicolas Ballas). **CVPR 2023.** 수작업 augmentation 없이 고수준 의미 표현을 학습하는 비생성적 자기지도 학습. 단일 context 블록의 표현으로부터 같은 이미지 내 여러 target 블록의 표현을 **임베딩 공간에서** 예측한다(픽셀 재구성 없음). 핵심 설계는 마스킹 전략 — 충분히 큰(의미 단위) target 블록과 공간적으로 분산된 정보성 있는 context 블록. EMA target encoder로 표현 붕괴를 방지하며, MAE 대비 10배 이상 효율적으로(ViT-H/14, 72시간 미만) linear probing·semantic transfer·object counting·depth에서 강한 성능을 달성.

- **arXiv:** https://arxiv.org/abs/2301.08243 (CVPR 2023)
- **PDF:** `2023_IJEPA.pdf` (원 파일명 `2023_JEPA.pdf`에서 정식 명칭으로 교정)
- **판정 메모:** "JEPA"라는 이름에도 불구하고 7-1(World Models)이 아님 — 예측 대상이 **같은 정적 이미지의 공간 마스크 블록**이지 시간 축 미래 상태가 아니며, primary objective가 의미론적 표현 품질(루트 README 부록 B 판례 참조). 단, V-JEPA → LeWorldModel로 이어지는 7-1 JEPA 계보의 **아키텍처 기원**이므로 7-1과 교차 참조
- **Study note:** `/paper-to-note`로 생성 예정

## Citation

```bibtex
@inproceedings{assran2023ijepa,
  title     = {Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture},
  author    = {Assran, Mahmoud and Duval, Quentin and Misra, Ishan and Bojanowski, Piotr and Vincent, Pascal and Rabbat, Michael and LeCun, Yann and Ballas, Nicolas},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2023}
}
```
