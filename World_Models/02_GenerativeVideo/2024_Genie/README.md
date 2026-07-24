# Genie: Generative Interactive Environments

**분류:** 7-2 Generative Video & World Simulators · **태그:** 기능=Renderer/Simulator, 조건화=action (unsupervised latent), 도메인=game/robotics, 물리 근거성=implicit

Google DeepMind (Jake Bruce, Michael Dennis, Ashley Edwards, Jack Parker-Holder 외). **ICML 2024.** 무라벨 인터넷 비디오만으로 학습한 최초의 **generative interactive environment** (11B foundation world model). 세 구성요소: (1) ST-transformer 기반 **비디오 토크나이저**, (2) 프레임 쌍 사이의 행동을 GT 라벨 없이 비지도로 추론하는 **latent action model**(VQ 코드북 |A|=8로 제한해 인간 조작성 확보), (3) latent action + 과거 프레임 토큰 조건으로 다음 프레임을 예측하는 MaskGIT **자기회귀 동역학 모델**. 텍스트 생성 이미지·스케치·사진 어떤 프롬프트든 프레임 단위로 조작 가능한 가상 세계로 만들며, 학습된 latent action 공간으로 미관측 비디오에서 정책 모방도 가능함을 시연 (RT-1 로봇 데이터로 일반성 확인).

- **arXiv:** https://arxiv.org/abs/2402.15391 (ICML 2024)
- **PDF:** `2024_Genie.pdf`
- **계보:** Genie → Genie 2 → Genie 3 / Project Genie (7-2 외부 예시) — 실시간 인터랙티브 world model 계열의 기원. 행동 라벨 없는 비디오에서 행동 조건화를 학습한다는 아이디어는 Vid2World(같은 7-2)의 "action-free 데이터 활용" 문제의식과 상통
- **Study note:** `/paper-to-note`로 생성 예정

## Citation

```bibtex
@inproceedings{bruce2024genie,
  title     = {Genie: Generative Interactive Environments},
  author    = {Bruce, Jake and Dennis, Michael and Edwards, Ashley and Parker-Holder, Jack and Shi, Yuge and Hughes, Edward and Lai, Matthew and Mavalankar, Aditi and Steigerwald, Richie and Apps, Chris and Aytar, Yusuf and Bechtle, Sarah and Behbahani, Feryal and Chan, Stephanie and Heess, Nicolas and Gonzalez, Lucy and Osindero, Simon and Ozair, Sherjil and Reed, Scott and Zhang, Jingwei and Zolna, Konrad and Clune, Jeff and de Freitas, Nando and Singh, Satinder and Rockt{\"a}schel, Tim},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2024}
}
```
