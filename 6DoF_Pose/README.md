# 5. 6D Pose Estimation

객체가 공간상 어디에, 어떤 자세로 놓여있는가? — 관측과 참조 모델 사이의 SE(3) 변환 관계 $(R, t)$ 추정이 PRIMARY optimization objective인 논문들.

전체 분류 체계와 배치 규칙은 루트 [README.md](../README.md)를 참조.

| 논문 | 디렉토리 | 비고 |
|---|---|---|
| FoundationPose (2024) | `2024_FoundationPose/` | Novel object 통합 6D 포즈 추정·추적, C++/CUDA 확장 포함 |
| Any6D (2025) | `2025_Any6D/` | Model-free novel object 6D 포즈 + metric scale 추정 |
