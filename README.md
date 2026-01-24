# Scene Understanding & 3D Tasks Taxonomy

## Rendering & Representation (어떻게 표현하고 그릴 것인가?)
* **핵심 키워드**: Photometric Consistency (광도 일관성)
* **분류 근거**: "기하학적 정확성(Geometry)보다는, 렌더링했을 때 입력 이미지와 **'얼마나 똑같이 보이는가'**에 최적화된 표현 방식(Representation)이기 때문입니다."
    * Mesh가 울퉁불퉁해도 렌더링된 결과물(이미지)만 좋으면 성공으로 칩니다.
* **ex)** NeRF, 3DGS, Gaussian Splatting

## Geometry & Structure (어떻게 형상을 복원할 것인가?)
* **핵심 키워드**: Correspondence (대응점) & Global Alignment
* **분류 근거**: "이미지 픽셀 값을 3D 좌표로 직접 매핑(Regression)하여, **'공간적 구조와 관계'**를 해석하는 것이 목표이기 때문입니다."
    * 렌더링 퀄리티보다는 3D 점들이 정확한 위치에 찍히는지가(Metric Accuracy) 중요합니다.
* **ex)** DUSt3R, MVSNet, COLMAP

## Generative 3D (어떻게 보이지 않는 곳을 상상하여 채울 것인가?)
* **핵심 키워드**: Hallucination (생성적 추론) & Prior (사전 지식)
* **분류 근거**: "보이지 않는 곳을 **'계산'**하는 것이 아니라, 학습된 데이터 분포(Prior)를 기반으로 **'상상'**해서 채워 넣기 때문입니다."
    * Input 데이터가 부족해도(Single-view), 그럴듯한(Plausible) 3D 형상을 만들어냅니다.
* **ex)** SAM 3D Objects, DreamFusion, Magic3D

## 6D Pose Estimation (객체가 공간상 어디에, 어떤 자세로 놓여있는가?)
* **핵심 키워드**: Alignment & Registration (정렬 및 정합)
* **분류 근거**: "대상의 형태를 바꾸거나 만드는 것이 아니라, 관측된 데이터(Observation)와 이미 알고 있는 모델(Reference) 사이의 **'공간적 변환 관계(Rotation, Translation)'**를 찾아내는 것이 목표이기 때문입니다."
    * 객체의 형상은 고정체(Rigid Body)로 가정하며, SE(3) 공간 상에서의 최적의 파라미터($R, t$)를 추정하는 것이 핵심입니다.
* **ex)** FoundationPose, MegaPose, PVNet

## Perception & Understanding (무엇인지 인지하고 해석할 것인가?)
* **핵심 키워드:** Semantics (의미) & Feature Extraction (특징 추출)
* **분류 근거:** "주어진 3D 데이터(Point Cloud 등)를 입력받아, 기하학적 형상 너머의 **'의미론적 클래스(Label)나 부위(Part)'**를 식별하는 것이 목표이기 때문입니다."
    * 단순한 좌표값(XYZ)을 고차원의 특징 벡터(Feature Vector)로 변환하여 '이것이 무엇인가'를 판단합니다.
* **하위 분류 (Modality & Method):**
    * **Point Cloud Analysis:** 불규칙한 점군 데이터를 처리하는 백본 네트워크 연구.
* **ex)**
    * **PointMamba (2024):** SSM(State Space Model)을 3D 점군에 적용하여 효율적인 특징 학습.
    * **PointGST:** Graph나 Transformer 기반의 점군 처리.
    * (Classic): PointNet++, DGCNN, Point Transformer.
