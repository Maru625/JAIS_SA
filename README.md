# DL-Project-Uncertainty

## 1. 개요

이 프로젝트는 불확실성(Uncertainty) 측정이 가능한 딥러닝 모델을 개발하고, MLOps 파이프라인을 통해 효율적으로 실험을 관리하는 것을 목표로 합니다.

- **Hydra**: 설정 파일(YAML)을 통해 모델 아키텍처, 하이퍼파라미터 등을 쉽게 변경하고 관리합니다.
- **MLFlow**: 모든 실험의 하이퍼파라미터, 결과, 학습된 모델 아티팩트를 추적하고 시각적으로 비교합니다.
- **모델**: 베이지안 신경망(BNN)과 몬테카를로 드롭아웃(MC Dropout) DNN 모델을 구현하여 예측의 불확실성을 정량화합니다.

---

## 2. 프로젝트 구조

```
/
├── configs/                  # Hydra 설정 파일
│   ├── config.yaml           # 메인 설정 파일
│   ├── data/                 # 데이터 관련 설정
│   │   └── default.yaml
│   └── model/                # 모델 구조 및 하이퍼파라미터 설정
│       ├── bnn.yaml
│       └── mc_dropout_dnn.yaml
├── data/                     # 데이터 저장 위치 (예시)
│   ├── raw/
│   └── processed/
├── src/                      # 소스 코드
│   ├── preprocess.py         # 데이터 전처리 스크립트
│   ├── train.py              # 모델 학습 스크립트
│   ├── evaluate.py           # 모델 평가 스크립트
│   ├── model.py              # 모델 아키텍처 정의
│   └── utils.py              # 유틸리티 함수
├── .venv/                    # Python 가상환경
├── mlflow_runs/              # MLFlow 실행 결과 저장
├── outputs/                  # Hydra 실행 결과 저장
├── .gitignore
├── pyproject.toml            # 프로젝트 의존성 및 메타데이터
└── README.md                 # 프로젝트 설명 파일
```

---

## 3. 파일 설명

### 3.1. `pyproject.toml`

- 프로젝트의 이름, 버전, 그리고 필요한 모든 Python 라이브러리(`hydra-core`, `mlflow`, `torch`, `torchbnn` 등)가 정의된 파일입니다.
- `uv pip install -e .` 명령어를 통해 이곳에 명시된 의존성을 가상환경에 설치합니다.

### 3.2. `configs/` (Hydra 설정)

- **`config.yaml`**: 메인 설정 파일. `model`과 `data`의 기본 조합을 지정하고, MLFlow에서 사용할 프로젝트 이름을 정의합니다.
- **`data/default.yaml`**: 데이터 경로, 배치 사이즈 등 데이터 관련 설정을 관리합니다.
- **`model/*.yaml`**: 각 모델(`bnn.yaml`, `mc_dropout_dnn.yaml`)의 아키텍처, 하이퍼파라미터(학습률, 에포크, 드롭아웃 비율 등)를 개별적으로 정의합니다.

### 3.3. `src/` (소스 코드)

- **`train.py`**:
    - **역할**: 모델 학습을 위한 메인 스크립트입니다.
    - **기능**:
        - `@hydra.main` 데코레이터를 통해 `configs`의 설정들을 불러옵니다.
        - MLFlow를 초기화하고 실험 기록을 시작합니다.
        - 설정에 따라 BNN 또는 MC Dropout 모델을 불러와 학습을 진행합니다.
        - BNN의 경우 ELBO 손실(NLL + KL Divergence), MC Dropout의 경우 Cross-Entropy 손실을 계산합니다.
        - 학습 과정의 지표(loss 등)와 최종 모델을 MLFlow에 저장합니다.

- **`model.py`**:
    - **역할**: 신경망 모델의 아키텍처를 정의합니다.
    - **기능**:
        - `get_model` 함수가 설정 파일(`bnn` 또는 `mc_dropout_dnn`)에 따라 적절한 모델 클래스를 반환합니다.
        - `BNN`: `torchbnn` 라이브러리를 사용하여 베이지안 레이어로 구성된 모델입니다.
        - `McDropoutDnn`: `nn.Dropout` 레이어를 포함하는 표준적인 DNN 모델입니다.

- **`evaluate.py`**:
    - **역할**: 학습이 완료된 모델의 성능과 불확실성을 평가합니다.
    - **기능**:
        - MLFlow에 저장된 모델을 `run_id`를 통해 불러옵니다.
        - **MC Dropout 모델**: 추론 시에도 드롭아웃을 활성화(`model.train()`)하고, 여러 번의 예측을 통해 평균 예측값과 분산(불확실성)을 계산합니다.
        - **BNN 모델**: 모델의 예측을 통해 정확도를 계산합니다.
        - 최종 정확도와 평균 불확실성을 MLFlow에 추가로 기록합니다.

- **`preprocess.py`**:
    - **역할**: 원본 데이터를 학습에 사용할 수 있는 형태로 가공합니다.
    - **기능**: 현재는 더미 데이터를 생성하는 예시 코드가 포함되어 있습니다.

---

## 4. 사용법

### 4.1. 초기 설정

1.  **가상환경 활성화**:
    ```bash
    # Windows
    .venv\Scripts\activate
    ```

2.  **의존성 설치**:
    ```bash
    uv pip install -e .
    ```

### 4.2. 모델 학습

- **단일 실험 실행** (예: BNN 모델만 학습)
  ```bash
  python src/train.py model=bnn
  ```

- **병렬 실험 실행** (BNN과 MC Dropout 모델 동시 학습)
  ```bash
  python src/train.py --multirun model=bnn,mc_dropout_dnn
  ```

### 4.3. 실험 결과 확인

- 아래 명령어를 실행하여 MLFlow UI를 웹 브라우저에서 엽니다.
  ```bash
  mlflow ui --backend-store-uri file:///%cd%/mlflow_runs
  ```
- UI에서 각 실험의 하이퍼파라미터, 손실 그래프를 비교하고, 가장 성능이 좋은 모델을 확인할 수 있습니다.

### 4.4. 모델 평가

1.  MLFlow UI에서 평가하고 싶은 실험의 **Run ID**를 복사합니다.
2.  `src/evaluate.py` 파일을 열어 `run_id = "<your_run_id>"` 부분을 복사한 Run ID로 수정합니다.
3.  아래 명령어를 실행하여 평가를 진행합니다.
    ```bash
    python src/evaluate.py
    ```
- 평가가 완료되면, 해당 Run ID의 MLFlow 실험 페이지에 `test_accuracy`와 `avg_test_uncertainty`가 기록됩니다.
