# VOB/VOC Gap Analysis Agent

LangGraph 기반 VOB(Voice of Business)와 VOC(Voice of Customer) 데이터 분석 워크플로우입니다.
제품 상세 페이지의 브랜드 제공 정보와 고객 리뷰/Q&A 데이터를 수집하고, 이들 간의 갭을 분석하여 인사이트를 도출합니다.

## 프로젝트 구조

```
├── data/
│   └── samples/          # 샘플 데이터 파일
├── logs/                  # 로그 파일
├── outputs/
│   └── intermediate/      # 중간 결과물
├── src/
│   ├── schemas.py         # Pydantic 데이터 모델 (VOBFeatures, VOCFeatures)
│   └── state.py           # LangGraph 상태 정의 (GraphState)
├── .env.example           # 환경변수 템플릿
├── .gitignore
├── requirements.txt
└── README.md
```

## 설치 방법

### 1. 저장소 클론

```bash
git clone <repository-url>
cd 01_amore_agent_mvp
```

### 2. 가상환경 생성 및 활성화

```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate     # Windows
```

### 3. 의존성 설치

```bash
pip install -r requirements.txt
```

### 4. 환경변수 설정

```bash
cp .env.example .env
# .env 파일을 편집하여 API 키 입력
```

## 핵심 컴포넌트

### Data Structures

- **VOBFeatures**: 브랜드 관점 데이터 (제품명, 가격, 혜택, 성분 등)
- **VOCFeatures**: 고객 관점 데이터 (리뷰, Q&A, 평점 분포 등)
- **GraphState**: LangGraph 워크플로우 상태 관리

## 개발 요구사항

- Python 3.10+
- Pydantic V2 (2.0 이상)
- LangGraph 0.2.0 이상

## 라이선스

MIT License
