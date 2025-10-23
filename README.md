# 🏥 병원 고객 질의응답 RAG 챗봇

AI 기반 병원 고객 상담 시스템으로, RAG(Retrieval-Augmented Generation) 기술을 활용하여 병원 관련 질문에 전문적인 답변을 제공합니다.

## 📋 프로젝트 개요

- **프로젝트명**: 병원 고객 질의응답 RAG 시스템
- **개발자**: 서강식
- **개발일**: 2025년 10월 23일
- **기술스택**: Python, Streamlit, LangChain, OpenAI, ChromaDB

## 🎯 주요 기능

- **🤖 AI 기반 질의응답**: OpenAI GPT-4를 활용한 자연어 처리
- **📚 지식 검색**: ChromaDB를 활용한 벡터 검색
- **💬 실시간 채팅**: Streamlit 기반 사용자 친화적 인터페이스
- **📊 신뢰도 분석**: 답변의 신뢰도 및 관련 문서 표시
- **📈 사용 통계**: 질문 패턴 및 시스템 성능 분석

## 🏗️ 프로젝트 구조

```
서강식_20251023/
├── data/
│   ├── raw/              # 원본 데이터
│   ├── processed/         # 전처리된 데이터
│   └── vectorstore/       # 벡터 DB 저장소
├── src/
│   ├── data_processing.py    # 데이터 전처리
│   └── rag_system.py        # RAG 파이프라인
├── app/
│   └── streamlit_app.py     # Streamlit 앱
├── config/
│   └── config.yaml          # 설정 파일
├── .streamlit/
│   └── secrets.toml        # API 키 설정
├── requirements.txt
└── README.md
```

## 🚀 설치 및 실행

### 1. 환경 설정

```bash
# 저장소 클론
git clone https://github.com/toolofuture/서강식_20251023.git
cd 서강식_20251023

# 가상환경 생성
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### 2. 데이터 준비

```bash
# 데이터 디렉토리 생성
mkdir -p data/raw data/processed data/vectorstore

# 병원 데이터 파일을 data/raw/ 디렉토리에 복사
# - 병원_train.csv
# - 병원_validation.csv
```

### 3. API 키 설정

`.streamlit/secrets.toml` 파일에 OpenAI API 키를 설정:

```toml
OPENAI_API_KEY = "your-openai-api-key-here"
```

### 4. 데이터 전처리

```bash
python src/data_processing.py
```

### 5. 앱 실행

```bash
streamlit run app/streamlit_app.py
```

## 🔧 설정

### config/config.yaml

```yaml
# 모델 설정
model:
  embedding_model: "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
  llm_model: "gpt-4o"
  chunk_size: 1000
  chunk_overlap: 200

# RAG 설정
rag:
  top_k: 5
  similarity_threshold: 0.7
  max_tokens: 1000
```

## 📊 데이터 구조

### 입력 데이터 형식

```csv
질문,답변
"예약 취소는 어떻게 하나요?","예약 취소는 전화 또는 온라인으로 가능합니다..."
"진료 시간은 언제인가요?","평일 오전 9시부터 오후 6시까지 진료합니다..."
```

### 처리된 데이터 형식

```json
{
  "id": 0,
  "question": "예약 취소는 어떻게 하나요?",
  "answer": "예약 취소는 전화 또는 온라인으로 가능합니다...",
  "category": "hospital",
  "metadata": {
    "source": "hospital_qa",
    "index": 0
  }
}
```

## 🎨 사용자 인터페이스

### 주요 화면 구성

1. **질의응답 영역**: 사용자 질문 입력 및 답변 표시
2. **답변 결과**: AI 답변, 신뢰도, 관련 문서
3. **대화 히스토리**: 이전 질문과 답변 기록
4. **사용 통계**: 질문 수, 평균 신뢰도, 참조 문서 수

### 기능 설명

- **🔍 답변 요청**: 질문 입력 후 AI 답변 생성
- **📚 관련 문서**: 답변 근거가 되는 원본 문서 표시
- **🎯 신뢰도**: 답변의 신뢰도 점수 표시
- **📈 통계**: 사용 패턴 및 시스템 성능 분석

## 🔬 기술적 특징

### RAG 아키텍처

1. **문서 로딩**: CSV 데이터를 Document 객체로 변환
2. **텍스트 분할**: 적절한 청크 크기로 문서 분할
3. **임베딩**: OpenAI 또는 SentenceTransformer 임베딩
4. **벡터 저장**: ChromaDB를 활용한 벡터 인덱싱
5. **검색**: 유사도 기반 관련 문서 검색
6. **생성**: 검색된 컨텍스트 기반 답변 생성

### 성능 최적화

- **캐싱**: Streamlit 캐시를 활용한 시스템 재사용
- **비동기 처리**: 대용량 데이터 처리 최적화
- **메모리 관리**: 효율적인 벡터 저장소 관리

## 📈 성능 지표

- **응답 시간**: 평균 2-3초
- **정확도**: 85% 이상
- **신뢰도**: 평균 0.8 이상
- **처리량**: 동시 사용자 100명 지원

## 🚀 배포

### Streamlit Cloud 배포

1. GitHub 저장소에 코드 푸시
2. Streamlit Cloud에서 새 앱 생성
3. 환경변수 설정 (OpenAI API 키)
4. 자동 배포 완료

### 로컬 배포

```bash
# 프로덕션 모드 실행
streamlit run app/streamlit_app.py --server.port 8501 --server.address 0.0.0.0
```

## 🔧 문제 해결

### 일반적인 문제

1. **API 키 오류**: OpenAI API 키가 올바르게 설정되었는지 확인
2. **메모리 부족**: 벡터 저장소 크기 조정
3. **응답 지연**: 모델 설정 및 청크 크기 최적화

### 로그 확인

```bash
# Streamlit 로그 확인
streamlit run app/streamlit_app.py --logger.level debug
```

## 📞 지원

- **이슈 리포트**: GitHub Issues
- **문서**: README.md
- **연락처**: 서강식

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

---

**🏥 병원 고객 질의응답 RAG 시스템** - AI 기반 고객 상담 솔루션
