패키지 구조

rag_toolkit/
├── pyproject.toml              # pip install -e . 가능
├── .env.example                # API 키 설정 템플릿
├── examples/
│   └── basic_usage.py          # 사용 예제
└── rag_toolkit/
    ├── __init__.py             # 공개 API (RAGConfig, RAGClient, ...)
    ├── config.py               # 설정 관리 (범용화)
    ├── client.py               # 문서 인덱싱/쿼리 클라이언트
    ├── query.py                # 쿼리 유틸리티 + 템플릿
    └── raganything/            # RAG-Anything 라이브러리 (13개 파일)
Paper2Slides에서 제거된 의존성
PROJECT_ROOT / sys.path.insert 하드코딩 제거
from summary.clean import clean_references 제거 → clean_func 선택적 콜백 파라미터로 변경
모든 from raganything.xxx 절대 import → from .xxx 상대 import로 변경
다른 프로젝트에서 사용하는 방법

# 1. 설치
# pip install -e ./rag_toolkit

# 2. 사용
from rag_toolkit import RAGClient, RAGConfig

config = RAGConfig.from_env()  # .env에서 RAG_LLM_API_KEY 읽기

async with RAGClient(config=config) as rag:
    await rag.index("document.pdf")
    answer = await rag.query("What is the main topic?")