"""
VOB/VOC 분석 워크플로우 유틸리티 모듈.

이 모듈은 워크플로우 전반에서 사용되는 유틸리티 함수들을 제공합니다:
- Logger: 일별 로그 파일 생성 및 관리
- Token Counter: tiktoken을 이용한 토큰 수 계산
- Retry Logic: tenacity를 이용한 재시도 데코레이터
- Mock Data Generator: 테스트용 더미 데이터 생성

사용 예시:
    from utils import get_logger, count_tokens, retry_with_exponential_backoff
    
    logger = get_logger("my_module")
    token_count = count_tokens("Hello, world!")
"""

from __future__ import annotations

import os
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Union, Callable, Any, TypeVar
from functools import wraps

import tiktoken
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

from .schemas import (
    VOBFeatures,
    VOCFeatures,
    GapAnalysisResult,
    ReviewSummary,
    QASummary,
)

# =============================================================================
# Configuration
# =============================================================================

# 환경 변수에서 설정 읽기
USE_MOCK = os.getenv("USE_MOCK", "false").lower() == "true"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_DIR = Path(__file__).parent.parent / "logs"

# Type variable for generic retry decorator
F = TypeVar("F", bound=Callable[..., Any])


# =============================================================================
# Logger 설정
# =============================================================================

def get_logger(
    name: str,
    level: str = LOG_LEVEL,
    log_dir: Path = LOG_DIR,
) -> logging.Logger:
    """
    일별 로그 파일을 생성하는 로거를 반환합니다.
    
    Args:
        name: 로거 이름 (모듈명 권장)
        level: 로그 레벨 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: 로그 파일 저장 디렉토리
        
    Returns:
        설정된 Logger 인스턴스
        
    Example:
        logger = get_logger("my_module")
        logger.info("Processing started")
    """
    # 로그 디렉토리 생성
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 일별 로그 파일명
    today = datetime.now().strftime("%Y-%m-%d")
    log_file = log_dir / f"{today}.log"
    
    # 로거 생성
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level, logging.INFO))
    
    # 이미 핸들러가 있으면 추가하지 않음 (중복 방지)
    if not logger.handlers:
        # 파일 핸들러
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        
        # 콘솔 핸들러
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, level, logging.INFO))
        
        # 포맷터
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return logger


# 기본 로거
_default_logger = get_logger("vob_voc_agent")


# =============================================================================
# Token Counter
# =============================================================================

# tiktoken 인코딩 캐시
_encoding_cache: dict[str, tiktoken.Encoding] = {}


def get_encoding(model: str = "gpt-4o") -> tiktoken.Encoding:
    """
    모델에 맞는 tiktoken 인코딩을 반환합니다 (캐시 사용).
    
    Args:
        model: OpenAI 모델명
        
    Returns:
        tiktoken Encoding 인스턴스
    """
    if model not in _encoding_cache:
        try:
            _encoding_cache[model] = tiktoken.encoding_for_model(model)
        except KeyError:
            # 알 수 없는 모델의 경우 cl100k_base 사용 (GPT-4 기본)
            _encoding_cache[model] = tiktoken.get_encoding("cl100k_base")
    return _encoding_cache[model]


def count_tokens(
    text: str,
    model: str = "gpt-4o",
) -> int:
    """
    텍스트의 토큰 수를 계산합니다.
    
    Args:
        text: 토큰 수를 계산할 텍스트
        model: OpenAI 모델명 (토큰화 방식 결정)
        
    Returns:
        토큰 수
        
    Example:
        count = count_tokens("Hello, world!")
        print(f"Token count: {count}")
    """
    if not text:
        return 0
    
    encoding = get_encoding(model)
    return len(encoding.encode(text))


def count_image_tokens(
    width: int,
    height: int,
    detail: str = "high",
) -> int:
    """
    이미지의 토큰 수를 계산합니다 (OpenAI Vision API 기준).
    
    OpenAI의 이미지 토큰 계산 공식:
    - low detail: 85 tokens (고정)
    - high detail: 129 tokens (512x512 타일당) + 85 tokens (기본)
    
    Args:
        width: 이미지 너비 (픽셀)
        height: 이미지 높이 (픽셀)
        detail: 디테일 수준 ("low" 또는 "high")
        
    Returns:
        토큰 수
    """
    if detail == "low":
        return 85
    
    # high detail 계산
    # 먼저 이미지를 2048x2048 내로 축소
    max_dim = 2048
    if max(width, height) > max_dim:
        scale = max_dim / max(width, height)
        width = int(width * scale)
        height = int(height * scale)
    
    # 가장 짧은 면이 768px이 되도록 축소
    min_dim = 768
    if min(width, height) > min_dim:
        scale = min_dim / min(width, height)
        width = int(width * scale)
        height = int(height * scale)
    
    # 512x512 타일 수 계산
    tiles_x = (width + 511) // 512
    tiles_y = (height + 511) // 512
    total_tiles = tiles_x * tiles_y
    
    # 토큰 계산: 타일당 129토큰 + 기본 85토큰
    return 129 * total_tiles + 85


def estimate_cost(
    input_tokens: int,
    output_tokens: int,
    model: str = "gpt-4o",
) -> float:
    """
    토큰 사용량에 따른 예상 비용을 계산합니다 (USD).
    
    Args:
        input_tokens: 입력 토큰 수
        output_tokens: 출력 토큰 수
        model: 모델명
        
    Returns:
        예상 비용 (USD)
    """
    # 2024년 기준 가격 (1M tokens 당)
    pricing = {
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4-turbo": {"input": 10.00, "output": 30.00},
        "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    }
    
    prices = pricing.get(model, pricing["gpt-4o"])
    
    input_cost = (input_tokens / 1_000_000) * prices["input"]
    output_cost = (output_tokens / 1_000_000) * prices["output"]
    
    return input_cost + output_cost


# =============================================================================
# Retry Logic
# =============================================================================

def retry_with_exponential_backoff(
    max_attempts: int = 3,
    min_wait: float = 1.0,
    max_wait: float = 60.0,
    retry_exceptions: tuple = (Exception,),
) -> Callable[[F], F]:
    """
    지수 백오프를 적용하는 재시도 데코레이터를 생성합니다.
    
    Rate Limit 등의 일시적 오류에 대응하기 위해 사용합니다.
    
    Args:
        max_attempts: 최대 재시도 횟수
        min_wait: 최소 대기 시간 (초)
        max_wait: 최대 대기 시간 (초)
        retry_exceptions: 재시도할 예외 타입들
        
    Returns:
        데코레이터 함수
        
    Example:
        @retry_with_exponential_backoff(max_attempts=3)
        def call_api():
            ...
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            @retry(
                stop=stop_after_attempt(max_attempts),
                wait=wait_exponential(multiplier=1, min=min_wait, max=max_wait),
                retry=retry_if_exception_type(retry_exceptions),
                before_sleep=before_sleep_log(_default_logger, logging.WARNING),
                reraise=True,
            )
            def _inner() -> Any:
                return func(*args, **kwargs)
            return _inner()
        
        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            @retry(
                stop=stop_after_attempt(max_attempts),
                wait=wait_exponential(multiplier=1, min=min_wait, max=max_wait),
                retry=retry_if_exception_type(retry_exceptions),
                before_sleep=before_sleep_log(_default_logger, logging.WARNING),
                reraise=True,
            )
            async def _inner() -> Any:
                return await func(*args, **kwargs)
            return await _inner()
        
        # 비동기 함수 여부에 따라 적절한 래퍼 반환
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper  # type: ignore
    
    return decorator


# =============================================================================
# Mock Data Generator
# =============================================================================

def generate_mock_vob(
    product_name: str = "테스트 에센스",
    brand_name: str = "테스트 브랜드",
) -> VOBFeatures:
    """
    테스트용 VOB 더미 데이터를 생성합니다.
    
    USE_MOCK=True 환경에서 실제 API 호출 대신 사용됩니다.
    
    Args:
        product_name: 제품명
        brand_name: 브랜드명
        
    Returns:
        VOBFeatures 인스턴스
    """
    return VOBFeatures(
        product_name=product_name,
        brand_name=brand_name,
        price=random.randint(20000, 80000),
        discounted_price=random.randint(15000, 60000),
        volume=random.choice(["30ml", "50ml", "100ml", "150ml"]),
        product_category=random.choice(["스킨케어", "에센스", "크림", "세럼"]),
        key_benefits=[
            "깊은 보습",
            "피부 진정",
            "탄력 개선",
            "피부결 정돈",
        ][:random.randint(2, 4)],
        key_ingredients=[
            "히알루론산",
            "세라마이드",
            "나이아신아마이드",
            "비타민C",
            "레티놀",
        ][:random.randint(2, 4)],
        usage_instructions="세안 후 토너 다음 단계에서 적당량을 덜어 얼굴 전체에 부드럽게 펴 바릅니다.",
        target_skin_type=random.choice(["모든 피부", "건성", "지성", "복합성", "민감성"]),
        certifications=random.sample(["비건", "유기농", "더마테스트", "무향료"], k=random.randint(0, 2)),
        claims=[
            "48시간 보습 지속",
            "피부 장벽 강화",
            "첫 사용부터 느껴지는 효과",
        ][:random.randint(1, 3)],
        raw_text="[Mock] 이것은 테스트용 더미 데이터입니다.",
    )


def generate_mock_voc(
    total_reviews: int = 150,
    average_rating: float = 4.2,
) -> VOCFeatures:
    """
    테스트용 VOC 더미 데이터를 생성합니다.
    
    USE_MOCK=True 환경에서 실제 API 호출 대신 사용됩니다.
    
    Args:
        total_reviews: 전체 리뷰 수
        average_rating: 평균 평점
        
    Returns:
        VOCFeatures 인스턴스
    """
    # 평점 분포 생성 (평균에 맞게 조정)
    rating_dist = {
        5: random.uniform(0.3, 0.5),
        4: random.uniform(0.2, 0.35),
        3: random.uniform(0.1, 0.2),
        2: random.uniform(0.05, 0.1),
        1: random.uniform(0.02, 0.08),
    }
    # 정규화
    total = sum(rating_dist.values())
    rating_dist = {k: v / total for k, v in rating_dist.items()}
    
    # 샘플 리뷰 요약 생성
    review_samples = [
        ReviewSummary(
            rating=random.randint(4, 5),
            summary="보습력이 정말 좋아요. 다음날 아침까지 촉촉해요.",
            sentiment="positive",
            key_points=["보습력", "지속력", "흡수력"],
        ),
        ReviewSummary(
            rating=random.randint(3, 4),
            summary="괜찮은 제품이에요. 향이 조금 강한 편이에요.",
            sentiment="neutral",
            key_points=["보통", "향"],
        ),
        ReviewSummary(
            rating=random.randint(1, 2),
            summary="제 피부에는 안 맞았어요. 트러블이 생겼어요.",
            sentiment="negative",
            key_points=["트러블", "피부 자극"],
        ),
    ]
    
    # Q&A 요약 생성
    qa_samples = [
        QASummary(
            question_theme="성분 관련",
            frequency=random.randint(5, 15),
            has_official_answer=True,
        ),
        QASummary(
            question_theme="사용 순서",
            frequency=random.randint(3, 10),
            has_official_answer=True,
        ),
        QASummary(
            question_theme="피부 타입 적합성",
            frequency=random.randint(5, 20),
            has_official_answer=False,
        ),
    ]
    
    return VOCFeatures(
        total_review_count=total_reviews,
        average_rating=average_rating,
        rating_distribution=rating_dist,
        positive_themes=[
            "보습력 좋음",
            "흡수가 빠름",
            "순한 사용감",
            "피부 진정 효과",
        ][:random.randint(2, 4)],
        negative_themes=[
            "향이 강함",
            "끈적임",
            "가격 대비 용량 적음",
        ][:random.randint(1, 3)],
        common_concerns=[
            "민감성 피부 사용 가능 여부",
            "다른 제품과 함께 사용 가능 여부",
        ],
        frequently_asked_questions=[
            "레티놀 제품과 함께 사용해도 되나요?",
            "아침/저녁 모두 사용 가능한가요?",
            "임산부도 사용할 수 있나요?",
        ][:random.randint(2, 3)],
        review_summaries=random.sample(review_samples, k=random.randint(1, 3)),
        qa_summaries=qa_samples,
        raw_reviews=["[Mock] 더미 리뷰 데이터입니다."],
        raw_qas=["[Mock] 더미 Q&A 데이터입니다."],
    )


def generate_mock_gap_analysis(
    vob: VOBFeatures,
    voc: VOCFeatures,
) -> list[GapAnalysisResult]:
    """
    테스트용 갭 분석 결과를 생성합니다.
    
    Args:
        vob: VOB 데이터
        voc: VOC 데이터
        
    Returns:
        GapAnalysisResult 리스트
    """
    gaps = [
        GapAnalysisResult(
            gap_category="정보 부족",
            vob_claim="민감성 피부에도 적합",
            voc_reality="민감성 피부 적합성에 대한 질문이 자주 올라옴",
            severity="medium",
            recommendation="상세 페이지에 민감성 피부 임상 테스트 결과 추가",
        ),
        GapAnalysisResult(
            gap_category="기대 불일치",
            vob_claim="48시간 보습 지속",
            voc_reality="실제 리뷰에서 6-8시간 정도라는 피드백 다수",
            severity="high",
            recommendation="클레임 수정 또는 사용법 가이드 강화",
        ),
        GapAnalysisResult(
            gap_category="숨겨진 가치",
            vob_claim="보습 효과 강조",
            voc_reality="진정 효과에 대한 긍정 리뷰 다수",
            severity="low",
            recommendation="진정 효과를 마케팅 포인트로 추가",
        ),
    ]
    
    return random.sample(gaps, k=random.randint(1, 3))


def is_mock_mode() -> bool:
    """
    현재 Mock 모드인지 확인합니다.
    
    Returns:
        USE_MOCK 환경변수 값
    """
    return USE_MOCK
