"""
VOB/VOC 데이터 분석을 위한 Pydantic 스키마 정의.

이 모듈은 Voice of Business(VOB)와 Voice of Customer(VOC) 데이터를
구조화하기 위한 Pydantic V2 모델을 정의합니다.

사용 예시:
    from schemas import VOBFeatures, VOCFeatures
    
    vob = VOBFeatures(
        product_name="예시 제품",
        brand_name="브랜드명",
        price=29900,
        ...
    )
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, ConfigDict


class VOBFeatures(BaseModel):
    """
    Voice of Business(VOB) - 브랜드 관점의 제품 정보.
    
    제품 상세 페이지에서 브랜드가 제공하는 공식 정보를 구조화합니다.
    가격, 혜택, 용량, 성분, 사용법 등 브랜드가 전달하고자 하는 핵심 메시지를 포함합니다.
    
    Attributes:
        product_name: 제품명 (필수)
        brand_name: 브랜드명 (필수)
        price: 정가 (원 단위)
        discounted_price: 할인가 (원 단위, 없으면 None)
        volume: 용량/중량 (예: "50ml", "30g")
        key_benefits: 핵심 효능/혜택 목록
        key_ingredients: 주요 성분 목록
        usage_instructions: 사용 방법
        target_skin_type: 권장 피부 타입 (예: "지성", "건성", "복합성")
        product_category: 제품 카테고리 (예: "스킨케어", "메이크업")
        certifications: 인증 정보 (예: "비건", "유기농")
        claims: 제품 클레임/주장 (예: "피부 보습 72시간 지속")
        raw_text: 원본 텍스트 데이터 (분석용)
    """
    
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_default=True,
        extra="forbid",
    )
    
    # 필수 필드
    product_name: str = Field(
        ...,
        min_length=1,
        description="제품명 (필수)"
    )
    brand_name: str = Field(
        ...,
        min_length=1,
        description="브랜드명 (필수)"
    )
    
    # 가격 정보
    price: Optional[int] = Field(
        default=None,
        ge=0,
        description="정가 (원 단위)"
    )
    discounted_price: Optional[int] = Field(
        default=None,
        ge=0,
        description="할인가 (원 단위)"
    )
    
    # 제품 기본 정보
    volume: Optional[str] = Field(
        default=None,
        description="용량/중량 (예: '50ml', '30g')"
    )
    product_category: Optional[str] = Field(
        default=None,
        description="제품 카테고리"
    )
    
    # 효능 및 성분
    key_benefits: list[str] = Field(
        default_factory=list,
        description="핵심 효능/혜택 목록"
    )
    key_ingredients: list[str] = Field(
        default_factory=list,
        description="주요 성분 목록"
    )
    
    # 사용 정보
    usage_instructions: Optional[str] = Field(
        default=None,
        description="사용 방법"
    )
    target_skin_type: Optional[str] = Field(
        default=None,
        description="권장 피부 타입"
    )
    
    # 인증 및 클레임
    certifications: list[str] = Field(
        default_factory=list,
        description="인증 정보 목록"
    )
    claims: list[str] = Field(
        default_factory=list,
        description="제품 클레임/주장 목록"
    )
    
    # 원본 데이터
    raw_text: Optional[str] = Field(
        default=None,
        description="원본 텍스트 데이터"
    )


class ReviewSummary(BaseModel):
    """
    개별 리뷰 요약 정보.
    
    Attributes:
        rating: 평점 (1-5)
        summary: 리뷰 요약
        sentiment: 감성 분류 (positive/negative/neutral)
        key_points: 핵심 언급 포인트
    """
    
    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
    )
    
    rating: int = Field(
        ...,
        ge=1,
        le=5,
        description="평점 (1-5)"
    )
    summary: str = Field(
        ...,
        description="리뷰 요약"
    )
    sentiment: str = Field(
        default="neutral",
        pattern="^(positive|negative|neutral)$",
        description="감성 분류"
    )
    key_points: list[str] = Field(
        default_factory=list,
        description="핵심 언급 포인트"
    )


class QASummary(BaseModel):
    """
    Q&A 요약 정보.
    
    Attributes:
        question_theme: 질문 주제/테마
        frequency: 유사 질문 빈도
        has_official_answer: 공식 답변 존재 여부
    """
    
    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
    )
    
    question_theme: str = Field(
        ...,
        description="질문 주제/테마"
    )
    frequency: int = Field(
        default=1,
        ge=1,
        description="유사 질문 빈도"
    )
    has_official_answer: bool = Field(
        default=False,
        description="공식 답변 존재 여부"
    )


class VOCFeatures(BaseModel):
    """
    Voice of Customer(VOC) - 고객 관점의 제품 피드백.
    
    제품에 대한 고객 리뷰, Q&A, 평점 분포 등 고객의 실제 목소리를 구조화합니다.
    긍정/부정 피드백, 자주 묻는 질문, 불만 사항 등을 포함합니다.
    
    Attributes:
        total_review_count: 전체 리뷰 수
        average_rating: 평균 평점
        rating_distribution: 평점 분포 (1~5점 각각의 비율)
        positive_themes: 긍정적 피드백 테마
        negative_themes: 부정적 피드백 테마
        common_concerns: 자주 언급되는 우려/불만
        frequently_asked_questions: 자주 묻는 질문 테마
        review_summaries: 주요 리뷰 요약 목록
        qa_summaries: Q&A 요약 목록
        raw_reviews: 원본 리뷰 텍스트 목록 (분석용)
        raw_qas: 원본 Q&A 텍스트 목록 (분석용)
    """
    
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_default=True,
        extra="forbid",
    )
    
    # 리뷰 통계
    total_review_count: int = Field(
        default=0,
        ge=0,
        description="전체 리뷰 수"
    )
    average_rating: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=5.0,
        description="평균 평점 (0.0-5.0)"
    )
    rating_distribution: dict[int, float] = Field(
        default_factory=lambda: {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0},
        description="평점 분포 (1~5점 각각의 비율, 0.0-1.0)"
    )
    
    # 피드백 테마
    positive_themes: list[str] = Field(
        default_factory=list,
        description="긍정적 피드백 테마"
    )
    negative_themes: list[str] = Field(
        default_factory=list,
        description="부정적 피드백 테마"
    )
    common_concerns: list[str] = Field(
        default_factory=list,
        description="자주 언급되는 우려/불만"
    )
    
    # Q&A 정보
    frequently_asked_questions: list[str] = Field(
        default_factory=list,
        description="자주 묻는 질문 테마"
    )
    
    # 상세 요약
    review_summaries: list[ReviewSummary] = Field(
        default_factory=list,
        description="주요 리뷰 요약 목록"
    )
    qa_summaries: list[QASummary] = Field(
        default_factory=list,
        description="Q&A 요약 목록"
    )
    
    # 원본 데이터
    raw_reviews: list[str] = Field(
        default_factory=list,
        description="원본 리뷰 텍스트 목록"
    )
    raw_qas: list[str] = Field(
        default_factory=list,
        description="원본 Q&A 텍스트 목록"
    )


class GapAnalysisResult(BaseModel):
    """
    VOB와 VOC 간 갭 분석 결과.
    
    Attributes:
        gap_category: 갭 카테고리 (예: "정보 부족", "기대 불일치")
        vob_claim: 브랜드 주장
        voc_reality: 고객 현실/피드백
        severity: 심각도 (low/medium/high)
        recommendation: 개선 권고사항
    """
    
    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
    )
    
    gap_category: str = Field(
        ...,
        description="갭 카테고리"
    )
    vob_claim: str = Field(
        ...,
        description="브랜드 주장"
    )
    voc_reality: str = Field(
        ...,
        description="고객 현실/피드백"
    )
    severity: str = Field(
        default="medium",
        pattern="^(low|medium|high)$",
        description="심각도"
    )
    recommendation: str = Field(
        ...,
        description="개선 권고사항"
    )
