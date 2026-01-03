"""
LangGraph 워크플로우 상태 정의.

이 모듈은 VOB/VOC 분석 워크플로우에서 사용되는 GraphState를 정의합니다.
TypedDict를 사용하여 각 노드 간 전달되는 상태를 명확하게 타입 정의합니다.

사용 예시:
    from state import GraphState
    from langgraph.graph import StateGraph
    
    workflow = StateGraph(GraphState)
"""

from __future__ import annotations

from typing import TypedDict, Optional, Annotated
from operator import add

from .schemas import VOBFeatures, VOCFeatures, GapAnalysisResult


class GraphState(TypedDict, total=False):
    """
    LangGraph 워크플로우의 전체 상태를 정의하는 TypedDict.
    
    워크플로우의 각 노드는 이 상태를 읽고 업데이트합니다.
    `total=False`로 설정하여 모든 필드가 선택적임을 나타냅니다.
    
    Attributes:
        # === 입력 필드 ===
        target_url: 분석 대상 제품 페이지 URL
        
        # === 로딩 상태 ===
        vob_raw_html: VOB 수집을 위한 원본 HTML
        voc_raw_html: VOC 수집을 위한 원본 HTML (리뷰/Q&A 페이지)
        
        # === 추출된 데이터 ===
        vob_features: 추출된 VOB 데이터 (Pydantic 모델)
        voc_features: 추출된 VOC 데이터 (Pydantic 모델)
        
        # === 분석 결과 ===
        gap_analysis_results: 갭 분석 결과 목록
        final_insights: 최종 인사이트 요약
        
        # === 오류 처리 ===
        errors: 워크플로우 중 발생한 오류 목록 (누적)
        
        # === 메타데이터 ===
        current_step: 현재 실행 중인 단계
        execution_log: 실행 로그 (누적)
    """
    
    # === 입력 필드 ===
    target_url: str
    
    # === 로딩 상태 ===
    vob_raw_html: Optional[str]
    voc_raw_html: Optional[str]
    
    # === 추출된 데이터 ===
    vob_features: Optional[VOBFeatures]
    voc_features: Optional[VOCFeatures]
    
    # === 분석 결과 ===
    gap_analysis_results: list[GapAnalysisResult]
    final_insights: Optional[str]
    
    # === 오류 처리 (Annotated로 reducer 정의 - 누적 방식) ===
    errors: Annotated[list[str], add]
    
    # === 메타데이터 ===
    current_step: str
    execution_log: Annotated[list[str], add]


class URLCacheEntry(TypedDict):
    """
    URL 캐시 엔트리.
    
    URL 데이터를 캐싱할 때 사용하는 구조체입니다.
    
    Attributes:
        url: 캐싱된 URL
        html_content: HTML 콘텐츠
        fetched_at: 수집 시간 (ISO format)
        ttl_seconds: TTL (초 단위)
    """
    
    url: str
    html_content: str
    fetched_at: str
    ttl_seconds: int


class NodeResult(TypedDict, total=False):
    """
    노드 실행 결과 표준 구조.
    
    각 노드가 반환하는 상태 업데이트의 표준 형식입니다.
    
    Attributes:
        success: 실행 성공 여부
        data: 결과 데이터 (노드별로 다름)
        error_message: 오류 발생 시 오류 메시지
        execution_time_ms: 실행 시간 (밀리초)
    """
    
    success: bool
    data: dict
    error_message: Optional[str]
    execution_time_ms: float
