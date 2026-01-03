"""
LangGraph 노드 함수 구현 모듈.

이 모듈은 VOB/VOC 분석 워크플로우의 각 단계에 해당하는 노드 함수들을 포함합니다.
각 노드는 GraphState를 입력받아 처리 후 업데이트된 GraphState를 반환합니다.

주요 노드:
- load_data_node: 데이터 로드
- extract_vob_features: VOB 특징 추출 (LLM/Mock)
- extract_voc_features: VOC 특징 추출 (LLM/Mock)
- gap_analysis_node: VOB-VOC 갭 분석 (LLM/Mock)
- save_results_node: 결과 저장
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from .state import GraphState, NodeResult
from .utils import (
    get_logger,
    is_mock_mode,
    generate_mock_vob,
    generate_mock_voc,
    generate_mock_gap_analysis,
)
from .schemas import GapAnalysisResult

# 로거 초기화
logger = get_logger("nodes")
output_dir = Path("outputs")


async def load_data_node(state: GraphState) -> GraphState:
    """
    초기 데이터를 로드하는 노드.
    
    URL을 확인하고 필요한 경우 HTML 콘텐츠를 가져옵니다.
    Mock 모드일 경우 플레이스홀더 데이터를 설정합니다.
    
    Args:
        state: 현재 그래프 상태
        
    Returns:
        업데이트된 그래프 상태
    """
    start_time = time.time()
    url = state.get("target_url", "")
    logger.info(f"Starting load_data_node for URL: {url}")
    
    try:
        # 상태 초기화 (필요한 경우)
        if "errors" not in state:
            state["errors"] = []
        if "execution_log" not in state:
            state["execution_log"] = []
            
        state["current_step"] = "loading"
        
        # Mock 모드 확인
        if is_mock_mode():
            logger.info("Running in MOCK mode. Skipping actual data fetch.")
            state["vob_raw_html"] = "<html><body>[Mock] VOB Content</body></html>"
            state["voc_raw_html"] = "<html><body>[Mock] VOC Content</body></html>"
            state["execution_log"].append(f"Loaded mock data for {url}")
        else:
            # TODO: 실제 데이터 페칭 로직 구현 (추후)
            # 현재는 리얼 모드でも 플레이스홀더 사용
            logger.warning("Real data fetch not implemented yet. Using placeholder.")
            state["vob_raw_html"] = "<html><body>Placeholder</body></html>"
            state["voc_raw_html"] = "<html><body>Placeholder</body></html>"
            
        return state
        
    except Exception as e:
        error_msg = f"Error in load_data_node: {str(e)}"
        logger.error(error_msg, exc_info=True)
        state["errors"].append(error_msg)
        return state


async def extract_vob_features(state: GraphState) -> GraphState:
    """
    VOB(Voice of Business) 특징을 추출하는 노드.
    
    제품 상세 페이지에서 브랜드가 제공하는 정보를 구조화합니다.
    
    Args:
        state: 현재 그래프 상태
        
    Returns:
        업데이트된 그래프 상태
    """
    url = state.get("target_url", "")
    logger.info("Starting extract_vob_features")
    state["current_step"] = "extracting_vob"
    
    try:
        if is_mock_mode():
            logger.info("Generating Mock VOB data")
            vob_data = generate_mock_vob()
            state["vob_features"] = vob_data
            state["execution_log"].append("Extracted VOB features (Mock)")
        else:
            # TODO: LLM 호출 로직 구현
            # 여기서는 API 키 유무 등을 확인하고 실제 호출
            logger.warning("Real VOB extraction not implemented. Using Mock fallback.")
            vob_data = generate_mock_vob()
            state["vob_features"] = vob_data
            
        return state
        
    except Exception as e:
        error_msg = f"Error in extract_vob_features: {str(e)}"
        logger.error(error_msg, exc_info=True)
        state["errors"].append(error_msg)
        return state


async def extract_voc_features(state: GraphState) -> GraphState:
    """
    VOC(Voice of Customer) 특징을 추출하는 노드.
    
    리뷰 및 Q&A 데이터를 분석하여 고객 피드백을 구조화합니다.
    
    Args:
        state: 현재 그래프 상태
        
    Returns:
        업데이트된 그래프 상태
    """
    logger.info("Starting extract_voc_features")
    state["current_step"] = "extracting_voc"
    
    try:
        if is_mock_mode():
            logger.info("Generating Mock VOC data")
            voc_data = generate_mock_voc()
            state["voc_features"] = voc_data
            state["execution_log"].append("Extracted VOC features (Mock)")
        else:
            # TODO: LLM 호출 로직 구현
            logger.warning("Real VOC extraction not implemented. Using Mock fallback.")
            voc_data = generate_mock_voc()
            state["voc_features"] = voc_data
            
        return state
        
    except Exception as e:
        error_msg = f"Error in extract_voc_features: {str(e)}"
        logger.error(error_msg, exc_info=True)
        state["errors"].append(error_msg)
        return state


async def gap_analysis_node(state: GraphState) -> GraphState:
    """
    VOB와 VOC 간의 갭을 분석하는 노드.
    
    두 데이터를 비교하여 불일치, 기회 요인 등을 도출합니다.
    
    Args:
        state: 현재 그래프 상태
        
    Returns:
        업데이트된 그래프 상태
    """
    logger.info("Starting gap_analysis_node")
    state["current_step"] = "analyzing_gap"
    
    try:
        vob = state.get("vob_features")
        voc = state.get("voc_features")
        
        if not vob or not voc:
            raise ValueError("VOB or VOC features missing for gap analysis")
            
        if is_mock_mode():
            logger.info("Generating Mock Gap Analysis")
            gaps = generate_mock_gap_analysis(vob, voc)
            state["gap_analysis_results"] = gaps
            state["execution_log"].append(" performed gap analysis (Mock)")
        else:
            # TODO: LLM 호출 로직 구현
            logger.warning("Real Gap Analysis not implemented. Using Mock fallback.")
            gaps = generate_mock_gap_analysis(vob, voc)
            state["gap_analysis_results"] = gaps
            
        return state
        
    except Exception as e:
        error_msg = f"Error in gap_analysis_node: {str(e)}"
        logger.error(error_msg, exc_info=True)
        state["errors"].append(error_msg)
        return state


async def save_results_node(state: GraphState) -> GraphState:
    """
    최종 결과를 저장하는 노드.
    
    분석 결과를 JSON 파일로 저장합니다.
    
    Args:
        state: 현재 그래프 상태
        
    Returns:
        업데이트된 그래프 상태
    """
    logger.info("Starting save_results_node")
    state["current_step"] = "saving"
    
    try:
        # 결과 디렉토리 생성
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 파일명 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"analysis_result_{timestamp}.json"
        filepath = output_dir / filename
        
        # 저장할 데이터 구성
        result_data = {
            "metadata": {
                "target_url": state.get("target_url"),
                "timestamp": timestamp,
                "execution_log": state.get("execution_log", []),
                "errors": state.get("errors", []),
            },
            "vob": state["vob_features"].model_dump() if state.get("vob_features") else None,
            "voc": state["voc_features"].model_dump() if state.get("voc_features") else None,
            "gaps": [gap.model_dump() for gap in state.get("gap_analysis_results", [])],
        }
        
        # 파일 쓰기
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Results saved to {filepath}")
        state["execution_log"].append(f"Results saved to {filepath}")
        
        return state
        
    except Exception as e:
        error_msg = f"Error in save_results_node: {str(e)}"
        logger.error(error_msg, exc_info=True)
        state["errors"].append(error_msg)
        return state
