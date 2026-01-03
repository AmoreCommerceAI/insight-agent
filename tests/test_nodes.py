"""
LangGraph 노드 단위 테스트.

각 노드가 GraphState를 올바르게 처리하고 업데이트하는지 검증합니다.
특히 Mock 모드에서의 동작을 중점적으로 테스트합니다.
"""

import pytest
import os
from pathlib import Path
from src.state import GraphState
from src.nodes import (
    load_data_node,
    extract_vob_features,
    extract_voc_features,
    gap_analysis_node,
    save_results_node,
)
from src.schemas import VOBFeatures, VOCFeatures

# Mock 환경 설정
os.environ["USE_MOCK"] = "true"

@pytest.fixture
def initial_state():
    """테스트용 초기 상태"""
    return GraphState(
        target_url="https://example.com/product/123",
        errors=[],
        execution_log=[],
        current_step="start"
    )

@pytest.mark.asyncio
async def test_load_data_node_mock(initial_state):
    """load_data_node가 Mock 데이터를 잘 로드하는지 테스트"""
    new_state = await load_data_node(initial_state)
    
    assert new_state["current_step"] == "loading"
    assert "[Mock]" in new_state["vob_raw_html"]
    assert "[Mock]" in new_state["voc_raw_html"]
    assert len(new_state["errors"]) == 0

@pytest.mark.asyncio
async def test_extract_vob_features_mock(initial_state):
    """extract_vob_features가 Mock VOB 데이터를 잘 생성하는지 테스트"""
    new_state = await extract_vob_features(initial_state)
    
    assert new_state["current_step"] == "extracting_vob"
    assert isinstance(new_state["vob_features"], VOBFeatures)
    assert new_state["vob_features"].product_name is not None
    assert len(new_state["errors"]) == 0

@pytest.mark.asyncio
async def test_extract_voc_features_mock(initial_state):
    """extract_voc_features가 Mock VOC 데이터를 잘 생성하는지 테스트"""
    new_state = await extract_voc_features(initial_state)
    
    assert new_state["current_step"] == "extracting_voc"
    assert isinstance(new_state["voc_features"], VOCFeatures)
    assert new_state["voc_features"].total_review_count > 0
    assert len(new_state["errors"]) == 0

@pytest.mark.asyncio
async def test_gap_analysis_node_mock(initial_state):
    """gap_analysis_node가 Mock 갭 분석 결과를 잘 생성하는지 테스트"""
    # 선행 데이터 주입
    state_with_features = await extract_vob_features(initial_state)
    state_with_features = await extract_voc_features(state_with_features)
    
    new_state = await gap_analysis_node(state_with_features)
    
    assert new_state["current_step"] == "analyzing_gap"
    assert len(new_state["gap_analysis_results"]) > 0
    assert new_state["gap_analysis_results"][0].gap_category is not None
    assert len(new_state["errors"]) == 0

@pytest.mark.asyncio
async def test_save_results_node(initial_state):
    """save_results_node가 파일을 정상적으로 생성하는지 테스트"""
    # 데이터 준비
    state = await extract_vob_features(initial_state)
    state = await extract_voc_features(state)
    state = await gap_analysis_node(state)
    
    new_state = await save_results_node(state)
    
    assert new_state["current_step"] == "saving"
    assert len(new_state["errors"]) == 0
    
    # 파일 생성 확인
    output_dir = Path("outputs")
    files = list(output_dir.glob("*.json"))
    assert len(files) > 0
    
    # 가장 최근 파일 확인
    latest_file = max(files, key=os.path.getctime)
    assert latest_file.stat().st_size > 0
