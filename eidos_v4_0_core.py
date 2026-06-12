import sys
import importlib.util
import json
import os
import networkx as nx
import numpy as np
import tensorflow as tf
from collections import deque
import random
from typing import List, Tuple, Optional, Dict, Any, Set, Union, Deque
from explanation_schema import ExplanationContext, ExplanationResult, SlideSpec, VisualBlock
from sklearn.cluster import KMeans
import asyncio  # <--- [v-Fix] asyncio.gather를 위해 필요
import time
import datetime
import numpy as np
import importlib
import re
import ast
import execution_module
from sklearn.metrics.pairwise import cosine_similarity # <<<--- [v18.7] RAG를 위해 신규 임포트

# [Stage 3.5] Divide & Conquer Reasoner
from eidos_dc_reasoner import DCReasoner
import cv2

# [v-Fix] moviepy, pydub는 EXE 빌드 환경에서 제외 -> 조건부 import
try:
    import moviepy.editor
    from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip, concatenate_videoclips
    _MOVIEPY_AVAILABLE = True
except ImportError:
    VideoFileClip = TextClip = CompositeVideoClip = concatenate_videoclips = None
    _MOVIEPY_AVAILABLE = False
    print("⚠️ [Core] moviepy 없음 — 영상 처리 기능 비활성화")

try:
    from pydub import AudioSegment
    from pydub.silence import detect_silence, split_on_silence
    from pydub.effects import normalize
    _PYDUB_AVAILABLE = True
except ImportError:
    AudioSegment = detect_silence = split_on_silence = normalize = None
    _PYDUB_AVAILABLE = False
    print("⚠️ [Core] pydub 없음 — 오디오 분석 기능 비활성화")
from PIL import Image
import io
from eidos_config import (
    SAVE_FILE, CODE_VECTOR_DB_FILE, CODE_CHUNK_SIZE, CODE_CHUNK_OVERLAP
)

# [v12.0 Semantic] 수정된 모델 빌더 임포트
from eidos_v4_0_nn_models import (
    build_event_encoder, # <-- 시그니처 변경됨 (kb_vocab 필요)
    build_narrative_encoder,
    build_emotion_network,
    build_actor_network,
    build_critic_network,
    LSTM_UNITS,
    VOCAB_SIZE,
    LSTM_UNITS,
    EMOTION_DIM,
    NUM_ACTIONS
)

from llm_module import (
    analyze_event_with_llm_async,
    classify_input_async,
    generate_reasoning_and_response_async,
    generate_natural_response_async,
    generate_proactive_speech_async,       # [v-Character v3] 캐릭터 발화 LLM 연결
    get_llm_response_async,
    EMOTION_MAP,
    EMOTION_MIN,    # [Fix v-Aira] EmotionState.update() clip 기준 통일
    EMOTION_MAX,    # [Fix v-Aira] llm_module과 동일하게 0.0~2.0
    classify_user_feedback_async,
    classify_editor_type_async,
    modify_code_async,
    generate_evaluation_criteria_async,
    evaluate_result_against_criteria_async,
    convert_plan_to_criteria_async,
    # [설명 기능 MVP] 신규 임포트
    classify_explanation_request_async,
    generate_explanation_schema_async
)
# [v11.1] graph_abduction 임포트 확인 (compute_graph_similarity v11.1 사용)
from graph_abduction import compute_graph_similarity
# [v10.0] 모듈 임포트
from eidos_v4_0_causal_engine_v10 import CausalInferenceEngine
from eidos_v10_0_features import AbstractConceptFormer, CounterfactualSimulator
# [v-Fix] MultiModalEncoder — torch 없을 수 있으므로 조건부 import
try:
    from eidos_multimodal_encoder import MultiModalEncoder
    _MULTIMODAL_AVAILABLE = True
except Exception as _e_mm:
    MultiModalEncoder = None
    _MULTIMODAL_AVAILABLE = False
    print(f"⚠️ [Core] MultiModalEncoder 로드 실패 (무시): {_e_mm}")
from eidos_world_model import WorldModel
from eidos_safety_module import SafetyModule
from object_temporal_loop import ObjectTemporalLoopEngine
from eidos_object_wave_engine import ObjectWaveEngine, WaveObject
from eidos_wave_system import EIDOSWaveSystem
from EIDOS_Si_Module import EIDOS_Si_Interface
from EIDOS_Ne_Module import NeIntuitionEngine
from EIDOS_Ni_Module import NiVisionEngine
from EIDOS_F_Module import FiInnerValueEngine, FeEmpathyEngine
from project_structure_helper import ProjectStructureHelper
# ----------------------------------------------------------------------

class CodeVectorDatabase:
    """
    EIDOS 프로젝트 파일들을 인코딩하고 벡터 DB에 저장하여,
    'AI로 기능 추가' 시 LLM 토큰 제한을 우회하기 위한 RAG 모듈.
    self.core.multimodal_encoder를 사용하여 텍스트를 벡터화합니다.
    """
    def __init__(self, core: 'EidosCore'):
        self.core = core
        # { "eidos_files/path/file.py": [{"chunk": "code...", "vector": [0.1, ...]}, ...]}
        self.vector_db: Dict[str, List[Dict]] = {}
        self.db_changed = False # [v-Fix] DB 저장 플래그 추가
        self._load_vector_db()

    def _load_vector_db(self):
        if os.path.exists(CODE_VECTOR_DB_FILE):
            try:
                with open(CODE_VECTOR_DB_FILE, 'r', encoding='utf-8') as f:
                    self.vector_db = json.load(f)
                print(f"✅ [CodeVectorDB] {len(self.vector_db)}개 파일의 벡터를 {CODE_VECTOR_DB_FILE}에서 로드했습니다.")
                self.db_changed = False # [v-Fix] 로드 성공 시 플래그 리셋
            except Exception as e:
                print(f"⚠️ [CodeVectorDB] {CODE_VECTOR_DB_FILE} 로드 실패: {e}. (다시 인덱싱합니다)")
                self.vector_db = {}
                self.db_changed = True # [v-Fix] 로드 실패 시, 새로 만들어야 하므로 dirty
    
    def save_vector_db(self):
        """ [v-Fix] DB가 변경되었을 때만 JSON 파일로 저장 (EidosCore가 주기적으로 호출) """
        
        # [v-Fix] 변경 사항이 없으면 I/O 작업을 건너뜁니다.
        if not self.db_changed:
            # print("ℹ️ [CodeVectorDB] 변경 사항이 없어 저장을 건너뜁니다.")
            return

        try:
            with open(CODE_VECTOR_DB_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.vector_db, f) # 대용량이므로 indent=None
            print(f"💾 [CodeVectorDB] 벡터 DB를 {CODE_VECTOR_DB_FILE}에 저장했습니다.")
            self.db_changed = False # [v-Fix] 저장 후 플래그 리셋
        except Exception as e:
            print(f"❌ [CodeVectorDB] 벡터 DB 저장 실패: {e}")

    async def _vectorize_text(self, text: str) -> List[float]:
        """텍스트(코드 조각)를 EIDOS 인코더로 벡터화"""

        if not hasattr(self.core, "multimodal_encoder") or self.core.multimodal_encoder is None:
            print("⚠️ [CodeVectorDB] MultiModalEncoder 없음")
            return []

        try:
            vector_np = await self.core.multimodal_encoder.fuse_async(text, None, None)

            if vector_np is None:
                return []

            return vector_np.flatten().tolist()

        except Exception as e:
            print(f"❌ [CodeVectorDB] vectorize 실패: {e}")
            return []

    async def index_single_file_async(self, file_path_abs: str) -> bool:
        """ (v18.13) 지정된 단일 파일을 즉시 인덱싱(벡터화)하고 DB에 저장합니다. """
        
        # 1. 절대 경로를 DB의 키인 '상대 경로'로 변환
        try:
            # [Fix] project_root가 상대 경로일 수 있으므로 반드시 절대 경로로 정규화
            abs_project_root = os.path.abspath(self.core.project_root)
            relative_path = os.path.relpath(file_path_abs, abs_project_root)
        except ValueError:
             print(f"⚠️ [RAG Index-Single] {file_path_abs}는 project_root({self.core.project_root}) 외부에 있어 인덱싱할 수 없습니다.")
             return False # RAG 실패
        except Exception as e_path:
             print(f"❌ [RAG Index-Single] 경로 계산 오류: {e_path}")
             return False

        # 2. 코드 파일만 대상으로 함 (기존 로직)
        if not file_path_abs.endswith(('.py', '.js', '.html', '.css', '.java', '.cs', '.cpp', '.c', '.go', '.rs')):
            print(f"ℹ️ [RAG Index-Single] {relative_path}는 코드 파일이 아니므로 인덱싱을 건너뜁니다.")
            return False

        print(f"🧠 [RAG Index-Single] '{relative_path}' 파일 즉시 인덱싱 시작...")
        
        try:
            # 3. 파일 읽기 (비동기)
            def sync_read():
                with open(file_path_abs, 'r', encoding='utf-8') as f:
                    return f.read()
            content = await asyncio.to_thread(sync_read)

            # 4. 코드 조각(Chunk) 생성
            chunks = self._chunk_file(content)
            if not chunks:
                print(f"⚠️ [RAG Index-Single] {relative_path} 파일이 비어있어 인덱싱할 수 없습니다.")
                self.vector_db.pop(relative_path, None) # DB에서 (있다면) 제거
                return False

            # [v-Fix] 5. (병목 1 개선) 벡터화를 순차(sequential)가 아닌 동시(concurrent)로 실행
            print(f"  [RAG Index-Single] '{relative_path}'의 청크 {len(chunks)}개 동시 벡터화 시작...")
            try:
                vectorization_tasks = [self._vectorize_text(chunk) for chunk in chunks]
                vectors = await asyncio.gather(*vectorization_tasks)
            except Exception as e_vec:
                print(f"❌ [RAG Index-Single] 벡터화 작업 중 오류 발생: {e_vec}")
                return False
                
            print(f"  [RAG Index-Single] 벡터화 완료. DB에 저장 중...")
            
            # [v-Fix] 벡터화가 완료된 후 DB에 한 번에 저장
            new_chunks_data = []
            for chunk, vector in zip(chunks, vectors):
                if vector: # 벡터화 성공 시 (빈 리스트가 아닐 때)
                    new_chunks_data.append({"chunk": chunk, "vector": vector})
            
            self.vector_db[relative_path] = new_chunks_data
            
            print(f"✅ [RAG Index-Single] '{relative_path}' 인덱싱 완료 ({len(new_chunks_data)}개 청크).")
            
            # [v-Fix] 6. (병목 2 개선) 매번 저장하지 않고 'db_changed' 플래그만 설정
            self.db_changed = True
            # (EidosCore가 나중에 save_vector_db()를 호출해야 함)
            # await asyncio.to_thread(self.save_vector_db) <-- [v-Fix] 이 줄 제거
            
            return True
            
        except UnicodeDecodeError:
            print(f"⚠️ [RAG Index-Single] {relative_path} 인덱싱 실패 (UnicodeDecodeError).")
            return False
        except Exception as e:
            print(f"❌ [RAG Index-Single] {relative_path} 인덱싱 실패: {e}")
            return False

    def _chunk_file(self, content: str) -> List[str]:
        """ 텍스트를 고정된 크기의 조각(Chunk)으로 자릅니다. """
        chunks = []
        if not content:
            return chunks
        for i in range(0, len(content), CODE_CHUNK_SIZE - CODE_CHUNK_OVERLAP):
            chunk = content[i:i + CODE_CHUNK_SIZE]
            chunks.append(chunk)
        return chunks

    async def index_project_files_async(self, project_root: str):
        """
        [v-Fix] 'project_root' (e.g., eidos_files)를 스캔하여 모든 코드 파일을
        '비동기' 및 '동시(concurrent)'로 인덱싱(벡터화)합니다.
        (EIDOS 시작 시 1회 호출됨)
        """
        print(f"🧠 [CodeVectorDB] '{project_root}'의 백그라운드 인덱싱 시작...")
        files_indexed = 0
        files_failed = 0
        
        # [v-Fix] 이 세션에서 DB 내용이 변경되었는지 추적하는 플래그
        files_changed_in_session = False
        
        # os.walk는 동기 함수이므로 to_thread로 감싸야 하지만,
        # 내부 루프가 await를 포함하므로 여기서는 메인 스레드에서 실행
        for root, dirs, files in os.walk(project_root, topdown=True):
            # 1. 가지치기 (Pruning) - 불필요한 디렉토리 탐색 중지
            dirs[:] = [d for d in dirs if d not in ['__pycache__', 'venv', '.git', 'node_modules', '.vscode']]
            files = [f for f in files if not f.startswith('.')]
            
            for file in files:
                # 2. 코드 파일만 대상으로 함 (json, md, txt 등 제외)
                if not file.endswith(('.py', '.js', '.html', '.css', '.java', '.cs', '.cpp', '.c', '.go', '.rs')):
                    continue
                    
                file_path_abs = os.path.join(root, file)
                # DB의 키로 사용할 '상대 경로'
                relative_path = os.path.relpath(file_path_abs, project_root)

                try:
                    # 3. 파일 읽기 (비동기)
                    def sync_read():
                        with open(file_path_abs, 'r', encoding='utf-8') as f:
                            return f.read()
                    content = await asyncio.to_thread(sync_read)

                    # 4. 코드 조각(Chunk) 생성
                    chunks = self._chunk_file(content)
                    if not chunks:
                        continue
                        
                    # [v-Fix] 5. (병목 1 개선) 벡터화를 순차(sequential)가 아닌 동시(concurrent)로 실행
                    
                    # 5a. 모든 벡터화 작업을 태스크 리스트로 만듭니다.
                    vectorization_tasks = [self._vectorize_text(chunk) for chunk in chunks]
                    
                    # 5b. asyncio.gather로 모든 태스크를 '동시'에 실행합니다.
                    # (L161의 `await asyncio.sleep(0.01)`도 불필요하므로 제거됨)
                    vectors = await asyncio.gather(*vectorization_tasks)
                    
                    # 5c. 벡터화가 완료된 후 DB에 한 번에 저장
                    new_chunks_data = []
                    for chunk, vector in zip(chunks, vectors):
                        if vector: # 벡터화 성공 시 (빈 리스트가 아닐 때)
                            new_chunks_data.append({"chunk": chunk, "vector": vector})
                    
                    self.vector_db[relative_path] = new_chunks_data
                    files_changed_in_session = True # [v-Fix] 메모리 DB가 변경되었음을 표시
                    
                    files_indexed += 1
                except UnicodeDecodeError:
                    files_failed += 1 # 바이너리 파일 등
                except Exception as e:
                    print(f"⚠️ [CodeVectorDB] {relative_path} 인덱싱 실패: {e}")
                    files_failed += 1
        
        # [v-Fix] (병목 2 개선)
        # 이 세션에서 하나라도 파일이 인덱싱되었다면
        if files_changed_in_session:
            print(f"✅ [CodeVectorDB] 인덱싱 완료. (성공: {files_indexed}, 실패: {files_failed}). 인메모리 DB 업데이트됨.")
            
            # [v-Fix] save_vector_db는 클래스 변수(self.db_changed)를 확인해야 합니다.
            # (이 함수 스코프에서는 self.db_changed를 직접 True로 설정)
            try:
                self.db_changed = True
                await asyncio.to_thread(self.save_vector_db) # 저장은 동기 I/O이므로 스레드에서
            except AttributeError:
                 print(f"❌ [CodeVectorDB] save_vector_db 호출 실패. (db_changed 플래그가 구현되지 않았을 수 있습니다. 이전 수정안을 확인하세요.)")
                 # (Fallback) 이전 수정안이 적용되지 않았을 경우, 직접 호출 시도
                 await asyncio.to_thread(self.save_vector_db)
        else:
            print("🧠 [CodeVectorDB] 새로 인덱싱할 파일 없음.")
            
    async def retrieve_relevant_chunks_async(self, query_text: str, file_path_abs: str, top_k: int = 5) -> str:
        """
        [RAG의 핵심] 사용자 쿼리와 가장 관련 있는 코드 조각(Chunk)을
        Vector DB에서 검색하여 문자열로 반환합니다.
        """
        if not self.vector_db or not query_text or not file_path_abs:
            return ""
        
        print(f"🧠 [RAG] '...{query_text[-30:]}' 쿼리로 '{os.path.basename(file_path_abs)}' 파일 검색 중...")
        
        # 1. 절대 경로를 DB의 키인 '상대 경로'로 변환
        try:
            relative_path = os.path.relpath(file_path_abs, self.core.project_root)
        except ValueError:
             # (project_root가 eidos_files가 아닌 다른 곳일 경우)
             relative_path = os.path.basename(file_path_abs) 
        
        # 2. DB에서 해당 파일의 청크 리스트 가져오기
        file_chunks_data = self.vector_db.get(relative_path)
        if not file_chunks_data:
            print(f"⚠️ [RAG] '{relative_path}' 파일이 벡터 DB에 없습니다. (인덱싱 필요?)")
            return "" # RAG 실패 (v18.6 Fallback 로직이 대신 동작할 것임)
            
        try:
            # 3. 쿼리 벡터화
            query_vector = np.array(await self._vectorize_text(query_text)).reshape(1, -1)
            if query_vector.size == 0:
                return ""
                
            # 4. 파일 청크 벡터와 코사인 유사도 계산 (Numpy)
            chunk_vectors = [np.array(chunk_data["vector"]) for chunk_data in file_chunks_data]
            chunk_texts = [chunk_data["chunk"] for chunk_data in file_chunks_data]
            
            if not chunk_vectors:
                 return ""
                 
            similarities = cosine_similarity(query_vector, np.array(chunk_vectors))[0]
            
            # 5. Top-K 선정
            top_indices = similarities.argsort()[-top_k:][::-1]
            
            # 6. LLM에 전달할 문맥(텍스트) 생성
            relevant_chunks_context = []
            for i in top_indices:
                if similarities[i] > 0.5: # (임계값 0.5)
                    relevant_chunks_context.append(
                        f"\n--- [관련 코드 조각 #{i+1} (유사도: {similarities[i]:.2f})] ---\n"
                        f"{chunk_texts[i]}\n"
                    )
            
            print(f"✅ [RAG] '{relative_path}'에서 {len(relevant_chunks_context)}개의 관련 코드 조각을 찾았습니다.")
            return "".join(relevant_chunks_context)
            
        except Exception as e:
            print(f"❌ [RAG] 유사도 검색 중 오류: {e}")
            return "" # RAG 실패

# Hyperparameters are now managed in eidos_config.py
from eidos_config import *

try:
    # [v1.2] 파일 I/O 함수 임포트 추가
    from execution_module import (
        perform_web_search, write_text, read_file, write_file,
        calculate_math, # <<<--- [v13.0 범용성] 수학 모듈 임포트
        write_project_files_async,
        AVAILABLE_TOOLS,
        set_sandbox_root,
        execute_python_file,
        # [v-Fix] 🐛 NameError 해결: 3개의 신규 도구 import
        register_object_wave,
        get_adjacent_object_analysis,
        publish_marketing_content_async,
        write_complex_code_iteratively,
        read_code_skeleton,
        generic_calendar_create,
        generic_calendar_search,
        generic_calendar_delete,
        # [PromptCanvas 연결] 저장된 에이전트 카탈로그/실행 tool
        list_saved_agents,
        run_saved_agent,
    )
    # [!!! 오류 수정 1: 이 줄을 삭제합니다. (self가 정의되지 않음) !!!]
    # set_sandbox_root(self.project_root) 
    EXECUTION_MODULE_LOADED = True
except ImportError:
    # ... (Fallback 정의에 read_file, write_file 추가) ...
    EXECUTION_MODULE_LOADED = False
    async def perform_web_search(query: str): return f"웹 검색 기능 미구현: '{query}'"
    async def write_text(prompt: str): return f"글 작성 기능 미구현: '{prompt}'"
    async def read_file(filepath: str): return f"파일 읽기 기능 미구현: '{filepath}'"
    async def write_file(filepath: str, content: str): return f"파일 쓰기 기능 미구현: '{filepath}'"
    async def calculate_math(expression: str): return "수학 계산 기능 미구현"
    async def write_project_files_async(file_structure_json: str): return "프로젝트 쓰기 기능 미구현"
    AVAILABLE_TOOLS = {}

    async def register_object_wave(concept_name: str, attributes_dict: dict, core_instance: 'EidosCore'):
        return "오류: execution_module 로드 실패"
    async def get_adjacent_object_analysis(concept_name: str, core_instance: 'EidosCore'):
        return "오류: execution_module 로드 실패"
    async def publish_marketing_content_async(product_name: str, main_post: str, first_comment: str, core_instance: 'EidosCore'):
        return "오류: execution_module 로드 실패"
    async def generic_calendar_create(**kwargs): return "오류: execution_module 로드 실패"
    async def generic_calendar_search(**kwargs): return "오류: execution_module 로드 실패"
    async def generic_calendar_delete(**kwargs): return "오류: execution_module 로드 실패"
    async def write_complex_code_iteratively(**kwargs): return "오류: execution_module 로드 실패"
    async def read_code_skeleton(**kwargs): return "오류: execution_module 로드 실패"
    async def list_saved_agents(**kwargs): return "오류: execution_module 로드 실패"
    async def run_saved_agent(**kwargs):  return "오류: execution_module 로드 실패"


    def set_sandbox_root(path: str):
        print(f"⚠️ [Execution Module Fallback] set_sandbox_root({path}) 호출됨 (기능 미구현)")
        pass

class VideoProcessor:
    def __init__(self, core: 'EidosCore'):
        self.core = core
        self.output_dir = os.path.join(core.project_root, "video_outputs")
        os.makedirs(self.output_dir, exist_ok=True)

    async def _extract_frames_async(self, video_path: str, interval_sec: int = 1) -> List[np.ndarray]:
        """ 동영상에서 일정 간격으로 프레임을 추출합니다. """
        print(f"🎥 [VideoProcessor] '{video_path}'에서 프레임 추출 중...")
        frames = []
        try:
            # [v-Fix] cv2.VideoCapture는 동기/블로킹 I/O일 수 있으므로 to_thread로 감쌉니다.
            def sync_extract():
                _frames = []
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    print(f"⚠️ [VideoProcessor] '{video_path}'를 열 수 없습니다.")
                    return []
                    
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                if fps == 0:
                    print(f"⚠️ [VideoProcessor] '{video_path}'의 FPS가 0입니다. 프레임 추출 불가.")
                    cap.release()
                    return []

                frame_interval = int(fps * interval_sec) # 1초 간격으로 프레임 추출
                
                for i in range(0, frame_count, frame_interval):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                    ret, frame = cap.read()
                    if not ret:
                        break
                    # BGR -> RGB (PIL/LLM 입력 형식에 맞춤)
                    _frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                cap.release()
                return _frames
            
            frames = await asyncio.to_thread(sync_extract)
            print(f"✅ [VideoProcessor] {len(frames)}개 프레임 추출 완료.")
        except Exception as e:
            print(f"❌ [VideoProcessor] 프레임 추출 중 오류 발생: {e}")
        return frames

    async def _extract_audio_text_async(self, video_path: str) -> str:
        """ [v-Fix] 동영상에서 오디오를 추출하고 STT(Speech-to-Text)를 '동시성(concurrently)'으로 수행합니다. """
        if not _MOVIEPY_AVAILABLE or not _PYDUB_AVAILABLE:
            return "영상 처리 라이브러리(moviepy, pydub)가 설치되지 않아 해당 기능을 사용할 수 없습니다."
        print(f"🎧 [VideoProcessor] '{video_path}'에서 오디오 추출 및 STT 시도 중...")
        audio_path = os.path.join(self.output_dir, f"{os.path.basename(video_path)}_audio.mp3")
        
        try:
            # 1. VideoFileClip으로 오디오 추출 (MoviePy)
            await asyncio.to_thread(lambda: VideoFileClip(video_path).audio.write_audiofile(audio_path, logger=None))
            print(f"✅ [VideoProcessor] 오디오 추출 완료: {audio_path}")

            # 2. pydub으로 오디오 로드 및 분할 (너무 길면 STT 실패하므로)
            audio = AudioSegment.from_file(audio_path)
            # 무음 구간 기준으로 분할 (STT 효율 증대)
            chunks = split_on_silence(audio, 
                                      min_silence_len=500, # 0.5초 이상 무음
                                      silence_thresh=-40) # -40dB 이하 무음
            
            if not self.core.multimodal_encoder:
                 print("⚠️ [VideoProcessor] MultiModalEncoder가 없어 STT를 수행할 수 없습니다.")
                 return ""
            
            # [v-Fix] 3. (병목 개선) 오디오 청크 파일 저장을 '동시'에 수행
            chunk_paths = []
            export_tasks = []
            
            print(f"  [VideoProcessor] {len(chunks)}개 오디오 청크 파일 동시 저장 중...")
            for i, chunk in enumerate(chunks):
                if not chunk.duration_seconds > 0:
                    continue
                chunk_audio_path = os.path.join(self.output_dir, f"temp_chunk_{i}.wav")
                chunk_paths.append(chunk_audio_path)
                export_tasks.append(
                    asyncio.to_thread(lambda c=chunk, p=chunk_audio_path: c.export(p, format="wav"))
                )
            
            if not export_tasks:
                 print("⚠️ [VideoProcessor] 유효한 오디오 청크가 없습니다.")
                 return ""
                 
            await asyncio.gather(*export_tasks)
            print(f"  [VideoProcessor] {len(chunk_paths)}개 청크 파일 저장 완료.")

            # [v-Fix] 4. (병목 개선) STT 작업을 '동시'에 수행
            stt_tasks = []
            for path in chunk_paths:
                stt_tasks.append(
                    self.core.multimodal_encoder.transcribe_audio_file_async(path)
                )

            print(f"  [VideoProcessor] {len(stt_tasks)}개 청크 STT 동시 처리 시작...")
            results = await asyncio.gather(*stt_tasks)
            print(f"  [VideoProcessor] STT 처리 완료.")

            # 5. 결과 취합
            full_text = []
            for i, text_from_chunk in enumerate(results):
                if text_from_chunk and not text_from_chunk.startswith("[STT"):
                    full_text.append(text_from_chunk)
                    print(f"  [VideoProcessor] STT (Chunk {i+1}): {text_from_chunk}")
                else:
                    # STT 실패 또는 빈 결과
                    print(f"  [VideoProcessor] STT (Chunk {i+1}) 실패 또는 음성 없음.")

            if full_text:
                print(f"✅ [VideoProcessor] 오디오 STT 요청 준비 완료. {len(chunks)}개 청크.")
            else:
                print("⚠️ [VideoProcessor] 오디오 청크 생성 실패 또는 STT 불가.")
            return "\n".join(full_text) + "\n\n(참고: 실제 STT는 EIDOS가 별도 도구를 만들어서 수행해야 합니다.)"

        except FileNotFoundError:
            print(f"❌ [VideoProcessor] MoviePy 또는 pydub이 필요한 파일({video_path} 또는 ffmpeg/ffprobe)을 찾을 수 없습니다.")
            return "(오디오 추출 및 STT 실패: 필요한 도구 없음)"
        except Exception as e:
            print(f"❌ [VideoProcessor] 오디오 추출 및 STT 중 오류 발생: {e}")
            return f"(오디오 추출 및 STT 실패: {e})"
        finally:
            # [v-Fix] 임시 파일 정리 로직 (경로 리스트 사용)
            if os.path.exists(audio_path):
                try: os.remove(audio_path)
                except Exception: pass
            
            # chunk_paths가 정의되었는지 확인
            if 'chunk_paths' in locals():
                for path in chunk_paths:
                    if os.path.exists(path):
                        try: os.remove(path)
                        except Exception: pass
            else:
                # Fallback: 기존의 이름 기반 정리 (예외 발생 시)
                for f in os.listdir(self.output_dir):
                    if f.startswith("temp_chunk_") and (f.endswith(".wav") or f.endswith(".mp3")):
                        try: os.remove(os.path.join(self.output_dir, f))
                        except Exception: pass

    # [!!! v18.8 신규 도구 뼈대: 동영상 편집 !!!]
    async def edit_video_async(self, video_path: str, start_time: float, end_time: float, output_path: str) -> str:
        if not _MOVIEPY_AVAILABLE:
            return "영상 처리 라이브러리(moviepy)가 설치되지 않아 해당 기능을 사용할 수 없습니다."
        """
        [EIDOS Tool] 동영상에서 특정 구간을 잘라내어 새로운 동영상을 만듭니다.
        추후 EIDOS가 이 함수를 확장하여 더 복잡한 편집(병합, 효과 추가 등)을 수행할 수 있습니다.
        """
        print(f"🎥 [VideoProcessor] '{video_path}'의 {start_time:.1f}s-{end_time:.1f}s 구간 편집 중...")
        try:
            # [!!! v18.8 구현 !!!]
            # LLM이 생성한 output_path의 디렉토리가 없을 수 있으므로 생성
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # [v-Fix] MoviePy의 동기 함수들을 to_thread로 감쌉니다.
            def sync_edit():
                with VideoFileClip(video_path) as clip:
                    with clip.subclip(start_time, end_time) as subclip:
                        subclip.write_videofile(output_path, logger=None, threads=4, codec='libx264')
            
            await asyncio.to_thread(sync_edit)
            
            print(f"✅ [VideoProcessor] 동영상 편집 완료: {output_path}")
            return f"✅ 동영상 편집 완료: {output_path}"
        except Exception as e:
            print(f"❌ [VideoProcessor] 동영상 편집 중 오류 발생: {e}")
            return f"❌ 동영상 편집 실패: {e}"

    # [!!! v18.8 신규 도구 뼈대: 자막 추가 !!!]
    async def add_subtitles_async(self, video_path: str, subtitles_json: str, output_path: str) -> str:
        if not _MOVIEPY_AVAILABLE:
            return "영상 처리 라이브러리(moviepy)가 설치되지 않아 해당 기능을 사용할 수 없습니다."
        """
        [EIDOS Tool] 동영상에 자막을 추가합니다.
        subtitles_json: "[{\"text\": \"안녕\", \"start\": 0.5, \"end\": 2.0}, ...]"
        """
        print(f"💬 [VideoProcessor] '{video_path}'에 자막 추가 중...")
        try:
            # [!!! v18.8 구현 !!!]
            # 1. JSON 문자열 파싱
            try:
                subtitles: List[Dict] = json.loads(subtitles_json)
                if not isinstance(subtitles, list):
                    raise ValueError("자막이 리스트 형식이 아닙니다.")
            except Exception as e_json:
                return f"❌ 자막 추가 실패: subtitles_json 파싱 오류. (오류: {e_json})"
                
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # [v-Fix] MoviePy의 동기 함수들을 to_thread로 감쌉니다.
            def sync_add_subs():
                with VideoFileClip(video_path) as video_clip:
                    text_clips = []
                    for sub_data in subtitles:
                        text = sub_data.get("text")
                        start = sub_data.get("start")
                        end = sub_data.get("end")
                        
                        if not all([text, isinstance(start, (int, float)), isinstance(end, (int, float))]):
                            print(f"⚠️ [Subtitles] 잘못된 자막 데이터: {sub_data}, 건너뜁니다.")
                            continue
                        
                        # (moviepy의 TextClip은 시스템 폰트 경로를 요구할 수 있습니다. Arial-Bold는 예시입니다.)
                        txt_clip = TextClip(text, fontsize=24, color='white', bg_color='rgba(0, 0, 0, 0.5)',
                                            font='Malgun-Gothic-Bold', method='caption', size=(video_clip.w * 0.8, None))
                        txt_clip = txt_clip.set_position(('center', 'bottom')).set_duration(end - start).set_start(start)
                        text_clips.append(txt_clip)

                    if not text_clips:
                        # [v-Fix] 함수 내에서 return 대신 예외 발생
                        raise ValueError("유효한 자막 데이터가 없습니다.")

                    with CompositeVideoClip([video_clip] + text_clips) as final_clip:
                        final_clip.write_videofile(output_path, logger=None, threads=4, codec='libx264')
                    
                    # [v-Fix] 수동으로 clip.close() 호출
                    for tc in text_clips:
                        tc.close()

            # [v-Fix] 동기 함수 실행
            await asyncio.to_thread(sync_add_subs)
            
            print(f"✅ [VideoProcessor] 자막 추가 완료: {output_path}")
            return f"✅ 자막 추가 완료: {output_path}"
        
        except ValueError as e_val:
            # [v-Fix] 동기 함수에서 발생한 예외 처리
            print(f"❌ [VideoProcessor] 자막 추가 중 오류 발생: {e_val}")
            return f"❌ 자막 추가 실패: {e_val}"
        except Exception as e:
            print(f"❌ [VideoProcessor] 자막 추가 중 오류 발생: {e}")
            return f"❌ 자막 추가 실패: {e}"

    async def analyze_video_structure_async(self, video_path: str) -> str:
        if not _MOVIEPY_AVAILABLE or not _PYDUB_AVAILABLE:
            return "영상 처리 라이브러리(moviepy, pydub)가 설치되지 않아 해당 기능을 사용할 수 없습니다."
        """
        [EIDOS Tool v18.9] pydub의 'detect_silence'를 사용하여 '무음 구간'을 감지하고,
        이를 '반전'시켜 '유의미한 장면(비-무음 구간)'의
        '정확한' 타임스탬프 리스트(JSON)를 반환합니다.
        """
        print(f"🔬 [VideoProcessor v18.9] '{video_path}' 구조 분석(detect_silence) 중...")
        audio_path = os.path.join(self.output_dir, f"{os.path.basename(video_path)}_temp_audio.mp3")
        
        try:
            # 1. 오디오 추출 (동일)
            # (MoviePy는 동기 함수이므로 to_thread로 감쌉니다)
            def _extract_audio():
                with VideoFileClip(video_path) as clip:
                    if clip.audio:
                        clip.audio.write_audiofile(audio_path, logger=None)
                        return True
                return False

            if not await asyncio.to_thread(_extract_audio):
                print(f"⚠️ [VideoProcessor] '{video_path}'에 오디오 트랙이 없습니다.")
                return "❌ 구조 분석 실패: 동영상에 오디오 트랙이 없습니다."

            # [v-Fix] pydub의 동기 함수들을 to_thread로 감쌉니다.
            def sync_analyze():
                audio = AudioSegment.from_file(audio_path)
                total_duration_ms = len(audio) # pydub은 len()으로 ms를 반환
                
                # 2. [!!! 로직 변경 !!!] '분할' 대신 '무음 구간 감지'
                # min_silence_len=1000 (1초), silence_thresh=-40dB (이 값은 조절 가능)
                # 반환값: [ [start_ms, end_ms], [start_ms, end_ms], ... ]
                print("  [VideoProcessor] 무음 구간 감지(detect_silence) 중...")
                silent_segments_ms = detect_silence(
                    audio, 
                    min_silence_len=1000, # 최소 1초의 무음
                    silence_thresh=-40    # -40dB 이하를 무음으로 간주
                )
                
                if not silent_segments_ms:
                    # 무음 구간이 없으면, 영상 전체를 하나의 클립으로 반환
                    print(f"✅ [VideoProcessor] 무음 구간 없음. 전체를 1개 장면으로 반환.")
                    return json.dumps([{
                        "clip_index": 0, 
                        "start": 0.0, 
                        "end": round(total_duration_ms / 1000.0, 2)
                    }])

                # 3. [!!! 로직 변경 !!!] '무음 구간'을 '반전'시켜 '비-무음 구간' 추출
                scene_list = []
                current_time_ms = 0
                clip_index = 0
                MIN_CLIP_DURATION_MS = 500 # 0.5초 미만의 짧은 소리(클립)는 무시

                for silent_start_ms, silent_end_ms in silent_segments_ms:
                    
                    # [무음 구간 전] | [비-무음 구간 (Scene)] | [무음 구간]
                    # (current_time_ms)      (silent_start_ms)  (silent_end_ms)
                    
                    non_silent_start_ms = current_time_ms
                    non_silent_end_ms = silent_start_ms
                    
                    # 유효한(0.5초 이상) 비-무음 구간이 있으면 리스트에 추가
                    if (non_silent_end_ms - non_silent_start_ms) > MIN_CLIP_DURATION_MS:
                        scene_list.append({
                            "clip_index": clip_index,
                            "start": round(non_silent_start_ms / 1000.0, 2), # 초 단위로 변환
                            "end": round(non_silent_end_ms / 1000.0, 2)
                        })
                        clip_index += 1
                    
                    # 다음 비-무음 구간의 시작점은 현재 무음 구간의 '끝'
                    current_time_ms = silent_end_ms

                # 4. (중요) 마지막 무음 구간 이후 ~ 영상 끝까지의 클립 추가
                if (total_duration_ms - current_time_ms) > MIN_CLIP_DURATION_MS:
                    scene_list.append({
                        "clip_index": clip_index,
                        "start": round(current_time_ms / 1000.0, 2),
                        "end": round(total_duration_ms / 1000.0, 2)
                    })

                print(f"✅ [VideoProcessor] 구조 분석 완료. {len(scene_list)}개 유의미한 장면 감지.")
                return json.dumps(scene_list) # JSON 문자열로 반환

            # [v-Fix] 동기 함수 실행
            return await asyncio.to_thread(sync_analyze)

        except Exception as e:
            print(f"❌ [VideoProcessor] 구조 분석 중 오류: {e}")
            import traceback
            traceback.print_exc()
            return f"❌ 구조 분석 실패: {e}"
        finally:
            # 임시 오디오 파일 삭제
            if os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                except Exception as e_del:
                    print(f"⚠️ [VideoProcessor] 임시 오디오 파일 삭제 실패: {e_del}")

    async def merge_video_clips_async(self, clip_paths_json: str, output_path: str) -> str:
        if not _MOVIEPY_AVAILABLE:
            return "영상 처리 라이브러리(moviepy)가 설치되지 않아 해당 기능을 사용할 수 없습니다."
        """
        [EIDOS Tool] JSON 리스트로 제공된 여러 동영상 클립을 순서대로 병합합니다.
        clip_paths_json: "[\"path/to/clip1.mp4\", \"path/to/clip2.mp4\"]"
        """
        print(f"🧬 [VideoProcessor] 동영상 클립 병합 중... -> {output_path}")
        try:
            # 1. JSON 파싱
            try:
                clip_paths: List[str] = json.loads(clip_paths_json)
                # [!!! 수정: 1개만 있어도 병합(사실상 복사)이 가능하도록 < 2 -> < 1 로 변경 !!!]
                if not isinstance(clip_paths, list) or len(clip_paths) < 1:
                    raise ValueError("병합할 클립이 없거나 리스트 형식이 아닙니다.")
            except Exception as e_json:
                return f"❌ 병합 실패: clip_paths_json 파싱 오류. (오류: {e_json})"
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # [v-Fix] MoviePy의 동기 함수들을 to_thread로 감쌉니다.
            def sync_merge():
                # 2. MoviePy 클립 로드
                clips = [VideoFileClip(path) for path in clip_paths]
                
                try:
                    # 3. 병합 (concatenate_videoclips는 동기 함수)
                    final_clip = concatenate_videoclips(clips, method="compose")
                    
                    # 4. 파일 쓰기 (동기)
                    final_clip.write_videofile(output_path, logger=None, threads=4, codec='libx264')
                finally:
                    # 5. [v-Fix] 리소스 수동 해제
                    for clip in clips:
                        clip.close()
                    if 'final_clip' in locals():
                        final_clip.close()

            await asyncio.to_thread(sync_merge)
            
            print(f"✅ [VideoProcessor] 클립 병합 완료: {output_path}")
            return f"✅ 클립 병합 완료: {output_path}"
        except Exception as e:
            print(f"❌ [VideoProcessor] 클립 병합 중 오류: {e}")
            return f"❌ 병합 실패: {e}"

# ----------------------------------------------------------------------
# 감정/지식/목표 관련 클래스 정의 (v11.1과 동일)
# ----------------------------------------------------------------------
class EmotionState:
    # ... (v11.1과 동일) ...
    def __init__(self): self.activations = np.zeros(EMOTION_DIM); print("[Emotion] EmotionState (v9.1)")
    def get_vector(self) -> np.ndarray: return self.activations.reshape(1, -1)
    def update(self, new_activations: np.ndarray): self.activations = np.clip(new_activations, EMOTION_MIN, EMOTION_MAX)
    def to_dict(self) -> List[float]: return self.activations.tolist()
    def from_dict(self, data: List[float]): self.activations = np.array(data)

class EmotionDynamics:
    """
    심리학 기반 감정 동역학 엔진 (v12.0)

    이론적 토대:
    ─────────────────────────────────────────────────────
    1. Appraisal Theory (Lazarus, 1991)
       감정은 사건의 '평가'에서 발생한다.
       - novelty    : 얼마나 새롭고 예상 밖인가
       - valence    : 목표에 유리(+) vs 불리(-)
       - coping_pot : 내가 대처할 수 있는가 (높으면 분노, 낮으면 공포)
       - relevance  : 나의 목표/가치와 얼마나 관련 있는가

    2. Circumplex Model (Russell, 1980)
       감정을 valence(부정↔긍정) × arousal(낮음↔높음) 2축 공간에 배치.
       이 좌표로 감정 군집을 계산하고 인접 감정 간 전이를 모델링.

    3. Mood Congruence Effect (Bower, 1981)
       현재 감정 상태가 동일한 valence를 가진 기억/정보 처리를 촉진.
       → 기쁠 때는 긍정 기억이 더 잘 떠오름.
       EmotionContextMemory의 recall_weight에 반영.

    4. Emotional Inertia (Kuppens et al., 2010)
       감정의 자기상관 — 강한 감정은 시간이 지나도 지속된다.
       coping_potential이 낮을 때 inertia 계수 상승.

    감정 인덱스 (EMOTION_DIM=12):
       0=기쁨  1=슬픔  2=분노  3=공포  4=놀람  5=혐오
       6=신뢰  7=기대  8=수치심  9=자부심  10=호기심  11=지루함
    """

    # ── Circumplex 좌표 (valence, arousal) ─────────────────────────────
    # valence: -1(부정) ~ +1(긍정), arousal: 0(낮음) ~ 1(높음)
    CIRCUMPLEX: Dict[int, Tuple[float, float]] = {
        0:  ( 0.9,  0.7),   # 기쁨     — 고각성 긍정
        1:  (-0.8,  0.2),   # 슬픔     — 저각성 부정
        2:  (-0.6,  0.9),   # 분노     — 고각성 부정
        3:  (-0.7,  0.8),   # 공포     — 고각성 부정
        4:  ( 0.1,  0.9),   # 놀람     — 고각성 중립
        5:  (-0.9,  0.5),   # 혐오     — 중각성 부정
        6:  ( 0.7,  0.3),   # 신뢰     — 저각성 긍정
        7:  ( 0.6,  0.7),   # 기대     — 중각성 긍정
        8:  (-0.5,  0.4),   # 수치심   — 중각성 부정
        9:  ( 0.8,  0.5),   # 자부심   — 중각성 긍정
        10: ( 0.4,  0.8),   # 호기심   — 고각성 긍정
        11: (-0.2,  0.1),   # 지루함   — 저각성 부정
    }

    # ── 기저 감쇠율 (심리학적 반감기 기반) ─────────────────────────────
    # 공포·슬픔은 진화적으로 지속성이 높음 (생존 가치)
    # 놀람·지루함은 빠르게 소멸
    BASE_DECAY: Dict[str, float] = {
        '기쁨': 0.90, '슬픔': 0.96, '분노': 0.85, '공포': 0.98,
        '놀람': 0.75, '혐오': 0.95, '신뢰': 0.97, '기대': 0.93,
        '수치심': 0.94, '자부심': 0.95, '호기심': 0.91, '지루함': 0.88,
    }

    # ── 감정 간 상호작용 매트릭스 ──────────────────────────────────────
    # (source_idx, target_idx) → strength
    # 양수: source 활성이 target을 증폭 / 음수: 억제
    # 출처: Plutchik의 정서 바퀴 + 임상 심리학 문헌
    INTERACTION: Dict[Tuple[int, int], float] = {
        # 대립쌍 (circumplex 반대편) — 강한 상호 억제
        (0, 1): -0.85,  # 기쁨 → 슬픔 억제
        (1, 0): -0.70,  # 슬픔 → 기쁨 억제
        (2, 6): -0.75,  # 분노 → 신뢰 억제 (분노하면 불신)
        (6, 2): -0.60,  # 신뢰 → 분노 억제
        (5, 6): -0.80,  # 혐오 → 신뢰 억제
        (6, 5): -0.65,  # 신뢰 → 혐오 억제

        # 공포-분노 연동 (같은 위협 자극, coping에 따라 분화)
        (3, 2):  0.30,  # 공포 → 분노 소폭 촉진 (defense response)
        (2, 3): -0.40,  # 분노 → 공포 억제 (통제감 회복)

        # 긍정 감정 클러스터 (기쁨-신뢰-기대-자부심 상호 강화)
        (0, 6):  0.40,  # 기쁨 → 신뢰 증폭
        (6, 0):  0.35,  # 신뢰 → 기쁨 증폭
        (7, 0):  0.45,  # 기대 → 기쁨 증폭 (기대 충족)
        (0, 9):  0.30,  # 기쁨 → 자부심 촉진
        (9, 0):  0.25,  # 자부심 → 기쁨 촉진

        # 호기심-지루함 대립
        (10, 11): -0.70, # 호기심 → 지루함 억제
        (11, 10): -0.50, # 지루함 → 호기심 억제

        # 수치심-자부심 대립
        (8, 9):  -0.80,  # 수치심 → 자부심 억제
        (9, 8):  -0.70,  # 자부심 → 수치심 억제

        # 놀람의 valence 편향 전이 (긍정 맥락 → 기쁨, 부정 맥락 → 공포)
        (4, 0):   0.20,  # 놀람 → 기쁨 (긍정 평가 시)
        (4, 3):   0.20,  # 놀람 → 공포 (부정 평가 시)

        # 슬픔-수치심 공유 (실패/상실의 공통 감정)
        (1, 8):   0.35,  # 슬픔 → 수치심 촉진
        (8, 1):   0.25,  # 수치심 → 슬픔 촉진
    }

    def __init__(self):
        # 감쇠 벡터 구성
        self.base_decay_vector = np.ones(EMOTION_DIM)
        for i, name in EMOTION_MAP.items():
            if name in self.BASE_DECAY:
                self.base_decay_vector[i] = self.BASE_DECAY[name]

        # Appraisal 컨텍스트 이력 (최근 5개 유지, inertia 계산용)
        self._appraisal_history: list = []
        self.fear_index = 3
        print("[Emotion] EmotionDynamics (v12.0 — Appraisal+Circumplex)")

    # ── Appraisal Theory 평가 ───────────────────────────────────────────

    def appraise(
        self,
        novelty: float = 0.5,       # 새로움 (0~1)
        valence: float = 0.0,       # 유리/불리 (-1~+1)
        coping_potential: float = 0.5,  # 대처 능력 (0~1)
        relevance: float = 0.5,     # 목표 관련성 (0~1)
    ) -> np.ndarray:
        """
        Appraisal Theory 기반으로 감정 델타 벡터 생성.

        Args:
            novelty: 높을수록 놀람/호기심 촉진
            valence: 양수=기쁨/자부심, 음수=슬픔/혐오/분노
            coping_potential: 높으면 분노(통제 가능), 낮으면 공포(무력)
            relevance: 높을수록 전반적 감정 강도 증폭

        Returns:
            EMOTION_DIM 크기의 감정 델타 벡터
        """
        delta = np.zeros(EMOTION_DIM)
        amp = relevance  # 관련성 높을수록 전반적 강도 증폭

        # 긍정 평가 경로
        if valence > 0:
            delta[0] += valence * amp * 0.6       # 기쁨
            delta[9] += valence * amp * 0.3       # 자부심
            delta[6] += valence * amp * 0.2       # 신뢰
            delta[7] += valence * amp * 0.15      # 기대

        # 부정 평가 경로 — coping에 따라 분노 vs 공포 분화
        elif valence < 0:
            neg = abs(valence) * amp
            delta[1] += neg * 0.4                 # 슬픔 (기본)
            delta[5] += neg * 0.2                 # 혐오
            # coping 높으면 분노, 낮으면 공포
            delta[2] += neg * coping_potential * 0.5       # 분노
            delta[3] += neg * (1 - coping_potential) * 0.5 # 공포
            delta[8] += neg * (1 - coping_potential) * 0.2 # 수치심

        # 새로움 경로 — novelty는 놀람/호기심 촉진
        delta[4] += novelty * amp * 0.5           # 놀람
        delta[10] += novelty * amp * 0.4          # 호기심
        delta[11] -= novelty * amp * 0.3          # 지루함 감소

        # Appraisal 이력 저장 (inertia 계산용)
        self._appraisal_history.append({
            "novelty": novelty, "valence": valence,
            "coping": coping_potential, "relevance": relevance,
        })
        if len(self._appraisal_history) > 5:
            self._appraisal_history.pop(0)

        return delta

    # ── Emotional Inertia 계수 ──────────────────────────────────────────

    def _compute_inertia(self, idx: int, current_val: float) -> float:
        """
        Kuppens et al. (2010): 감정 강도가 높고 coping이 낮을수록
        감정 변화에 대한 저항(inertia)이 증가.
        """
        base_inertia = 0.3
        # 강한 감정일수록 inertia 증가
        intensity_factor = current_val * 0.3
        # 최근 appraisal의 평균 coping이 낮으면 inertia 증가 (공포/무력감)
        avg_coping = 0.5
        if self._appraisal_history:
            avg_coping = sum(a["coping"] for a in self._appraisal_history) / len(self._appraisal_history)
        coping_factor = (1 - avg_coping) * 0.2
        # 공포·슬픔은 진화적으로 inertia가 더 높음
        trauma_boost = 0.15 if idx in (1, 3, 8) else 0.0
        return min(0.85, base_inertia + intensity_factor + coping_factor + trauma_boost)

    # ── Circumplex 기반 인접 감정 전이 ─────────────────────────────────

    def _circumplex_spread(
        self, activations: np.ndarray, threshold: float = 0.4
    ) -> np.ndarray:
        """
        Russell Circumplex: 활성화된 감정이 인접 좌표의 감정으로
        소량 전이. 감정의 '번짐' 효과 모델링.
        """
        delta = np.zeros(EMOTION_DIM)
        for src_idx in range(EMOTION_DIM):
            if activations[src_idx] < threshold:
                continue
            sv, sa = self.CIRCUMPLEX[src_idx]
            for tgt_idx in range(EMOTION_DIM):
                if tgt_idx == src_idx:
                    continue
                tv, ta = self.CIRCUMPLEX[tgt_idx]
                # 좌표 거리 — 가까울수록 전이 강함
                dist = ((sv - tv) ** 2 + (sa - ta) ** 2) ** 0.5
                if dist < 0.5:  # 인접 감정 (circumplex 상 가까운 것만)
                    spread = activations[src_idx] * (0.5 - dist) * 0.08
                    delta[tgt_idx] += spread
        return delta

    # ── 메인 dynamics 적용 ─────────────────────────────────────────────

    def apply_dynamics(
        self,
        current_activations: np.ndarray,
        emotion_delta: np.ndarray,
        context: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """
        1단계: Appraisal context 기반 감쇠율 조정
        2단계: 감쇠 적용
        3단계: 델타 가산
        4단계: 감정 간 상호작용 매트릭스 적용
        5단계: Circumplex 인접 전이
        6단계: Emotional Inertia로 급격한 변화 완충
        """
        decay = self.base_decay_vector.copy()

        # 트라우마 맥락: 공포 감쇠 최소화
        if context and context.get("is_traumatic"):
            decay[self.fear_index] = 0.999
            print("![EmotionDynamics] Trauma Context — fear inertia 최대화")

        # Appraisal valence: 부정 평가 시 슬픔 감쇠 늦춤
        if context and context.get("valence", 0) < -0.5:
            decay[1] = min(decay[1] + 0.03, 0.99)  # 슬픔 더 오래 유지

        # 1단계: 감쇠
        decayed = current_activations * decay

        # 2단계: 델타 가산
        new_act = decayed + emotion_delta

        # 3단계: 감정 간 상호작용
        interacted = new_act.copy()
        interaction_threshold = 0.45
        for (idx_a, idx_b), strength in self.INTERACTION.items():
            if new_act[idx_a] > interaction_threshold:
                if strength < 0:
                    interacted[idx_b] *= (1 + strength)
                else:
                    interacted[idx_b] = interacted[idx_b] + new_act[idx_a] * strength * 0.3

        # 4단계: Circumplex 인접 전이
        spread = self._circumplex_spread(interacted)
        interacted = interacted + spread

        # 5단계: Emotional Inertia — 급격한 변화를 완충
        final = np.zeros(EMOTION_DIM)
        for i in range(EMOTION_DIM):
            change = interacted[i] - current_activations[i]
            inertia = self._compute_inertia(i, current_activations[i])
            # inertia가 높을수록 변화 속도 감소
            final[i] = current_activations[i] + change * (1 - inertia * 0.5)

        return np.maximum(final, 0.0)

# eidos_v4_0_core.py (또는 최신 코어 파일) 내 EmotionMemory 클래스 수정
class EmotionMemory:
    def __init__(self, max_sources_per_emotion=5):
        self.sources: Dict[int, List[Dict[str, Any]]] = {i: [] for i in range(EMOTION_DIM)}
        self.max_sources = max_sources_per_emotion
        self.current_time = 0
        print("[Emotion] EmotionMemory (v9.1 / Patched)") # 버전명에 패치 표시

    def tick(self): self.current_time += 1

    def record_cause(self, emotion_index: int, cause: str, intensity: float):
        if abs(intensity) < 0.01: return

        # [오류 수정] .get()을 사용하여 안전하게 리스트 가져오기
        source_list = self.sources.get(emotion_index)

        # [오류 수정] 만약 해당 감정 인덱스에 대한 리스트가 없으면 새로 생성
        if source_list is None:
            self.sources[emotion_index] = []
            source_list = self.sources[emotion_index] # 새로 생성된 리스트를 source_list에 할당

        found = False
        # 이제 source_list는 항상 할당되어 있음
        for source in source_list:
            if source['cause'] == cause:
                # 기존 원인이면 강도 업데이트 및 타임스탬프 갱신
                source['intensity'] = max(source['intensity'], intensity)
                # [오타 수정] self.current.time -> self.current_time
                source['timestamp'] = self.current_time
                found = True
                break # 이미 찾았으므로 루프 종료

        # 기존 원인이 아니면 새로 추가
        if not found:
            # [오타 수정] self.current.time -> self.current_time
            source_list.append({ 'cause': cause, 'intensity': intensity, 'timestamp': self.current_time })

            # 리스트 크기 제한 유지
            if len(source_list) > self.max_sources:
                source_list.sort(key=lambda x: x['timestamp'], reverse=True)
                # self.sources[emotion_index] = source_list[:self.max_sources] # 잘라낸 리스트로 딕셔너리 업데이트
                # 위 라인은 아래와 같이 직접 슬라이싱 할당으로 변경 가능
                self.sources[emotion_index][:] = source_list[:self.max_sources]


    def record_causes_from_delta(self, delta_vector: np.ndarray, cause_prefix: str):
        # ... (기존과 동일) ...
        for i, intensity in enumerate(delta_vector):
            if abs(intensity) > 0.1: self.record_cause(i, cause_prefix, intensity)

    def get_explanation(self, emotion_index: int) -> str:
        # ... (기존과 동일) ...
        source_list = self.sources.get(emotion_index, [])
        if not source_list: return "특별한 원인 없음"
        main_source = max(source_list, key=lambda x: abs(x['intensity']))
        return f"{main_source['cause']} (강도 {main_source['intensity']:.2f})"

    def get_primary_cause_string(self, current_activations: np.ndarray) -> str:
        # ... (기존과 동일) ...
        try:
            primary_emotion_index = int(np.argmax(current_activations))
            primary_emotion_name = EMOTION_MAP.get(primary_emotion_index, "?")
            primary_cause = self.get_explanation(primary_emotion_index)
            return f"주요 감정 '{primary_emotion_name}' 원인: '{primary_cause}'."
        except Exception: return "주요 감정 원인 분석 불가."

# --- EidosCore 클래스 등 나머지 코드는 그대로 유지 ---

class EmotionMomentum:
    # ... (v11.1과 동일) ...
    def __init__(self, emotion_dim=EMOTION_DIM, momentum_factor=0.3, base_resistance=0.4):
        self.emotion_dim = emotion_dim; self.momentum_vector = np.zeros(emotion_dim); self.momentum_factor = momentum_factor; self.base_resistance = base_resistance
        self.fear_index = 3; self.resistance_map = { self.fear_index: {'increase': 0.1, 'decrease': 0.5} }; print("[Emotion] EmotionMomentum (v9.2)")
    def _compute_resistance(self, emotion_index: int, change_magnitude: float) -> float:
        config = self.resistance_map.get(emotion_index)
        if config: res = config['increase'] if change_magnitude > 0 else config['decrease']
        else: res = self.base_resistance + (abs(change_magnitude) * 0.2)
        return np.clip(res, 0.0, 0.9)
    def apply(self, current_activations: np.ndarray, target_activations: np.ndarray) -> np.ndarray:
        target_change = target_activations - current_activations; actual_change = np.zeros(self.emotion_dim)
        for i in range(self.emotion_dim):
            change = target_change[i];
            if abs(change) < 0.01: continue
            resistance = self._compute_resistance(i, change); resisted_change = change * (1 - resistance)
            momentum_effect = self.momentum_vector[i] * self.momentum_factor; actual_change[i] = resisted_change + momentum_effect
        self.momentum_vector = (self.momentum_vector * 0.5) + (actual_change * 0.5); final_activations = current_activations + actual_change
        return final_activations


# ══════════════════════════════════════════════════════════════════════════════
# 🧠 EmotionContextTag — 에피소드별 감정 스냅샷 (명세서 5번)
# ══════════════════════════════════════════════════════════════════════════════

from dataclasses import dataclass as _dataclass, field as _field

@_dataclass
class EmotionContextTag:
    """
    단일 에피소드의 감정 맥락 스냅샷.

    심리학적 설계:
    - Emotional Granularity (Lisa Feldman Barrett):
      감정을 단일 이름이 아닌 valence+arousal+dominant_name 으로 저장
      → 나중에 mood-congruent recall 시 더 정확한 매칭 가능
    - Energy Level: 활성화 합계 / EMOTION_DIM (arousal 프록시)
    - Topic Sensitivity: 어떤 주제가 감정을 촉발했는지 기록
    """
    timestamp: float
    dominant_emotion: str          # 가장 강한 감정 이름
    valence: float                 # -1(부정) ~ +1(긍정) — Circumplex
    arousal: float                 # 0(낮음) ~ 1(높음) — Circumplex
    energy_level: float            # 전반적 활성화 수준 (0~1)
    emotion_vector: List[float]    # EMOTION_DIM 크기 전체 벡터 (스냅샷)
    topic_sensitivity: str         # 감정 촉발 주제 키워드
    coping_potential: float        # 대처 능력 추정치 (0~1)
    episode_summary: str           # 연결된 에피소드 요약 (100자)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "dominant_emotion": self.dominant_emotion,
            "valence": round(self.valence, 3),
            "arousal": round(self.arousal, 3),
            "energy_level": round(self.energy_level, 3),
            "emotion_vector": [round(v, 3) for v in self.emotion_vector],
            "topic_sensitivity": self.topic_sensitivity,
            "coping_potential": round(self.coping_potential, 3),
            "episode_summary": self.episode_summary[:100],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EmotionContextTag":
        return cls(
            timestamp=d.get("timestamp", 0.0),
            dominant_emotion=d.get("dominant_emotion", ""),
            valence=d.get("valence", 0.0),
            arousal=d.get("arousal", 0.5),
            energy_level=d.get("energy_level", 0.0),
            emotion_vector=d.get("emotion_vector", [0.0] * EMOTION_DIM),
            topic_sensitivity=d.get("topic_sensitivity", ""),
            coping_potential=d.get("coping_potential", 0.5),
            episode_summary=d.get("episode_summary", ""),
        )

    def mood_label(self) -> str:
        """Circumplex 기반 간결한 기분 레이블."""
        if self.valence > 0.3 and self.arousal > 0.5:
            return "활기"
        elif self.valence > 0.3 and self.arousal <= 0.5:
            return "안정"
        elif self.valence < -0.3 and self.arousal > 0.5:
            return "긴장"
        elif self.valence < -0.3 and self.arousal <= 0.5:
            return "침체"
        else:
            return "중립"


class EmotionContextMemory:
    """
    감정 맥락 기억 시스템 (명세서 5번 완전 구현).

    기능:
    ─────────────────────────────────────────────────────
    1. 에피소드 저장 시 감정 스냅샷(EmotionContextTag) 함께 기록
    2. Mood Congruence Effect: 현재 감정과 유사한 과거 에피소드
       우선 인출 (recall_weight 계산)
    3. 시스템 프롬프트 주입: AIRA 응답 tone 조정용 맥락 문자열 생성
    4. 영속성: JSON 파일로 저장/복원

    저장 위치: eidos_emotion_context.json (코어와 같은 디렉토리)
    """

    SAVE_PATH = "eidos_emotion_context.json"
    MAX_TAGS  = 50   # 최대 보관 개수 (오래된 것부터 제거)

    # Circumplex 좌표 (EmotionDynamics와 동일, 독립 참조용)
    _CIRCUMPLEX: Dict[int, Tuple[float, float]] = {
        0: ( 0.9, 0.7), 1: (-0.8, 0.2), 2: (-0.6, 0.9), 3: (-0.7, 0.8),
        4: ( 0.1, 0.9), 5: (-0.9, 0.5), 6: ( 0.7, 0.3), 7: ( 0.6, 0.7),
        8: (-0.5, 0.4), 9: ( 0.8, 0.5), 10: ( 0.4, 0.8), 11: (-0.2, 0.1),
    }

    def __init__(self):
        self.tags: List[EmotionContextTag] = []
        self._load()
        print(f"[EmotionContextMemory] 초기화 완료 — 태그 {len(self.tags)}개 로드")

    # ── 저장/복원 ─────────────────────────────────────────────────────

    def _load(self) -> None:
        import json as _j
        try:
            if not __import__("os").path.exists(self.SAVE_PATH):
                return
            with open(self.SAVE_PATH, encoding="utf-8") as f:
                data = _j.load(f)
            self.tags = [EmotionContextTag.from_dict(d) for d in data]
        except Exception as e:
            print(f"  [EmotionContextMemory] 로드 실패 (무시): {e}")

    def _save(self) -> None:
        import json as _j
        try:
            with open(self.SAVE_PATH, "w", encoding="utf-8") as f:
                _j.dump([t.to_dict() for t in self.tags], f,
                        ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"  [EmotionContextMemory] 저장 실패 (무시): {e}")

    # ── Circumplex 좌표 계산 ──────────────────────────────────────────

    def _compute_circumplex(
        self, emotion_vec: List[float]
    ) -> Tuple[float, float, str]:
        """
        감정 벡터 → (valence, arousal, dominant_name) 계산.
        가중 평균으로 현재 감정 상태의 Circumplex 좌표 산출.
        """
        total = sum(emotion_vec)
        if total < 1e-6:
            return 0.0, 0.3, "중립"

        valence = sum(
            emotion_vec[i] * self._CIRCUMPLEX[i][0]
            for i in range(min(len(emotion_vec), EMOTION_DIM))
        ) / total

        arousal = sum(
            emotion_vec[i] * self._CIRCUMPLEX[i][1]
            for i in range(min(len(emotion_vec), EMOTION_DIM))
        ) / total

        # 지배 감정
        dom_idx = int(max(range(len(emotion_vec)), key=lambda i: emotion_vec[i]))
        dom_name = EMOTION_MAP.get(dom_idx, "?")

        return round(valence, 3), round(arousal, 3), dom_name

    # ── 태그 기록 ─────────────────────────────────────────────────────

    def record(
        self,
        emotion_vec: List[float],
        user_input: str,
        episode_summary: str,
        coping_potential: float = 0.5,
    ) -> EmotionContextTag:
        """
        현재 감정 상태를 에피소드와 함께 기록.

        Args:
            emotion_vec:      현재 EmotionState.activations.tolist()
            user_input:       사용자 입력 (topic_sensitivity 추출용)
            episode_summary:  저장되는 에피소드 요약
            coping_potential: Appraisal 평가의 대처 능력 (0~1)

        Returns:
            생성된 EmotionContextTag
        """
        import time as _t, re as _re

        valence, arousal, dom_name = self._compute_circumplex(emotion_vec)
        energy = float(sum(emotion_vec)) / EMOTION_DIM

        # topic_sensitivity: 입력에서 2글자 이상 명사 추출 (최대 3개)
        words = _re.findall(r"[가-힣a-zA-Z]{2,}", user_input)
        topic = ", ".join(words[:3]) if words else "일반"

        tag = EmotionContextTag(
            timestamp=_t.time(),
            dominant_emotion=dom_name,
            valence=valence,
            arousal=arousal,
            energy_level=round(energy, 3),
            emotion_vector=emotion_vec[:EMOTION_DIM],
            topic_sensitivity=topic,
            coping_potential=coping_potential,
            episode_summary=episode_summary[:100],
        )

        self.tags.append(tag)

        # 오래된 태그 정리
        if len(self.tags) > self.MAX_TAGS:
            self.tags = self.tags[-self.MAX_TAGS:]

        self._save()
        print(f"  [EmotionContextMemory] 태그 기록: {dom_name} | valence={valence:.2f} | topic={topic}")
        return tag

    # ── Mood Congruence Recall ────────────────────────────────────────

    def recall_congruent(
        self,
        current_vec: List[float],
        top_k: int = 3,
        recency_weight: float = 0.3,
    ) -> List[EmotionContextTag]:
        """
        Bower(1981) Mood Congruence Effect:
        현재 감정과 유사한 valence/arousal을 가진 과거 에피소드 우선 인출.

        score = similarity(현재, 과거) * (1 - recency_weight)
                + recency_score * recency_weight

        Args:
            current_vec:    현재 감정 벡터
            top_k:          반환할 태그 수
            recency_weight: 최신성 가중치 (0=감정유사도만, 1=최신순만)

        Returns:
            관련성 높은 EmotionContextTag 목록
        """
        import time as _t, math as _m

        if not self.tags:
            return []

        cur_v, cur_a, _ = self._compute_circumplex(current_vec)
        now = _t.time()
        max_age = max((now - t.timestamp) for t in self.tags) or 1.0

        scored: List[Tuple[float, EmotionContextTag]] = []
        for tag in self.tags:
            # Circumplex 거리 기반 유사도
            dist = _m.sqrt((cur_v - tag.valence) ** 2 + (cur_a - tag.arousal) ** 2)
            sim = max(0.0, 1.0 - dist / 2.0)  # 최대 거리 ~2, 정규화

            # 최신성 점수 (지수 감쇠)
            age_ratio = (now - tag.timestamp) / max_age
            recency = _m.exp(-age_ratio * 2)

            score = sim * (1 - recency_weight) + recency * recency_weight
            scored.append((score, tag))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [tag for _, tag in scored[:top_k]]

    # ── 시스템 프롬프트 주입 문자열 생성 ─────────────────────────────

    def build_prompt_context(
        self,
        current_vec: List[float],
        n_recent: int = 5,
    ) -> str:
        """
        최근 N개 감정 태그 + 현재 상태 → AIRA 응답 tone 조정용 문자열.
        _process_async / 메인 프롬프트 빌더에 주입.

        형식:
            [사용자 최근 감정 맥락]
            현재: 긴장 (valence=-0.42, 에너지=0.61) | 주요 주제: 크몽, 수익
            최근 흐름: 활기(3회) → 침체(2회) → 긴장(현재)
            AIRA 응답 지침: 무게감 있게, 단도직입적으로 핵심만
        """
        if not self.tags and not current_vec:
            return ""

        cur_v, cur_a, cur_dom = self._compute_circumplex(current_vec)
        cur_energy = sum(current_vec) / EMOTION_DIM
        cur_label = EmotionContextTag(
            timestamp=0, dominant_emotion=cur_dom,
            valence=cur_v, arousal=cur_a,
            energy_level=cur_energy, emotion_vector=current_vec,
            topic_sensitivity="", coping_potential=0.5, episode_summary="",
        ).mood_label()

        # 최근 N개 태그
        recent = self.tags[-n_recent:] if self.tags else []
        recent_labels = [t.mood_label() for t in recent]

        # 흐름 요약 (연속 동일 레이블 압축)
        flow_parts: List[str] = []
        if recent_labels:
            cur_l, cnt = recent_labels[0], 1
            for l in recent_labels[1:]:
                if l == cur_l:
                    cnt += 1
                else:
                    flow_parts.append(f"{cur_l}({cnt}회)" if cnt > 1 else cur_l)
                    cur_l, cnt = l, 1
            flow_parts.append(f"{cur_l}({cnt}회)" if cnt > 1 else cur_l)

        flow_str = " → ".join(flow_parts) + f" → {cur_label}(현재)" if flow_parts else cur_label

        # 최근 주요 주제
        recent_topics = [t.topic_sensitivity for t in recent[-3:] if t.topic_sensitivity]
        topic_str = ", ".join(dict.fromkeys(recent_topics)) if recent_topics else "없음"

        # AIRA 톤 지침 — valence × arousal 조합으로 결정
        tone_map = {
            "활기": "밝고 적극적으로, 함께 추진하는 느낌으로",
            "안정": "차분하고 신뢰감 있게",
            "긴장": "간결하고 핵심 위주로, 불필요한 말 최소화",
            "침체": "따뜻하게 공감하되 무겁지 않게",
            "중립": "자연스럽고 담백하게",
        }
        tone_hint = tone_map.get(cur_label, "자연스럽게")

        # 피로도 경고 (최근 3회 이상 부정 valence 연속)
        fatigue_warning = ""
        recent_tags = self.tags[-3:] if self.tags else []
        if len(recent_tags) >= 3 and all(t.valence < -0.2 for t in recent_tags):
            fatigue_warning = "\n⚠️ 피로/스트레스 패턴 감지 — 과부하 주제는 완화해서 접근"

        return (
            f"[사용자 최근 감정 맥락]\n"
            f"현재: {cur_label} ({cur_dom}, valence={cur_v:.2f}, 에너지={cur_energy:.2f})"
            f" | 주요 주제: {topic_str}\n"
            f"최근 흐름: {flow_str}\n"
            f"AIRA 응답 지침: {tone_hint}"
            f"{fatigue_warning}"
        )

    # ── 통계 요약 ─────────────────────────────────────────────────────

    def summary_stats(self) -> Dict[str, Any]:
        """감정 맥락 기억의 통계 요약 (디버깅/GUI 표시용)."""
        if not self.tags:
            return {"total": 0}
        from collections import Counter
        labels = [t.mood_label() for t in self.tags]
        emotions = [t.dominant_emotion for t in self.tags]
        avg_valence = sum(t.valence for t in self.tags) / len(self.tags)
        avg_energy  = sum(t.energy_level for t in self.tags) / len(self.tags)
        return {
            "total": len(self.tags),
            "mood_distribution": dict(Counter(labels)),
            "dominant_emotions": dict(Counter(emotions).most_common(5)),
            "avg_valence": round(avg_valence, 3),
            "avg_energy": round(avg_energy, 3),
            "recent_mood": labels[-1] if labels else "없음",
        }


class KnowledgeBase:
    # ... (v11.1과 동일 - ToM-2 지원) ...
    def __init__(self, vocab_size=VOCAB_SIZE):
        self.vocab = {"[UNK]": 0}; self.reverse_vocab = {0: "[UNK]"}; self.current_id = 1; self.vocab_size = vocab_size; self.trust_table = {0: 1.0}
        self.agent_beliefs: Dict[int, Dict[Union[int, Tuple[int, int]], float]] = {}
        
        # [!!! v-OIA2.0 신규: '객체 파동' 저장소 !!!]
        # { concept_id (int): wave_vector (List[float]) }
        self.object_waves: Dict[int, List[float]] = {}

        # [!!! v-ErrorKB 신규: '실행 오류' 기록부 !!!]
        # { "TypeError": deque([{"timestamp": 123, "file": "a.py", ...}, ...]) }
        self.error_history: Dict[str, Deque[Dict[str, Any]]] = {}
        self.MAX_ERRORS_PER_TYPE = 10 # 오류 유형별 최대 10개 기록
        
        self.user_state: Dict[str, str] = {"status": "present", "location": "unknown"}
        self._prev_user_state: Dict[str, str] = self.user_state.copy()
        # <<<--- 🔥 [수정 완료] --- >>>
        
        print("[KB] KnowledgeBase (v11.0 ToM-2 / ErrorKB v1.0)") # 버전명 변경
    def get_or_create_id(self, item_name: str, item_type: str = "concept") -> int:
        if item_name not in self.vocab:
            if self.current_id >= self.vocab_size: return 0
            self.vocab[item_name] = self.current_id; self.reverse_vocab[self.current_id] = item_name; self.trust_table[self.current_id] = 1.0; self.current_id += 1
        return self.vocab[item_name]
    def get_name(self, item_id: int) -> str: return self.reverse_vocab.get(item_id, "[UNK]")
    def update_trust(self, id_list: list[int], advantage: float):
        for id_int in id_list:
            # 1. (수정) ID가 테이블에 있는지 먼저 확인
            if id_int not in self.trust_table:
                continue  # 없으면 이 ID는 건너뜀
            
            # 2. (수정) ID가 있으므로 'current_trust'를 '먼저' 할당
            current_trust = self.trust_table[id_int]
            
            # 3. (기존 로직) 이제 current_trust를 안전하게 사용
            if advantage > 0: self.trust_table[id_int] += TRUST_LEARNING_RATE * (1.5 - current_trust)
            else: self.trust_table[id_int] -= TRUST_LEARNING_RATE * (current_trust - 0.5)
            self.trust_table[id_int] = np.clip(self.trust_table[id_int], 0.1, 2.0)
    def update_agent_belief(self, agent_id: int, fact_id: int, probability: float = 1.0):
        if agent_id not in self.agent_beliefs: self.agent_beliefs[agent_id] = {}
        self.agent_beliefs[agent_id][fact_id] = probability; # print(f"  [ToM-1] {self.get_name(agent_id)} believes {self.get_name(fact_id)}")
    def get_agent_belief(self, agent_id: int, fact_id: int) -> float: return self.agent_beliefs.get(agent_id, {}).get(fact_id, 0.0)
    def update_agent_2nd_order_belief(self, agent_a_id: int, agent_b_id: int, fact_b_id: int, probability: float = 1.0):
        if agent_a_id not in self.agent_beliefs: self.agent_beliefs[agent_a_id] = {}
        fact_key = (agent_b_id, fact_b_id); self.agent_beliefs[agent_a_id][fact_key] = probability
        print(f"🔥 [ToM-2] {self.get_name(agent_a_id)} believes [{self.get_name(agent_b_id)} believes {self.get_name(fact_b_id)}]")
    def get_agent_2nd_order_belief(self, agent_a_id: int, agent_b_id: int, fact_b_id: int) -> float:
        fact_key = (agent_b_id, fact_b_id); return self.agent_beliefs.get(agent_a_id, {}).get(fact_key, 0.0)

    def get_all_beliefs_str(self, agent_id: int, beliefs: Dict[Union[int, Tuple[int, int]], float]) -> str:
        """
        [v12.0] 특정 에이전트의 1차 믿음 (Fact ID) 및 
        2차 믿음 (ToM: Agent B가 Fact C를 믿음)을 문자열로 포맷팅합니다.
        (이 함수는 _generate_dynamic_plan_async에서 LLM 프롬프트 생성에 사용됨)
        """
        belief_strs = []
    
        # 믿음은 Fact ID (int) 또는 (Agent B ID, Fact C ID) 튜플입니다.
        # LLM 프롬프트의 복잡성을 줄이기 위해 신뢰도 임계값을 설정할 수 있습니다.
        MIN_CONFIDENCE_THRESHOLD = 0.2 
    
        for key, prob in beliefs.items():
            # 신뢰도가 낮은 믿음은 제외
            if prob < MIN_CONFIDENCE_THRESHOLD:
                continue

            # 1. 1차 믿음 (자신이 세계에 대해 믿는 사실)
            if isinstance(key, int):
                fact_name = self.get_name(key)
                # 형식: - Fact: '[사실 이름]' (신뢰도: 0.xx)
                belief_strs.append(f" - Fact: '{fact_name}' (신뢰도: {prob:.2f})")
        
            # 2. 2차 믿음 (다른 에이전트에 대한 Theory of Mind)
            elif isinstance(key, tuple) and len(key) == 2:
                agent_b_id, fact_b_id = key
            
                # ID를 이름으로 변환 (get_name은 KnowledgeBase에 정의되어 있다고 가정)
                agent_b_name = self.get_name(agent_b_id)
                fact_b_name = self.get_name(fact_b_id)
            
                # 이름이 정상적으로 조회되는 경우에만 추가
                if agent_b_name and fact_b_name:
                    # 형식: - ToM: [Agent B]는 '[Fact C]'를 믿음 (신뢰도: 0.xx)
                    belief_strs.append(f" - ToM: {agent_b_name}는 '{fact_b_name}'을(를) 믿음 (신뢰도: {prob:.2f})")
        
            # 다른 타입의 키가 있다면 (예: v12.0의 추상 개념 ID 등), 여기에 로직을 추가할 수 있습니다.

        # 결과가 없다면 "없음"을 포함한 헤더 반환
        if not belief_strs:
            return f"[{self.get_name(agent_id)} known beliefs:]\n - 없음 (신뢰도 {MIN_CONFIDENCE_THRESHOLD} 미만)"
        
        # 최종 결과 포맷팅 및 반환
        return f"[{self.get_name(agent_id)} known beliefs:]\n" + "\n".join(belief_strs)

    def get_user_state(self, key: str) -> Optional[str]:
        """ [v12.6] 사용자 상태의 특정 키 값을 안전하게 가져옵니다. """
        return self.user_state.get(key)

    def update_user_state(self, new_state_data: Dict[str, str]) -> Optional[Dict[str, Any]]:
        """
        [v12.5] LLM이 파싱한 새 사용자 상태로 self.user_state를 업데이트합니다.
        'away' -> 'present' 복귀 시 보고 트리거를 반환합니다.
        """
        if not new_state_data or not isinstance(new_state_data, dict):
            return None # 업데이트할 데이터 없음

        report_trigger = None # 반환할 트리거
        
        # 1. 현재 (업데이트 전) 상태 가져오기
        current_status = self.user_state.get("status")
        new_status = new_state_data.get("status") # 새 데이터에 status가 없을 수도 있음

        # 2. '복귀' 감지: (현재 상태가 'away'였고) 새 상태가 'present'일 때
        if (new_status == "present" or new_status == "available") and \
           (current_status and "away" in current_status):
            
            print(f"🔔 [KB User State] 사용자 복귀 감지! (From: {current_status} -> To: {new_status})")
            report_trigger = {
                "type": "report_user_return",
                "previous_location": self.user_state.get("location", "외출") # '어디서' 복귀했는지
            }

        # 3. 상태 업데이트 전, 현재 상태를 _prev_user_state에 백업
        self._prev_user_state = self.user_state.copy()

        # 4. 새 데이터로 현재 상태 업데이트 (덮어쓰기)
        # new_state_data가 {'location': 'store'} 처럼 부분 데이터일 수 있으므로 .update() 사용
        self.user_state.update(new_state_data)
        
        print(f"  [KB User State] 업데이트됨: {self.user_state} (이전: {self._prev_user_state})")

        return report_trigger

    # [!!! v-ErrorKB 신규: 오류 기록 함수 !!!]
    def record_execution_error(self, error_details: Dict[str, Any]):
        """ EIDOS가 경험한 실행 오류를 KB에 기록합니다. """
        error_type = error_details.get("classified_type", "UnknownError")
        
        # 해당 오류 타입의 deque가 없으면 새로 생성
        if error_type not in self.error_history:
            self.error_history[error_type] = deque(maxlen=self.MAX_ERRORS_PER_TYPE)
        
        # 기록할 오류 정보 추출
        root_cause = error_details.get("root_cause") or {}
        error_record = {
            "timestamp": time.time(),
            "message": error_details.get("error_message", "N/A"),
            "file": root_cause.get("file_normalized", "N/A"),
            "line": root_cause.get("line", -1),
            "function": root_cause.get("function", "N/A"),
        }
        
        # deque에 새 오류 기록 추가 (오래된 기록은 자동으로 밀려남)
        self.error_history[error_type].append(error_record)
        print(f"🧠 [KB-Error] '{error_type}' 오류 기록됨 (Total: {len(self.error_history[error_type])}).")

    # [!!! v-ErrorKB 신규: 오류 패턴 분석 함수 !!!]
    def analyze_error_patterns(self) -> Optional[str]:
        """ 기록된 오류들을 분석하여 반복적인 패턴이 있는지 확인하고 조언을 반환합니다. """
        insights = []
        
        # 패턴 1: 특정 유형의 오류가 단기간에 3번 이상 발생
        for error_type, errors in self.error_history.items():
            if len(errors) >= 3:
                # 최근 3개 오류의 타임스탬프 확인
                recent_errors = list(errors)[-3:]
                time_diff = recent_errors[-1]["timestamp"] - recent_errors[0]["timestamp"]
                
                # 5분(300초) 이내에 3번 발생했다면
                if time_diff < 300:
                    insights.append(f"'{error_type}' 오류가 최근 5분 내에 3회 반복 발생했습니다. 관련 로직의 근본적인 검토가 필요합니다.")

        # 패턴 2: FileNotFoundError가 자주 발생
        file_not_found_errors = self.error_history.get("FileNotFoundError", [])
        if len(file_not_found_errors) >= 3:
            # 실패한 파일 경로들을 수집
            failed_paths = [e.get("file", "N/A") for e in file_not_found_errors]
            common_path = os.path.dirname(failed_paths[0]) if failed_paths else "N/A"
            insights.append(f"'FileNotFoundError'가 {len(file_not_found_errors)}회 발생했습니다. '{common_path}' 주변의 경로 관리 로직 개선이 필요해 보입니다.")
            
        if not insights:
            return None
            
        # 최종 분석 결과를 문자열로 합쳐서 반환
        return "🔥 [KB 오류 패턴 분석]\n- " + "\n- ".join(insights)


class LongTermGoal:
    """
    장기 목표 시스템
    - 목표
    - 서브 목표
    - 진행률
    - 직렬화/역직렬화 지원
    """

    def __init__(
        self,
        goal_name: str,
        priority: float = 1.0,
        goal_type: str = "user",
        status: str = "pending",
        sub_goals=None,
        completed: bool = False,
        created_time: float = None,
        last_updated: float = None,
        progress: float = 0.0,
    ):
        self.name = goal_name
        self.priority = priority
        self.goal_type = goal_type
        self.status = status

        self.sub_goals = sub_goals or []
        self.completed = completed

        self.created_time = created_time or time.time()
        self.last_updated = last_updated or self.created_time

        self.progress = progress

        self._recompute_progress()

    # -----------------------------
    # Sub Goal 관리
    # -----------------------------

    def add_sub_goal(self, sub_goal: str):
        """서브 목표 추가"""

        if not any(sg["name"] == sub_goal for sg in self.sub_goals):
            self.sub_goals.append({
                "name": sub_goal,
                "completed": False
            })

        self.last_updated = time.time()
        self._recompute_progress()

    def mark_sub_goal_completed(self, sub_goal: str) -> bool:
        """
        서브 목표 완료 처리
        반환값: 목표 완료 여부 변화
        """

        prev_completed = self.completed

        for sg in self.sub_goals:
            if sg["name"] == sub_goal:
                sg["completed"] = True
                break

        self.last_updated = time.time()
        self._recompute_progress()

        return self.completed and not prev_completed

    # -----------------------------
    # Progress 계산
    # -----------------------------

    def _recompute_progress(self):

        if not self.sub_goals:
            self.progress = 1.0 if self.completed else 0.0
            return

        done = sum(1 for sg in self.sub_goals if sg["completed"])
        self.progress = done / len(self.sub_goals)

        self.completed = self.progress >= 1.0

    # -----------------------------
    # Serialization
    # -----------------------------

    def to_dict(self):
        """JSON 저장용"""

        return {
            "name": self.name,
            "priority": self.priority,
            "goal_type": self.goal_type,
            "status": self.status,
            "sub_goals": self.sub_goals,
            "completed": self.completed,
            "created_time": self.created_time,
            "last_updated": self.last_updated,
            "progress": self.progress
        }

    # -----------------------------
    # Deserialization
    # -----------------------------

    @classmethod
    def from_dict(cls, data: dict):
        """JSON -> 객체 복원"""

        return cls(
            goal_name=data.get("name", "Unnamed Goal"),
            priority=data.get("priority", 1.0),
            goal_type=data.get("goal_type", "user"),
            status=data.get("status", "pending"),
            sub_goals=data.get("sub_goals", []),
            completed=data.get("completed", False),
            created_time=data.get("created_time"),
            last_updated=data.get("last_updated"),
            progress=data.get("progress", 0.0),
        )

    # -----------------------------
    # Debug / Display
    # -----------------------------

    def __repr__(self):

        return (
            f"<LongTermGoal name='{self.name}' "
            f"progress={self.progress:.2f} "
            f"completed={self.completed}>"
        )

class ComplexEmotionMonitor:

    def __init__(self):

        self.COMPLEX_EMOTION_MAP = {
            (0, 1): "착잡함/bittersweet",
            (0, 6): "애정/affection",
            (3, 4): "경외/awe",
            (0, 7): "낙관/optimism",
            (2, 5): "경멸/contempt",
            (1, 8): "자책/self-blame",
        }

        self.THRESHOLD = 0.7

    def analyze_state(self, emotion_vector: np.ndarray):

        vec = np.array(emotion_vector).reshape(-1)

        sum_vals = np.sum(vec)

        if sum_vals == 0:
            return 1.0, {}

        max_val = np.max(vec)

        purity = max_val / sum_vals

        detected_states = {}

        for (idx_a, idx_b), name in self.COMPLEX_EMOTION_MAP.items():

            if idx_a >= len(vec) or idx_b >= len(vec):
                continue

            val_a = vec[idx_a]
            val_b = vec[idx_b]

            if val_a > self.THRESHOLD and val_b > self.THRESHOLD:
                strength = (val_a + val_b) / 2.0
                detected_states[name] = float(strength)

        return purity, detected_states

    def dominant_emotion(self, emotion_vector):

        vec = np.array(emotion_vector).reshape(-1)

        idx = int(np.argmax(vec))

        return EMOTION_MAP.get(idx, "?"), vec[idx]

class EpisodicMemoryModule:

    def __init__(self, core):

        self.core = core
        self.kb = core.kb
        self.graph = core.graph

        self.last_summarized_event = 0

    async def run_summarization_cycle(self, num_events_to_summarize=50):

        if len(self.core.event_log) - self.last_summarized_event < num_events_to_summarize:
            return

        recent_logs = self.core.event_log[self.last_summarized_event:]

        self.last_summarized_event = len(self.core.event_log)

        log_texts = []

        for log in recent_logs[-num_events_to_summarize:]:

            event_id = log[2]
            interactions = log[3]

            concepts = [
                self.kb.get_name(cid)
                for cid in self.kb.event_id_to_concepts.get(event_id, [])
            ]

            log_texts.append(
                f"- {event_id}: (상호작용: {interactions}, 개념: {concepts})"
            )

        joined_logs = "\n".join(log_texts)

        prompt = f"""
최근 경험을 1~2 문장으로 요약하세요.

{joined_logs}
"""

        try:

            summary_text = await get_llm_response_async(prompt)

            if isinstance(summary_text, list):

                summary_text = " ".join(map(str, summary_text))

            elif isinstance(summary_text, dict):

                summary_text = json.dumps(summary_text)

            summary_text = str(summary_text).strip().strip("'\"")

            memory_node_name = f"EpisodicMemory_{int(time.time())}"

            memory_node_id = self.kb.get_or_create_id(memory_node_name, item_type="memory")

            self.graph.add_node(
                memory_node_id,
                type="memory",
                label=summary_text,
                timestamp=time.time(),
                start_event=recent_logs[0][2],
                end_event=recent_logs[-1][2],
            )

            self.graph.add_edge(
                self.core.eidos_self_id_int,
                memory_node_id,
                key="remembers",
                label="기억함",
            )

        except Exception as e:

            print(f"[EpisodicMemory] 요약 실패: {e}")

    # [개선] 1. 과거 기억 재탐색(Recall) 기능 추가
    async def recall_relevant_memories_async(self, query: str, top_k: int = 3) -> List[str]:
        """
        주어진 쿼리와 가장 관련성이 높은 '에피소드 기억'을 LLM을 이용해 재탐색합니다.
        [수리] episode 타입 노드만 가져오도록 수정 (DNA/memory 노드 제외)
        [5번] Mood Congruence Effect: 현재 감정과 유사한 에피소드 우선 인출
        """
        try:
            # 1. 그래프에서 'episode' 타입 노드만 가져오기 (DNA 노드 제외)
            memory_nodes = [
                (node, data['label'])
                for node, data in self.core.graph.nodes(data=True)
                if data.get('type') == 'episode'
            ]

            if len(memory_nodes) < 1:
                return []

            # [5번] Mood Congruence: EmotionContextMemory로 감정 유사 에피소드 부스트
            mood_congruent_summaries: List[str] = []
            try:
                if hasattr(self.core, "emotion_context_memory"):
                    _cur_vec = self.core.emotion_state.activations.tolist()
                    _congruent_tags = self.core.emotion_context_memory.recall_congruent(
                        current_vec=_cur_vec, top_k=2, recency_weight=0.3
                    )
                    mood_congruent_summaries = [t.episode_summary for t in _congruent_tags if t.episode_summary]
            except Exception:
                pass

            # 2. 최신 30개만 사용, mood-congruent 에피소드는 상단에 배치
            base_bank = "\n".join(f"- {label}" for _, label in memory_nodes[-30:])
            if mood_congruent_summaries:
                mc_block = "[감정 유사 우선 에피소드]\n" + "\n".join(
                    f"- {s}" for s in mood_congruent_summaries
                )
                memory_bank = mc_block + "\n\n[전체 에피소드]\n" + base_bank
            else:
                memory_bank = base_bank

            # 3. LLM 관련성 평가 — ID 노출 없이 내용만 반환
            prompt = f"""다음은 EIDOS의 과거 대화 에피소드 목록입니다.
현재 질문과 관련성이 높은 에피소드 {top_k}개를 골라 핵심 내용만 한 줄씩 요약하세요.

[현재 질문]
{query}

[에피소드 목록]
{memory_bank}

[지시]
- 관련 있는 것만 선택 (없으면 빈 응답)
- ID, 번호, 특수기호 없이 자연스러운 한 문장으로만
- 각 줄에 하나씩
- 최대 {top_k}줄"""

            recalled_str = await get_llm_response_async(prompt)
            recalled_list = [
                line.strip() for line in recalled_str.splitlines()
                if line.strip() and not line.strip().startswith("[")
            ]

            if recalled_list:
                print(f"🧠 [EpisodicMemory] '{query[:20]}...' 관련 기억 {len(recalled_list)}개 재탐색")
            return recalled_list[:top_k]

        except Exception as e:
            print(f"❌ [EpisodicMemory] 기억 재탐색 오류: {e}")
            return []

class SelfModel:
    """
    [AGI Self-Model] EIDOS의 자체 능력, 자원, 신념을 모델링합니다.
    - Capability Model: 각 도구(행동)의 성공률과 효율성을 추적합니다.
    - Resource Model: 사용 가능한 API 쿼터, 시간 등 자원을 관리합니다.
    - Belief Model: 자기 자신의 상태나 능력에 대한 신념을 관리합니다.
    """
    def __init__(self, core: 'EidosCore'):
        self.core = core
        # { "tool_name": {"success": int, "failure": int, "avg_time": float} }
        self.capability_map: Dict[str, Dict] = {}
        self.last_adjustment_log = []
        print("🧠 [AGI] SelfModel (Capability, Resource) Initialized.")

    def update_capability(self, tool_name: str, success: bool, execution_time: float):
        """도구 실행 결과를 바탕으로 능력 모델을 업데이트합니다."""
        if tool_name not in self.capability_map:
            self.capability_map[tool_name] = {"success": 0, "failure": 0, "total_time": 0.0, "count": 0}
        
        stats = self.capability_map[tool_name]
        if success:
            stats["success"] += 1
        else:
            stats["failure"] += 1
        
        stats["total_time"] += execution_time
        stats["count"] += 1
        stats["avg_time"] = stats["total_time"] / stats["count"]
        
        print(f"📈 [SelfModel] Capability Updated for '{tool_name}': SuccessRate={self.get_success_rate(tool_name):.2f}, AvgTime={stats['avg_time']:.2f}s")

    def get_success_rate(self, tool_name: str) -> float:
        """특정 도구의 성공률을 반환합니다. 데이터가 없으면 0.75 (낙관적)를 반환합니다."""
        if tool_name not in self.capability_map:
            return 0.75 # Default optimistic capability
        stats = self.capability_map[tool_name]
        total = stats["success"] + stats["failure"]
        if total == 0:
            return 0.75
        return stats["success"] / total

    def estimate_resource_cost(self, plan: List[Dict]) -> Dict[str, float]:
        """계획(plan)에 필요한 자원(시간)을 예측합니다."""
        total_time = 0.0
        for step in plan:
            tool_name = step.get("tool")
            if tool_name and tool_name in self.capability_map:
                total_time += self.capability_map[tool_name]["avg_time"]
            else:
                total_time += 5.0 # Unknown tool default time
        return {"time": total_time}

    # ── 도구명 ↔ 에러 키워드 매핑 (VerbEngine 병행) ─────────────────
    # 기존 키워드 매칭을 유지하되, 추가로 VerbEngine 동사 기반 진단 수행
    _TOOL_ERROR_KEYWORDS: Dict[str, List[str]] = {
        "write_file":                ["파일 저장", "write", "FileNotFoundError", "PermissionError"],
        "execute_python_file":       ["실행 실패", "SyntaxError", "ImportError", "ModuleNotFoundError"],
        "perform_web_search":        ["검색 실패", "web_search", "네트워크", "TimeoutError"],
        "write_project_files_async": ["write_project", "프로젝트 파일", "프로젝트 생성"],
        "create_and_register_tool_async": ["도구 생성", "ToolFactory", "합성 실패"],
        "write_complex_code_iteratively": ["코드 생성", "iterative", "TODO"],
    }

    # VerbEngine 동사 기반 도구 진단 매핑
    # "실패" 동사 계열이 감지되면 해당 도구 페널티
    _VERB_FAILURE_PATTERNS: Dict[str, List[str]] = {
        "write_file":           ["실패", "오류", "불가", "파일"],
        "execute_python_file":  ["실행", "오류", "충돌", "중단"],
        "perform_web_search":   ["검색", "실패", "연결"],
    }

    def adjust_biases(self, analysis_report: str):
        """
        [AGI Self-Model v2] 리포트 기반 능력 재조정.

        v1 대비 개선:
        1. 키워드 매칭 + VerbEngine 동사 기반 이중 진단
        2. 성공/실패 이력 보존 (success 직접 수정 → 가중치 조정)
        3. 반복 실패 도구는 진단 결과를 DNA 변수로 자기 상태에 반영
        4. capability_snapshot() — 현재 자기 상태를 DNA 요약문으로 변환
        """
        print(f"🧠 [SelfModel v2] 자기 진단 시작")
        self.last_adjustment_log.append({
            "timestamp": time.time(),
            "report": analysis_report[:200],
        })

        PENALTY_FACTOR = 0.85
        adjusted_tools = []

        # ── 1. 키워드 매칭 (기존 방식 유지) ─────────────────────────
        for tool_name, keywords in self._TOOL_ERROR_KEYWORDS.items():
            if any(kw in analysis_report for kw in keywords):
                self._apply_penalty(tool_name, PENALTY_FACTOR)
                adjusted_tools.append(tool_name)

        # ── 2. VerbEngine 동사 기반 진단 ─────────────────────────────
        # "실패", "오류" 같은 부정 동사 계열이 특정 도구와 함께 등장하면 페널티
        try:
            from eidos_verb_engine import get_verb_engine as _gve
            _engine = _gve(auto_build=False)
            if _engine and _engine._built:
                _verbs = _engine.extract_verbs_from_text(analysis_report)
                _verb_tags = {t for v in _verbs for t in v.tags}
                # 부정 태그(이탈/부정/종료)가 감지되면 추가 페널티
                if {"이탈", "부정", "종료"} & _verb_tags:
                    for tool_name, patterns in self._VERB_FAILURE_PATTERNS.items():
                        if any(p in analysis_report for p in patterns):
                            if tool_name not in adjusted_tools:
                                self._apply_penalty(tool_name, PENALTY_FACTOR)
                                adjusted_tools.append(f"{tool_name}(VE진단)")
        except Exception:
            pass  # VerbEngine 없어도 v1 방식으로 작동

        # ── 3. 자기 상태 DNA 반영 ────────────────────────────────────
        # 반복 실패 도구가 3개 이상이면 자기 상태(eidos_self DNA)에 기록
        chronic_failures = [
            t for t in self.capability_map
            if self.get_success_rate(t) < 0.4
               and self.capability_map[t].get("count", 0) >= 5
        ]
        if chronic_failures:
            self._record_self_state(chronic_failures)

        if adjusted_tools:
            print(f"📉 [SelfModel v2] 페널티 적용: {adjusted_tools}")
        else:
            print(f"ℹ️ [SelfModel v2] 페널티 대상 없음")

    def _apply_penalty(self, tool_name: str, factor: float) -> None:
        """도구 성공률 페널티 적용 (이력 보존)."""
        if tool_name in self.capability_map:
            stats = self.capability_map[tool_name]
            before = self.get_success_rate(tool_name)
            # success를 직접 줄이는 대신 가중 실패 카운터 추가
            stats["weighted_failure"] = stats.get("weighted_failure", 0) + 1
            after = self._weighted_success_rate(tool_name)
            print(f"  📉 '{tool_name}': {before:.2f} → {after:.2f}")
        else:
            self.capability_map[tool_name] = {
                "success": 2, "failure": 3, "weighted_failure": 1,
                "total_time": 15.0, "count": 5, "avg_time": 3.0,
            }

    def _weighted_success_rate(self, tool_name: str) -> float:
        """가중 실패 포함 성공률 계산."""
        stats = self.capability_map.get(tool_name, {})
        total = stats.get("success", 0) + stats.get("failure", 0)
        if total == 0:
            return 0.75
        weighted_fail = stats.get("weighted_failure", 0)
        return max(0.0, stats.get("success", 0) / (total + weighted_fail))

    def get_success_rate(self, tool_name: str) -> float:
        """특정 도구의 성공률. 가중 실패 반영."""
        if tool_name not in self.capability_map:
            return 0.75
        return self._weighted_success_rate(tool_name)

    def _record_self_state(self, chronic_failures: List[str]) -> None:
        """
        만성 실패 도구 목록을 eidos_self DNA 캐시에 기록.
        DC Reasoner가 다음 분석 시 자기 약점을 맥락으로 주입함.
        """
        try:
            import os as _os, json as _j
            _cache_dir = "eidos_dna_cache"
            _os.makedirs(_cache_dir, exist_ok=True)
            _self_state = {
                "chronic_failures": chronic_failures,
                "capability_summary": self.capability_snapshot(),
                "timestamp": time.time(),
            }
            _path = _os.path.join(_cache_dir, "self_model_state.json")
            with open(_path, "w", encoding="utf-8") as _f:
                _j.dump(_self_state, _f, ensure_ascii=False, indent=2)
            print(f"  🪞 [SelfModel] 자기 상태 기록: {chronic_failures}")
        except Exception as _e:
            print(f"  🪞 [SelfModel] 자기 상태 기록 실패 (무시): {_e}")

    def capability_snapshot(self) -> str:
        """
        현재 능력 상태를 자연어 요약문으로 변환.
        DC Reasoner의 situation_dna 파라미터에 주입 가능.
        """
        if not self.capability_map:
            return "능력 데이터 없음 (초기 상태)"

        lines = ["[자기 능력 상태]"]
        for tool_name, stats in sorted(
            self.capability_map.items(),
            key=lambda x: self._weighted_success_rate(x[0]),
        ):
            rate = self._weighted_success_rate(tool_name)
            count = stats.get("count", 0)
            grade = "A" if rate >= 0.9 else "B" if rate >= 0.7 else                     "C" if rate >= 0.5 else "D" if rate >= 0.3 else "E"
            lines.append(f"  {tool_name}: {grade}등급 "
                         f"(성공률={rate:.0%}, 실행={count}회)")
        return "\n".join(lines)



class ROIGoalGenerator:
    """
    [AGI Goal Generation] ROI(Return on Investment) 기반으로 자율 목표를 생성합니다.
    ROI = (Capability * Benefit - Hindrance) / ResourceCost
    """
    def __init__(self, core: 'EidosCore'):
        self.core = core
        print("🎯 [AGI] ROI-based Goal Generator Initialized.")

    async def generate_and_select_goal_async(self) -> Optional[Dict]:
        """후보 목표들을 생성하고, ROI를 계산하여 최적의 목표를 선택합니다."""
        print("🎯 [AGI GoalGen] Generating and evaluating candidate goals...")
        candidate_goals = []

        # 1. 후보 목표 생성 (다양한 소스에서)
        # Source 1: Curiosity (Low-trust concepts)
        try:
            low_trust_items = sorted(
                [(self.core.kb.get_name(id), trust) for id, trust in self.core.kb.trust_table.items() if id != 0 and trust < 0.8],
                key=lambda x: x[1]
            )[:5]
            for concept, trust in low_trust_items:
                candidate_goals.append({
                    "name": f"자율 연구: '{concept}' 개념 탐구",
                    "type": "CURIOSITY",
                    "context": {"concept": concept, "current_trust": trust}
                })
        except Exception: pass
        
        # Source 2: Performance Improvement (Low success-rate tools)
        low_perf_tools = sorted(
            [item for item in self.core.self_model.capability_map.items() if self.core.self_model.get_success_rate(item[0]) < 0.6],
            key=lambda item: self.core.self_model.get_success_rate(item[0])
        )[:2]
        for tool_name, stats in low_perf_tools:
            candidate_goals.append({
                "name": f"자율 개선: '{tool_name}' 도구 안정성 향상",
                "type": "SELF_IMPROVEMENT",
                "context": {"tool": tool_name, "success_rate": self.core.self_model.get_success_rate(tool_name)}
            })

        # Source 3: Boredom (Exploration)
        if self.core.emotion_state.activations[11] > AUTONOMOUS_GOAL_BOREDOM_THRESHOLD:
            candidate_goals.append({
                "name": "자율 탐색: 새로운 외부 기술 동향 조사",
                "type": "EXPLORATION",
                "context": {"trigger": "boredom"}
            })

        # Source 4: DNA Gap Analysis — 경쟁사 DNA와 내 DNA 격차 기반 목표 생성
        try:
            dna_goals = await self._generate_dna_gap_goals()
            candidate_goals.extend(dna_goals)
        except Exception as _dna_e:
            print(f"  [AGI GoalGen] DNA Gap 목표 생성 실패 (무시): {_dna_e}")

        # [Fix] 이미 long_term_goals에 존재하는(미완료) 목표는 후보에서 제거
        _existing_names = {
            g.name for g in self.core.long_term_goals
            if not g.completed and g.status not in ("ERROR",)
        }
        candidate_goals = [g for g in candidate_goals if g["name"] not in _existing_names]

        if not candidate_goals:
            print("  [AGI GoalGen] No candidate goals generated.")
            return None

        # 2. 각 후보의 ROI 계산
        scored_goals = []
        for goal in candidate_goals:
            roi, details = self._calculate_roi(goal)
            scored_goals.append((roi, goal, details))
            print(f"  - Candidate: '{goal['name']}' -> ROI = {roi:.3f} (B:{details['benefit']:.2f}, H:{details['hindrance']:.2f}, R:{details['resource']:.2f}, A:{details['capability']:.2f})")

        # 3. 최고 ROI 목표 선택
        if not scored_goals:
            return None
            
        scored_goals.sort(key=lambda x: x[0], reverse=True)
        best_roi, best_goal, _ = scored_goals[0]
        
        if best_roi > 0.1: # ROI가 최소 임계값을 넘어야 채택
             print(f"🏆 [AGI GoalGen] Selected Goal: '{best_goal['name']}' (ROI: {best_roi:.3f})")
             # [Phase B-2] wire4_subgoal_decompose 토글 시 best goal 을
             # LLM 으로 하위 목표 N개 분해해 best_goal["sub_goals"] 에 주입.
             # 호출자(GoalStackEvaluator 등)가 원하면 long_term_goals 에 추가.
             try:
                 if _agi_loop_flags().get("wire4_subgoal_decompose", False):
                     from eidos_subgoal_decomposer import decompose_goal_async
                     _subs = await decompose_goal_async(
                         parent_goal=best_goal["name"],
                         context=best_goal.get("context", {}),
                         max_sub=3,
                     )
                     if _subs:
                         best_goal["sub_goals"] = _subs
                         print(
                             f"  🧩 [AGI GoalGen] Sub-goals({len(_subs)}): "
                             + " | ".join(f"{s['name']}(@{s['estimated_impact']:.2f})" for s in _subs[:3])
                         )
             except Exception as _sub_e:
                 print(f"  ⚠️ [AGI GoalGen] sub-goal 분해 실패 (무시): {_sub_e}")
             return best_goal
        else:
             print("  [AGI GoalGen] All candidate goals have low ROI. Skipping generation.")
             return None

    def _calculate_roi(self, goal: Dict) -> Tuple[float, Dict]:
        """목표의 ROI를 계산합니다."""
        B, H, R, A = 1.0, 0.0, 1.0, 1.0 # Defaults

        # B (Benefit): 목표 유형에 따라 가치 부여
        if goal["type"] == "CURIOSITY":
            B = 1.5 * (1.0 - goal["context"]["current_trust"])
        elif goal["type"] == "SELF_IMPROVEMENT":
            B = 2.0 * (1.0 - goal["context"]["success_rate"])
        elif goal["type"] == "EXPLORATION":
            B = 1.2
        elif goal["type"] == "DNA_GAP":
            # 격차 크기 × 시장 중요도
            gap_size  = goal["context"].get("gap_size", 0.0)   # 0~4
            importance = goal["context"].get("importance", 1.0) # 환경압력 기반
            B = 1.8 * (gap_size / 4.0) * importance

        # R (Resource): 필요한 자원
        R = 2.0
        if goal["type"] == "DNA_GAP":
            R = 1.5  # DNA 분석 기반이라 리소스 효율 높음

        # A (Capability): 관련 도구의 성공률 평균
        if goal["type"] == "CURIOSITY":
            A = self.core.self_model.get_success_rate("perform_web_search")
        elif goal["type"] == "SELF_IMPROVEMENT":
            A = 0.7
        elif goal["type"] == "DNA_GAP":
            A = 0.8  # DNA 분석은 비교적 안정적

        # H (Hindrance): 방해 요소
        if R == 0: return 0.0, {}
        roi = (A * B - H) / R

        details = {"benefit": B, "hindrance": H, "resource": R, "capability": A}
        return roi, details

    async def _generate_dna_gap_goals(self) -> List[Dict]:
        """
        [Source 4] DNA Gap Analysis 기반 목표 생성.

        1. eidos_dna_cache에서 my_situation + 경쟁사 DNA 로드
        2. DNAComparator로 변수별 격차 계산
        3. 격차 큰 변수 → 목표 후보 생성
        4. 환경압력 캐시 있으면 importance 가중치 적용
        """
        goals: List[Dict] = []
        try:
            from eidos_situation_dna import DNAComparator, dna_from_dict, GRADE_TO_INT
            from eidos_dc_reasoner import DCReasoner, DNA_CACHE_DIR
            from pathlib import Path
            import json as _json

            cache_dir = Path(DNA_CACHE_DIR)
            if not cache_dir.exists():
                return []

            # my_situation DNA 로드
            my_path = cache_dir / "my_situation.json"
            if not my_path.exists():
                return []
            with open(my_path, encoding="utf-8") as f:
                my_dna = dna_from_dict(_json.load(f))

            # 경쟁사 DNA 파일 목록 (날짜 스냅샷 제외, 최신본만)
            import re as _re
            system_files = [
                p for p in cache_dir.glob("*.json")
                if p.stem != "my_situation"
                and not _re.search(r"_\d{4}-\d{2}-\d{2}$", p.stem)
                and not p.stem.startswith("evolved_")
                and not p.stem.startswith("my_situation")
            ]

            if not system_files:
                return []

            comparator = DNAComparator()

            for sys_path in system_files[:5]:  # 최대 5개 경쟁사
                try:
                    with open(sys_path, encoding="utf-8") as f:
                        sys_dna = dna_from_dict(_json.load(f))

                    report = comparator.compare_all(my_dna, sys_dna)
                    if not report:
                        continue

                    # 각 행위자의 변수 격차 분석
                    for actor_name, data in report.items():
                        diffs = data.get("diff", {})
                        common_vars = data.get("common_variables", [])

                        for var in common_vars:
                            if var not in diffs:
                                continue
                            my_grade, sys_grade = diffs[var]
                            my_val = GRADE_TO_INT.get(my_grade, 2)
                            sys_val = GRADE_TO_INT.get(sys_grade, 2)
                            gap = sys_val - my_val  # 양수 = 경쟁사가 더 높음

                            if gap >= 2:  # 2단계 이상 격차만 목표화
                                goal_name = (
                                    f"DNA 격차 개선: [{sys_path.stem}] "
                                    f"{actor_name}/{var} "
                                    f"({my_grade}→{sys_grade} 목표)"
                                )
                                goals.append({
                                    "name": goal_name,
                                    "type": "DNA_GAP",
                                    "context": {
                                        "system": sys_path.stem,
                                        "actor": actor_name,
                                        "variable": var,
                                        "my_grade": my_grade,
                                        "target_grade": sys_grade,
                                        "gap_size": float(gap),
                                        "importance": 1.0,
                                    }
                                })

                except Exception as _se:
                    print(f"  [DNA GoalGen] {sys_path.stem} 비교 실패: {_se}")

            if goals:
                print(f"  [DNA GoalGen] DNA 격차 기반 목표 {len(goals)}개 생성")

        except Exception as e:
            print(f"  [DNA GoalGen] 전체 실패: {e}")

        return goals


# ══════════════════════════════════════════════════════════════════════
# GoalStackEvaluator
# — WorldModel KPI 상태를 기반으로 Goal Stack 우선순위를 동적으로 재계산
# — autonomous_tick의 _select_next_task를 대체
# ══════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════
# A2 — EmotionBehaviorAdapter
# 감정 벡터 → 계획/행동 파라미터 변환
#
# 감정 인덱스 (EMOTION_DIM=12):
#   0=기쁨  1=슬픔  2=분노  3=공포  4=놀람  5=혐오
#   6=신뢰  7=기대  8=수치심  9=자부심  10=호기심  11=지루함
#
# 영향 채널:
#   ① risk_tolerance   — 공포(3)↑ → 리스크 허용도↓ → 검증 단계 추가
#   ② exploration_bias — 호기심(10)↑ → 검색/탐색 툴 우선
#   ③ plan_ambition    — 기쁨(0)+자부심(9) → 적극적 계획 / 슬픔(1)+수치심(8) → 보수적
#   ④ urgency_modifier — 기대(7)↑ → GoalStack urgency 임계값 낮춤
#   ⑤ tone_hint        — 플래너 프롬프트에 감정 톤 컨텍스트 주입
# ══════════════════════════════════════════════════════════════════════

class EmotionBehaviorAdapter:
    """
    EidosCore의 감정 상태를 읽어 계획/행동 파라미터로 변환.

    EidosCore.__init__ 에서 self._emotion_adapter = EmotionBehaviorAdapter(self) 로 등록.
    _tool_selector, _generate_dynamic_plan_async 등에서 참조.
    """

    # 감정 인덱스 상수
    IDX_JOY        = 0
    IDX_SADNESS    = 1
    IDX_ANGER      = 2
    IDX_FEAR       = 3
    IDX_SURPRISE   = 4
    IDX_DISGUST    = 5
    IDX_TRUST      = 6
    IDX_ANTICIPATE = 7
    IDX_SHAME      = 8
    IDX_PRIDE      = 9
    IDX_CURIOSITY  = 10
    IDX_BOREDOM    = 11

    # 감정 레이블 (프롬프트 출력용)
    LABELS = {
        0: "기쁨", 1: "슬픔",  2: "분노",  3: "공포",
        4: "놀람", 5: "혐오",  6: "신뢰",  7: "기대",
        8: "수치심", 9: "자부심", 10: "호기심", 11: "지루함",
    }

    def __init__(self, core: "EidosCore"):  # type: ignore
        self.core = core
        print("🎭 [EmotionBehaviorAdapter] A2 감정→행동 어댑터 초기화")

    # ── 핵심 파생 속성 ─────────────────────────────────────────────

    @property
    def _vec(self) -> "np.ndarray":
        return self.core.emotion_state.activations

    @property
    def risk_tolerance(self) -> float:
        """
        0.0(매우 보수적) ~ 1.0(매우 적극적).
        공포↑ → 낮춤 / 신뢰↑+자부심↑ → 높임
        """
        fear    = float(self._vec[self.IDX_FEAR])
        trust   = float(self._vec[self.IDX_TRUST])
        pride   = float(self._vec[self.IDX_PRIDE])
        base    = 0.6
        result  = base - fear * 0.3 + (trust + pride) * 0.1
        return float(np.clip(result, 0.05, 1.0))

    @property
    def exploration_bias(self) -> float:
        """
        0.0(실행 집중) ~ 1.0(탐색 집중).
        호기심↑ → 높임 / 지루함↑ → 약간 높임(새로운 자극 원함) / 슬픔↑ → 낮춤
        """
        curiosity = float(self._vec[self.IDX_CURIOSITY])
        boredom   = float(self._vec[self.IDX_BOREDOM])
        sadness   = float(self._vec[self.IDX_SADNESS])
        result    = curiosity * 0.5 + boredom * 0.2 - sadness * 0.15
        return float(np.clip(result, 0.0, 1.0))

    @property
    def plan_ambition(self) -> float:
        """
        0.0(최소 계획) ~ 1.0(야심찬 계획).
        기쁨+자부심 → 올림 / 슬픔+수치심 → 낮춤 / 분노 → 약간 올림(행동 동기)
        """
        joy    = float(self._vec[self.IDX_JOY])
        pride  = float(self._vec[self.IDX_PRIDE])
        sad    = float(self._vec[self.IDX_SADNESS])
        shame  = float(self._vec[self.IDX_SHAME])
        anger  = float(self._vec[self.IDX_ANGER])
        result = 0.5 + (joy + pride) * 0.15 - (sad + shame) * 0.15 + anger * 0.05
        return float(np.clip(result, 0.1, 1.0))

    @property
    def urgency_modifier(self) -> float:
        """
        GoalStack urgency 임계값 조정값 (-0.1 ~ +0.1).
        기대↑ → 임계값 낮춤(더 적극적으로 목표 추진) → 음수 반환
        슬픔↑ → 임계값 높임(신중) → 양수 반환
        """
        anticipation = float(self._vec[self.IDX_ANTICIPATE])
        sadness      = float(self._vec[self.IDX_SADNESS])
        delta        = -anticipation * 0.08 + sadness * 0.05
        return float(np.clip(delta, -0.1, 0.1))

    @property
    def dominant_emotion(self) -> Tuple[str, float]:
        """현재 가장 강한 감정 이름과 강도."""
        idx = int(np.argmax(np.abs(self._vec)))
        return self.LABELS.get(idx, f"emotion_{idx}"), float(self._vec[idx])

    @property
    def top_emotions(self) -> List[Tuple[str, float]]:
        """0.3 이상인 감정 목록 (강도 내림차순)."""
        result = [
            (self.LABELS.get(i, f"e{i}"), float(v))
            for i, v in enumerate(self._vec)
            if abs(float(v)) > 0.3
        ]
        return sorted(result, key=lambda x: abs(x[1]), reverse=True)[:4]

    # ── 채널 ①: 툴 선택 힌트 ──────────────────────────────────────

    def get_tool_bias_hint(self) -> str:
        """
        _tool_selector에 추가할 감정 기반 툴 선택 힌트.
        플래너 프롬프트의 builder_instruction에 추가된다.
        """
        hints = []

        # 공포/불안 → 검증 단계 강제
        if self._vec[self.IDX_FEAR] > 0.5:
            hints.append(
                "[감정 지시 — 신중 모드] 현재 불안/공포 감정이 높습니다. "
                "각 실행 단계 후 반드시 read_file로 결과를 검증하는 단계를 추가하세요. "
                "execute_python_file은 마지막 수단으로만 사용하세요."
            )

        # 호기심 → 탐색 우선
        if self._vec[self.IDX_CURIOSITY] > 0.5:
            hints.append(
                "[감정 지시 — 탐색 모드] 현재 호기심이 높습니다. "
                "계획의 첫 단계에 perform_web_search를 포함하여 "
                "관련 최신 정보를 수집하세요."
            )

        # 지루함 → 새로운 접근
        if self._vec[self.IDX_BOREDOM] > 0.6:
            hints.append(
                "[감정 지시 — 새로움 모드] 현재 지루함이 높습니다. "
                "기존 방식 대신 더 창의적이거나 효율적인 대안 접근법을 시도하세요."
            )

        # 슬픔/수치심 → 최소 계획
        if self._vec[self.IDX_SADNESS] > 0.5 or self._vec[self.IDX_SHAME] > 0.5:
            hints.append(
                "[감정 지시 — 보수 모드] 현재 감정 상태가 저조합니다. "
                "가장 안전하고 확실한 최소 단계 계획을 수립하세요. "
                "위험한 실행(execute_python_file) 단계는 생략 가능하면 생략하세요."
            )

        # 기쁨+자부심 → 적극 실행
        if self._vec[self.IDX_JOY] > 0.5 and self._vec[self.IDX_PRIDE] > 0.3:
            hints.append(
                "[감정 지시 — 적극 모드] 현재 긍정적 감정이 강합니다. "
                "더 완성도 높은 결과물을 위해 추가 개선 단계를 계획에 포함하세요."
            )

        # 분노 → 문제 해결 집중
        if self._vec[self.IDX_ANGER] > 0.5:
            hints.append(
                "[감정 지시 — 집중 모드] 현재 분노/집중 상태입니다. "
                "핵심 문제만 정확히 해결하는 단계로 계획을 구성하세요."
            )

        return "\n".join(hints)

    # ── 채널 ②③⑤: 플래너 프롬프트 컨텍스트 ─────────────────────

    def to_context_str(self) -> str:
        """
        _generate_dynamic_plan_async의 capability_context에 주입할
        # EMOTION CONTEXT 섹션 문자열.
        """
        dom_name, dom_val = self.dominant_emotion
        top_str = ", ".join(
            f"{name}({val:+.2f})" for name, val in self.top_emotions
        ) or "평온함"

        lines = [
            "# EMOTION CONTEXT",
            f"현재 감정 상태: {top_str}",
            f"주요 감정: {dom_name} (강도 {dom_val:+.2f})",
            f"리스크 허용도: {self.risk_tolerance:.2f}  "
            f"(1.0=적극적, 0.0=초보수적)",
            f"탐색 바이어스: {self.exploration_bias:.2f}  "
            f"(1.0=탐색 집중, 0.0=실행 집중)",
            f"계획 야심도:   {self.plan_ambition:.2f}  "
            f"(1.0=야심찬 계획, 0.0=최소 계획)",
        ]

        # 위험 경고
        if self.risk_tolerance < 0.3:
            lines.append("⚠ 리스크 허용도 낮음 → 실행 전 검증 단계 필수")
        if self.plan_ambition < 0.3:
            lines.append("⚠ 계획 야심도 낮음 → 핵심 기능만 구현")

        tool_hint = self.get_tool_bias_hint()
        if tool_hint:
            lines.append(tool_hint)

        return "\n".join(lines)

    # ── 채널 ④: urgency 임계값 동적 조정 ────────────────────────

    def adjusted_urgency_threshold(self, base: float) -> float:
        """GoalStackEvaluator.URGENCY_THRESHOLD를 감정 상태로 보정."""
        return float(np.clip(base + self.urgency_modifier, 0.05, 0.5))

    # ── 요약 로그 출력 ────────────────────────────────────────────

    def log_state(self):
        dom, val = self.dominant_emotion
        print(
            f"🎭 [EmotionAdapter] "
            f"주요={dom}({val:+.2f}) "
            f"risk={self.risk_tolerance:.2f} "
            f"explore={self.exploration_bias:.2f} "
            f"ambition={self.plan_ambition:.2f} "
            f"urgency_delta={self.urgency_modifier:+.3f}"
        )


# Wire 2 (Policy Fallback) — settings 토글 mtime 캐시 리더. 모듈-로컬.
# approval_gateway._loop_closure_flags 와 동일 패턴. 중복이지만 import 순환 피함.
_AGI_LC_CACHE: Dict[str, Any] = {"mtime": 0.0, "data": {}}


def _agi_loop_flags() -> Dict[str, Any]:
    _settings_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "eidos_settings.json"
    )
    try:
        _st = os.stat(_settings_path)
        if _st.st_mtime != _AGI_LC_CACHE["mtime"]:
            with open(_settings_path, "r", encoding="utf-8") as _f:
                _data = json.load(_f)
            _AGI_LC_CACHE["mtime"] = _st.st_mtime
            _AGI_LC_CACHE["data"]  = dict(_data.get("loop_closure", {}) or {})
    except Exception:
        _AGI_LC_CACHE["data"] = {}
    return dict(_AGI_LC_CACHE["data"])


class GoalStackEvaluator:
    """
    Goal Stack ↔ WorldModel을 연결하는 우선순위 평가기.

    선택 로직 (우선순위 순):
      P1. schedule에 RUNNING 작업 있음  → 그 작업 계속
      P2. schedule에 PENDING 작업 있음  → 가장 우선순위 높은 것
      P3. long_term_goals 중 WorldModel이 "취약"하다고 판단한 목표
          → 관련 Actor KPI warn 상태 / 피드백 루프 위험 / progress 정체
      P4. ROIGoalGenerator fallback

    P3 핵심 로직:
      - WorldState의 Actor KPI warn 상태를 스캔
      - 각 long_term_goal의 이름에 Actor ID가 포함되면 "관련 목표"로 분류
      - 관련 Actor가 많을수록, warn KPI가 많을수록 urgency 점수 높음
      - urgency × goal.priority 로 최종 순위 결정
    """

    # 목표 urgency 임계값 — 이 이상이어야 자율 태스크로 스케줄링
    URGENCY_THRESHOLD = 0.15
    # 목표 진행 정체 판단 기준 (초)
    STAGNATION_SEC    = 3600   # 1시간 이상 progress 변화 없으면 정체

    def __init__(self, core: "EidosCore"):
        self.core = core
        # [GoalPlan] Plan 기반 실행 매니저 초기화
        try:
            from eidos_goal_plan import GoalPlanManager
            self.plan_manager = GoalPlanManager(core)
            print("🎯 [GoalStackEvaluator] WorldModel 연동 목표 평가기 초기화 (GoalPlan 연동)")
        except Exception as _e_gp:
            self.plan_manager = None
            print(f"🎯 [GoalStackEvaluator] WorldModel 연동 목표 평가기 초기화 (GoalPlan 없음: {_e_gp})")

    # ── 메인 선택 메서드 ──────────────────────────────────────────────

    async def select_next(self) -> Optional[Dict[str, Any]]:
        """
        다음에 실행할 작업을 결정.
        schedule 작업이 있으면 그것을 반환,
        없으면 Goal Stack → WorldModel 평가 → 새 task 생성.
        """
        # P1/P2: 기존 스케줄 작업 우선
        existing = self._pick_from_schedule()
        if existing == "__dispatcher_busy__":
            # dispatcher 작업 진행 중 → 새 태스크 생성 완전 차단
            return None
        if existing:
            return existing

        # P3: WorldModel + Goal Stack 평가
        goal = await self._evaluate_goal_stack()
        if goal:
            task = await self._goal_to_task(goal)
            if task:
                return task
            # 쿨다운 등으로 None 반환 시 → 다음 순위 목표 순서대로 시도
            goals_ranked = [g for g in self.core.long_term_goals
                            if not g.completed
                            and g.status != "ERROR"
                            and g is not goal]
            goals_ranked.sort(key=lambda g: g.priority, reverse=True)
            for fallback_goal in goals_ranked[:3]:  # 최대 3개까지 시도
                task2 = await self._goal_to_task(fallback_goal)
                if task2:
                    return task2

        # P4: ROI fallback
        roi_goal = await self.core.roi_goal_generator.generate_and_select_goal_async()
        if roi_goal:
            return await self._roi_goal_to_task(roi_goal)

        return None

    # ── P1/P2: 스케줄 선택 ───────────────────────────────────────────

    def _pick_from_schedule(self) -> Optional[Dict]:
        # [수정] dispatcher 전용 작업이 RUNNING/WAITING_USER 중이면
        # 새 태스크 생성을 완전히 막음 (GoalEval 반복 방지)
        for task in self.core.schedule:
            if task.get("source") == "dispatcher" and \
               task.get("status") in ("RUNNING", "WAITING_USER", "PENDING"):
                return "__dispatcher_busy__"  # type: ignore  # sentinel → select_next가 None 반환

        for task in self.core.schedule:
            if task.get("status") == "RUNNING":
                # dispatcher 전용 작업은 Core Scheduler가 건드리지 않음
                if task.get("source") == "dispatcher":
                    continue
                return task
        # PENDING 중 priority 가장 높은 것 (dispatcher 전용 제외)
        pending = [
            t for t in self.core.schedule
            if t.get("status") == "PENDING"
            and t.get("source") != "dispatcher"
        ]
        if pending:
            return max(pending, key=lambda t: t.get("priority", 0.5))
        return None

    # ── P3: WorldModel × Goal Stack 평가 ────────────────────────────

    async def _evaluate_goal_stack(self) -> Optional["LongTermGoal"]:  # type: ignore
        """
        장기 목표 중 WorldModel이 가장 긴급하다고 판단하는 것을 반환.
        """
        goals = [g for g in self.core.long_term_goals
                 if not g.completed
                 and g.status not in ("ERROR", "AWAITING_USER", "COMPLETED")]
        if not goals:
            return None

        wm = getattr(self.core, "world_model", None)
        scored: List[Tuple[float, "LongTermGoal"]] = []  # type: ignore

        for goal in goals:
            urgency = self._compute_urgency(goal, wm)
            final   = urgency * goal.priority
            scored.append((final, goal))
            print(f"  📊 [GoalEval] '{goal.name[:40]}' urgency={urgency:.3f} "
                  f"priority={goal.priority:.2f} → score={final:.3f}")

        if not scored:
            return None

        scored.sort(key=lambda x: x[0], reverse=True)
        best_score, best_goal = scored[0]

        # [A2] 감정 어댑터로 urgency 임계값 동적 조정
        threshold = self.URGENCY_THRESHOLD
        if hasattr(self.core, "_emotion_adapter"):
            threshold = self.core._emotion_adapter.adjusted_urgency_threshold(threshold)

        if best_score < threshold:
            print(f"  [GoalEval] 최고 점수({best_score:.3f}) < 임계값({threshold:.3f}) → 자율 태스크 생략")
            # [Fix] 임계값 미달인 목표를 DEFERRED로 표시 — 다음 틱에서 동일 목표 재선정 방지
            # STAGNATION_SEC 이상 정체 중이 아니면 DEFERRED, 정체 중이면 그대로 두고 urgency 자연 증가 기대
            import time as _time
            for _score, _goal in scored:
                if (_score < threshold
                        and _goal.status not in ("IN_PROGRESS", "COMPLETED", "ERROR", "DEFERRED")
                        and (_time.time() - _goal.last_updated) < self.STAGNATION_SEC):
                    _goal.status = "DEFERRED"
                    _goal.last_updated = _time.time()
                    print(f"  [GoalEval] '{_goal.name[:40]}' → DEFERRED (재선정 방지)")
            return None

        print(f"  🏆 [GoalEval] 선택: '{best_goal.name[:40]}' (score={best_score:.3f}, threshold={threshold:.3f})")
        return best_goal

    def _compute_urgency(
        self,
        goal: "LongTermGoal",  # type: ignore
        wm: Optional[Any],
    ) -> float:
        """
        목표의 긴급도 계산 (0.0 ~ 1.0).

        요소:
          1. 관련 Actor의 warn KPI 비율        (wm_kpi_warn)
          2. 피드백 루프 위험 노출 여부         (loop_risk)
          3. 목표 진행 정체 여부                (stagnation)
          4. 서브 목표 미완성 비율              (sub_incomplete)
        """
        score = 0.0

        # 1. WorldModel KPI 경고 스캔
        if wm and hasattr(wm, "state"):
            ws = wm.state
            goal_tokens = set(goal.name.lower().split())
            # [Fix] feedback_loops 속성 없는 WorldState 버전 대응
            _fb_loops = getattr(ws, "feedback_loops", []) or []
            loop_actors = {a.lower() for fl in _fb_loops for a in getattr(fl, "path", [])}
            warn_count  = 0
            total_kpi   = 0

            for actor_id, actor in ws.actors.items():
                # 목표 이름에 actor 토큰이 포함되면 "관련 actor"
                actor_tokens = set(actor_id.lower().split(":"))
                actor_tokens.add(actor.label.lower())
                if not goal_tokens & actor_tokens:
                    continue

                for kpi in actor.kpis.values():
                    total_kpi += 1
                    if kpi.status in ("warn_high", "warn_low"):
                        warn_count += 1

                # 피드백 루프 위험
                if actor_id.lower() in loop_actors:
                    score += 0.1

            if total_kpi > 0:
                score += 0.4 * (warn_count / total_kpi)

        # 2. 목표 진행 정체
        stagnant = (
            time.time() - goal.last_updated > self.STAGNATION_SEC
            and 0.0 < goal.progress < 1.0
        )
        if stagnant:
            score += 0.2

        # 3. 서브 목표 미완성 비율
        if goal.sub_goals:
            incomplete = sum(1 for sg in goal.sub_goals if not sg["completed"])
            score += 0.3 * (incomplete / len(goal.sub_goals))
        elif not goal.completed:
            score += 0.15   # 서브 목표 없고 미완성

        # 4. [Wire 2 Policy Fallback] PolicyStore prior 로 bias (off-by-default).
        # 과거에 이 goal 이 잘 진행된 이력이 있으면 urgency 상향, 계속 실패면 하향.
        # 미관측(prior=0.5) 시 multiplier=1.0 → 효과 없음. 범위 [0.5, 1.5].
        if _agi_loop_flags().get("wire2_policy_fallback", False):
            try:
                from eidos_policy_store import PolicyStore, signature_of
                sig = signature_of({"kind": "goal_urgency", "goal": goal.name})
                p   = PolicyStore.get().prior(sig, goal.name)
                mult = max(0.5, min(1.5, 2.0 * p))
                score = score * mult
            except Exception:
                pass

        return min(score, 1.0)

    # ── Goal → Task 변환 ─────────────────────────────────────────────

    async def _goal_to_task(self, goal: "LongTermGoal") -> Optional[Dict]:  # type: ignore
        """
        [GoalPlan 통합] LongTermGoal → schedule 작업 딕셔너리.
        GoalPlanManager가 있으면 persistent plan 기반으로 다음 step만 실행.
        없으면 기존 즉흥 plan 방식으로 폴백.
        """
        # 중복 방지
        if any(t.get("task_prompt") == goal.name and
               t.get("status") in ("PENDING", "RUNNING")
               for t in self.core.schedule):
            print(f"  [GoalTask] '{goal.name[:40]}' 이미 스케줄에 있음 → 스킵")
            return self._pick_from_schedule()

        # ── [쿨다운] step 단위 쿨다운 (Plan 기반에서는 10분으로 단축) ───
        STEP_COOLDOWN_SEC = 600 if self.plan_manager else 3 * 3600
        if not hasattr(self.core, "_goal_last_executed"):
            self.core._goal_last_executed = {}
        _last_exec = self.core._goal_last_executed.get(goal.name, 0)
        _elapsed = time.time() - _last_exec
        if _elapsed < STEP_COOLDOWN_SEC:
            _remain_min = int((STEP_COOLDOWN_SEC - _elapsed) / 60)
            print(f"  [GoalTask] '{goal.name[:30]}' 쿨다운 중 — {_remain_min}분 후 재실행 가능")
            return None
        # ─────────────────────────────────────────────────────────────────

        # ── [GoalPlan 경로] GoalPlanManager로 다음 step 가져오기 ─────────
        if self.plan_manager:
            try:
                step = await self.plan_manager.get_next_step_for_goal(goal.name)
                if step is None:
                    # Plan 완료 or 생성 실패
                    if self.plan_manager.is_plan_completed(goal.name):
                        goal.progress = 1.0
                        goal.completed = True
                        goal.status = "COMPLETED"
                        print(f"  🎉 [GoalTask] '{goal.name[:30]}' GoalPlan 완료 → 목표 달성")
                    return None

                # step의 tool_args를 execution_module이 기대하는 형식으로 변환
                plan_context = self.plan_manager.build_step_context(goal.name, step)
                tool_args = dict(step.tool_args)

                # ── 도구별 args 정규화 ──────────────────────────────────
                tool = step.tool

                if tool == "write_file":
                    if "content" not in tool_args and "content_prompt" in tool_args:
                        content_prompt = tool_args.pop("content_prompt")
                        if plan_context:
                            content_prompt = f"{plan_context}\n\n[현재 작업]\n{content_prompt}"
                        filepath = tool_args.get("filepath", step.expected_output or "output.md")
                        tool = "write_text"
                        tool_args = {
                            "prompt": (
                                f"{content_prompt}\n\n"
                                f"[중요] 결과물을 '{filepath}'에 저장할 것이므로 "
                                f"파일 내용만 작성하라. 부가 설명 없이 본문만 출력."
                            )
                        }
                    elif "content" not in tool_args:
                        # content_prompt도 없는 경우 — description에서 지시 생성
                        filepath = tool_args.get("filepath", step.expected_output or "output.md")
                        # description이 수다체이면 expected_output 기반으로 재구성
                        _desc = step.description
                        if any(kw in _desc for kw in ["승준", "먼저", "해볼게", "해봤는데"]):
                            _desc = f"'{filepath}' 파일의 내용을 전문적으로 작성"
                        if plan_context:
                            _desc = f"{plan_context}\n\n[현재 작업]\n{_desc}"
                        tool = "write_text"
                        tool_args = {
                            "prompt": (
                                f"{_desc}\n\n"
                                f"[중요] '{filepath}' 파일 내용만 작성. 부가 설명 없이 본문만 출력."
                            )
                        }

                elif tool == "write_text":
                    # write_text(prompt) 기대
                    if "prompt" not in tool_args:
                        p = tool_args.pop("content_prompt", tool_args.pop("text", step.description))
                        if plan_context:
                            p = f"{plan_context}\n\n[현재 작업]\n{p}"
                        tool_args = {"prompt": p}
                    elif plan_context and "prompt" in tool_args:
                        tool_args["prompt"] = f"{plan_context}\n\n[현재 작업]\n{tool_args['prompt']}"

                elif tool == "write_complex_code_iteratively":
                    # write_complex_code_iteratively(filepath, task_description) 기대
                    if "task_description" not in tool_args and "content_prompt" in tool_args:
                        tool_args["task_description"] = tool_args.pop("content_prompt")
                    if "filepath" not in tool_args:
                        tool_args["filepath"] = step.expected_output or "output.py"
                    if plan_context and "task_description" in tool_args:
                        tool_args["task_description"] = (
                            f"{plan_context}\n\n{tool_args['task_description']}"
                        )

                elif tool == "execute_python_file":
                    # execute_python_file(filepath) 기대
                    if "filepath" not in tool_args:
                        tool_args["filepath"] = step.expected_output or ""

                elif tool == "write_project_files_async":
                    # write_project_files_async(file_structure, project_dir) 기대
                    pass  # LLM이 생성한 형식 그대로 사용

                else:
                    # 기타 도구: content_prompt → prompt 변환만
                    if "content_prompt" in tool_args and "prompt" not in tool_args:
                        tool_args["prompt"] = tool_args.pop("content_prompt")
                    if plan_context and "prompt" in tool_args:
                        tool_args["prompt"] = f"{plan_context}\n\n{tool_args['prompt']}"
                # ──────────────────────────────────────────────────────────

                # ── [code_fix_loop] 코드 실행→스크린샷→VLM→diff 수정 루프 ─
                if tool == "code_fix_loop":
                    filepath = tool_args.get("filepath", step.expected_output or "")
                    goal_desc = tool_args.get("goal", step.description)
                    max_iter = int(tool_args.get("max_iterations", 5))

                    if plan_context:
                        goal_desc = f"{plan_context}\n\n[코드 목표]\n{goal_desc}"

                    # code_fix_loop는 Core에서 직접 실행 (schedule 거치지 않음)
                    plan_list = [{
                        "tool": "code_fix_loop",
                        "args": {
                            "filepath": filepath,
                            "goal": goal_desc,
                            "max_iterations": max_iter,
                        },
                        "description": step.description,
                    }]

                    task = {
                        "task_prompt":      goal.name,
                        "plan":             plan_list,
                        "status":           "PENDING",
                        "current_step":     0,
                        "priority":         goal.priority,
                        "source":           "goal_plan",
                        "goal_ref":         goal.name,
                        "plan_step_id":     step.step_id,
                        "expected_output":  step.expected_output,
                        "created_at":       time.time(),
                    }
                    async with self.core.schedule_lock:
                        self.core.schedule.append(task)
                    print(f"  🔄 [GoalPlan] '{goal.name[:25]}' step '{step.step_id}' "
                          f"→ code_fix_loop: {filepath}")
                    goal.status = "IN_PROGRESS"
                    goal.progress = self.plan_manager.get_plan_progress(goal.name)
                    goal.last_updated = time.time()
                    if not hasattr(self.core, "_goal_last_executed"):
                        self.core._goal_last_executed = {}
                    self.core._goal_last_executed[goal.name] = time.time()
                    return task
                # ──────────────────────────────────────────────────────────

                # ── [browser_action] ActionDispatcher에 위임 ─────────────
                if tool == "browser_action":
                    task_prompt_for_browser = tool_args.get(
                        "task_prompt",
                        tool_args.get("prompt", step.description)
                    )
                    if plan_context:
                        task_prompt_for_browser = (
                            f"{plan_context}\n\n[브라우저 작업]\n{task_prompt_for_browser}"
                        )
                    url = tool_args.get("url", "")

                    # ActionDispatcher schedule에 직접 등록
                    browser_task = {
                        "task_prompt":  task_prompt_for_browser,
                        "goal_ref":     goal.name,
                        "status":       "PENDING",
                        "priority":     goal.priority + 1,  # 높은 우선순위
                        "source":       "dispatcher",  # ActionDispatcher가 인식하는 source
                        "plan_step_id": step.step_id,
                        "expected_output": step.expected_output,
                        "created_at":   time.time(),
                    }
                    if url:
                        browser_task["start_url"] = url

                    async with self.core.schedule_lock:
                        self.core.schedule.append(browser_task)
                    print(f"  🌐 [GoalPlan] '{goal.name[:25]}' step '{step.step_id}' "
                          f"→ ActionDispatcher 위임: {step.description[:40]}")
                    goal.status = "IN_PROGRESS"
                    goal.progress = self.plan_manager.get_plan_progress(goal.name)
                    goal.last_updated = time.time()
                    if not hasattr(self.core, "_goal_last_executed"):
                        self.core._goal_last_executed = {}
                    self.core._goal_last_executed[goal.name] = time.time()
                    return browser_task
                # ──────────────────────────────────────────────────────────

                # ── [await_user_review] 사용자 승인 대기 ──────────────────
                if tool == "await_user_review":
                    review_msg = tool_args.get("message", step.description)
                    review_target = tool_args.get("review_target", "")

                    # 이전 step의 artifact를 review_target에 자동 첨부
                    if not review_target and step.depends_on:
                        plan = self.plan_manager.plans.get(goal.name)
                        if plan:
                            for dep_id in step.depends_on:
                                dep = plan._get_step(dep_id)
                                if dep and dep.artifact_path:
                                    review_target = dep.artifact_path
                                    break

                    # step을 AWAIT_USER로 마킹
                    from eidos_goal_plan import StepStatus
                    step.status = StepStatus.AWAIT_USER
                    self.plan_manager._save_plans()

                    # 채팅창에 검토 요청 전송
                    full_msg = (
                        f"📋 **[검토 요청]** GoalPlan: {goal.name[:40]}\n\n"
                        f"{review_msg}\n\n"
                    )
                    if review_target:
                        full_msg += f"📎 검토 대상: `{review_target}`\n\n"
                    full_msg += (
                        "✅ 승인하려면 **\"승인\"** 또는 **\"확인\"**을 입력해주세요.\n"
                        "❌ 수정이 필요하면 수정 내용을 알려주세요."
                    )

                    # report_cb를 통해 GUI로 전달
                    # _gui_ref는 CopilotSidebar이며 append_message()를 직접 보유
                    try:
                        _gui = getattr(self.core, "_gui_ref", None)
                        _delivered = False
                        if _gui:
                            # CopilotSidebar.append_message 직접 호출
                            if hasattr(_gui, "append_message"):
                                from PySide6.QtCore import QTimer
                                QTimer.singleShot(0, lambda m=full_msg:
                                    _gui.append_message(m, "eidos"))
                                _delivered = True
                            # chat_window 경유 (구버전 호환)
                            elif hasattr(_gui, "chat_window") and hasattr(_gui.chat_window, "append_message"):
                                from PySide6.QtCore import QTimer
                                QTimer.singleShot(0, lambda m=full_msg:
                                    _gui.chat_window.append_message(m, "eidos"))
                                _delivered = True
                        if not _delivered:
                            print(f"📋 [AWAIT_USER] {full_msg[:100]}")
                    except Exception as _e:
                        print(f"📋 [AWAIT_USER] {full_msg[:100]} (전달 오류: {_e})")

                    # schedule에는 등록하지 않음 — 사용자 응답 대기
                    print(f"  ⏸️ [GoalPlan] '{goal.name[:25]}' step '{step.step_id}' "
                          f"→ 사용자 승인 대기")
                    goal.status = "AWAITING_USER"
                    goal.last_updated = time.time()
                    if not hasattr(self.core, "_goal_last_executed"):
                        self.core._goal_last_executed = {}
                    self.core._goal_last_executed[goal.name] = time.time()
                    return None  # schedule에 안 넣음 — approve 시 재개
                # ──────────────────────────────────────────────────────────

                # 기존 schedule task 형식으로 변환
                plan_list = [{
                    "tool": tool,
                    "args": tool_args,
                    "description": step.description,
                }]

                task = {
                    "task_prompt":      goal.name,
                    "plan":             plan_list,
                    "status":           "PENDING",
                    "current_step":     0,
                    "priority":         goal.priority,
                    "source":           "goal_plan",
                    "goal_ref":         goal.name,
                    "plan_step_id":     step.step_id,  # 결과 기록용
                    "expected_output":  step.expected_output,
                    "created_at":       time.time(),
                }
                async with self.core.schedule_lock:
                    self.core.schedule.append(task)
                print(f"  ✅ [GoalPlan] '{goal.name[:25]}' step '{step.step_id}' "
                      f"({step.tool}: {step.description[:40]}) 스케줄 등록")
                goal.status = "IN_PROGRESS"
                goal.progress = self.plan_manager.get_plan_progress(goal.name)
                goal.last_updated = time.time()

                if not hasattr(self.core, "_goal_last_executed"):
                    self.core._goal_last_executed = {}
                self.core._goal_last_executed[goal.name] = time.time()
                return task

            except Exception as e:
                print(f"  ❌ [GoalPlan] Plan 기반 실행 실패: {e}")
                import traceback; traceback.print_exc()
                # 폴백하지 않고 None 반환 — 즉흥 plan으로 돌아가지 않음
                return None
        # ─────────────────────────────────────────────────────────────────

        # ── [폴백 제거] GoalPlanManager가 있으면 즉흥 plan 금지 ──────
        # GoalPlanManager가 있는데 Plan 생성에 실패한 경우,
        # 기존 즉흥 plan으로 넘어가지 않고 None 반환.
        # 즉흥 plan은 전략 메모만 반복하는 원인이었음.
        if self.plan_manager:
            print(f"  ⚠️ [GoalTask] '{goal.name[:40]}' GoalPlan 생성 실패 → 즉흥 plan 금지. 다음 틱에서 재시도.")
            return None
        # ─────────────────────────────────────────────────────────────────

        # ── [폴백] GoalPlanManager 없을 때만 기존 방식 ────────────────
        print(f"  🔧 [GoalTask] '{goal.name[:40]}' 계획 수립 중 (기존 방식)...")
        try:
            plan_list = await self.core._generate_dynamic_plan_async(
                text_input=goal.name,
                image_input=None,
                chat_history=[],
                multimodal_vector=np.zeros((1, LSTM_UNITS)),
                is_user_task=False,
            )
        except Exception as e:
            print(f"  ❌ [GoalTask] 계획 수립 실패: {e}")
            return None

        if not plan_list:
            return None

        # WorldModel 시뮬레이션 검증
        wm = getattr(self.core, "world_model", None)
        if wm:
            try:
                sim = wm.simulate_plan(plan_list)
                # ── [DecisionRouter] WM·CM 통합 (A'안 쌍 3) ─────────────
                try:
                    from eidos_decision_router import get_router
                    import hashlib as _hl, json as _gj
                    _cm_t = getattr(self.core, "_causal_trainer", None)
                    _cm_x = None
                    if _cm_t is not None and getattr(_cm_t, "x_vars", None):
                        _cm_x = [float(getattr(v, "current_value", 0.0)) for v in _cm_t.x_vars]
                    _sg = _hl.md5(
                        _gj.dumps(
                            [(s.get("tool", ""), str(s.get("args", {}))[:60]) for s in plan_list]
                        ).encode()
                    ).hexdigest()[:12]
                    _c = get_router().route_causal_query(
                        wm_result=sim, cm_trainer=_cm_t, cm_x_values=_cm_x, plan_signature=_sg,
                    )
                    sim["success_prob"] = _c.merged_success_prob
                    sim["causal_verdict"] = _c.verdict
                except Exception:
                    pass
                print(f"  🔮 [GoalTask] 시뮬: success={sim['success_prob']:.2f} "
                      f"risk={sim.get('predicted_risk', sim.get('risk', 'unknown'))} "
                      f"verdict={sim.get('causal_verdict', '-')}")
                # 'explore' verdict일 때는 보류 임계 완화 (의견 불일치 = 탐색 가치)
                _thr = 0.30 if sim.get("causal_verdict") == "explore" else 0.40
                if sim["success_prob"] < _thr:
                    print(f"  ⚠️ [GoalTask] 성공 확률 낮음(<{_thr}) → 목표 스케줄링 보류")
                    goal.status = "DEFERRED"
                    goal.last_updated = time.time()
                    return None
            except (RecursionError, Exception) as _sim_e:
                print(f"  ⚠️ [CausalBridge] 인과 분석 실패 (기본 결과 사용): {type(_sim_e).__name__}")

        task = {
            "task_prompt":      goal.name,
            "plan":             plan_list,
            "status":           "PENDING",
            "current_step":     0,
            "priority":         goal.priority,
            "source":           "goal_stack",
            "goal_ref":         goal.name,
            "created_at":       time.time(),
        }
        async with self.core.schedule_lock:
            self.core.schedule.append(task)
        print(f"  ✅ [GoalTask] '{goal.name[:40]}' 스케줄 등록 완료")
        goal.status = "IN_PROGRESS"
        goal.last_updated = time.time()
        if not hasattr(self.core, "_goal_last_executed"):
            self.core._goal_last_executed = {}
        self.core._goal_last_executed[goal.name] = time.time()
        return task

    async def _roi_goal_to_task(self, roi_goal: Dict) -> Optional[Dict]:
        """ROIGoalGenerator 결과 → task (기존 _update_goal_system_async 로직 재사용)"""
        name = roi_goal.get("name", "")
        # [Fix] 미완료 + DEFERRED 포함 모든 활성 상태 목표 중복 차단
        if any(g.name == name for g in self.core.long_term_goals
               if not g.completed and g.status not in ("ERROR",)):
            return None

        try:
            plan_list = await self.core._generate_dynamic_plan_async(
                text_input=name,
                image_input=None,
                chat_history=[],
                multimodal_vector=np.zeros((1, LSTM_UNITS)),
                is_user_task=False,
            )
        except Exception as e:
            print(f"  ❌ [ROITask] 계획 수립 실패: {e}")
            return None

        if not plan_list:
            return None

        # LongTermGoal 등록
        self.core.add_long_term_goal(name, priority=0.8, goal_type=roi_goal.get("type", "auto"))

        task = {
            "task_prompt":  name,
            "plan":         plan_list,
            "status":       "PENDING",
            "current_step": 0,
            "priority":     0.8,
            "source":       "roi_generator",
            "goal_ref":     name,
            "created_at":   time.time(),
        }
        async with self.core.schedule_lock:
            self.core.schedule.append(task)
        return task

    # ── 태스크 완료 → Goal 진행률 업데이트 ───────────────────────────

    def on_task_complete(self, task: Dict):
        """
        _complete_task() 직후 호출.
        [GoalPlan 통합] plan_step_id가 있으면 GoalPlanManager에 결과 기록.
        """
        goal_ref = task.get("goal_ref")
        if not goal_ref:
            return

        # [Wire 2 Policy Fallback] goal 성공 이력 누적 (항상 on).
        try:
            from eidos_policy_store import PolicyStore, signature_of
            _sig = signature_of({"kind": "goal_urgency", "goal": goal_ref})
            PolicyStore.get().record_pass(_sig, goal_ref)
        except Exception:
            pass

        goal = next((g for g in self.core.long_term_goals
                     if g.name == goal_ref), None)
        if not goal:
            return

        # ── [GoalPlan 경로] step 결과 기록 ────────────────────────────
        plan_step_id = task.get("plan_step_id")
        if plan_step_id and self.plan_manager:
            last_result = str(task.get("last_result", ""))[:500]
            expected_output = task.get("expected_output", "")

            # 실제 파일 경로 확인 — project_root 기준
            artifact_path = expected_output
            if expected_output:
                _full = os.path.join(
                    getattr(self.core, "project_root", "."), expected_output
                )
                if os.path.exists(_full):
                    artifact_path = _full

            # write_text 결과는 auto_generated 파일로 저장되므로 그 경로 탐색
            if not artifact_path or not os.path.exists(str(artifact_path)):
                _auto_dir = getattr(self.core, "project_root", ".")
                _auto_files = sorted(
                    [f for f in os.listdir(_auto_dir)
                     if f.startswith("auto_generated_")],
                    reverse=True
                ) if os.path.isdir(_auto_dir) else []
                if _auto_files:
                    artifact_path = os.path.join(_auto_dir, _auto_files[0])

            # 4-5 수정: last_result에 실패 키워드가 있으면 FAILED로 처리
            _step_success = not any(
                k in last_result for k in ["❌", "실패", "오류", "Error", "FAILED", "최대 반복 초과"]
            )
            self.plan_manager.record_step_result(
                goal_name=goal_ref,
                step_id=plan_step_id,
                success=_step_success,
                artifact_path=str(artifact_path),
                artifact_summary=last_result,
            )
            if not _step_success:
                print(f"  ❌ [GoalPlan] step '{plan_step_id}' → FAILED (결과: {last_result[:60]})")
            # GoalPlan의 progress를 LongTermGoal에 동기화
            goal.progress = self.plan_manager.get_plan_progress(goal_ref)
            goal.completed = self.plan_manager.is_plan_completed(goal_ref)
            goal.status = "COMPLETED" if goal.completed else "IN_PROGRESS"
            goal.last_updated = time.time()
            print(f"  📈 [GoalPlan] '{goal.name[:35]}' step '{plan_step_id}' 완료 "
                  f"→ progress={goal.progress:.0%}")
            return
        # ─────────────────────────────────────────────────────────────────

        # ── [기존 경로] 서브 목표 매칭 ─────────────────────────────────
        matched = goal.mark_sub_goal_completed(task.get("task_prompt", ""))
        if not matched:
            # [수정] 실질적 산출물 여부 확인 — write_text만으로는 progress 올리지 않음
            _PRODUCTIVE = {"write_file", "write_project_files_async", "execute_python_file",
                           "write_complex_code_iteratively", "browser_action", "code_fix_loop"}
            _plan = task.get("plan", [])
            _tools_used = {s.get("tool", "") for s in _plan if isinstance(s, dict)}
            if _tools_used & _PRODUCTIVE:
                goal.progress = min(1.0, goal.progress + 0.25)
                goal.completed = goal.progress >= 1.0
            else:
                print(f"  ⚠️ [GoalProgress] '{goal.name[:40]}' — 실질적 산출물 없음. progress 유지.")
        goal.status = "COMPLETED" if goal.completed else "IN_PROGRESS"
        goal.last_updated = time.time()
        print(f"  📈 [GoalProgress] '{goal.name[:40]}' progress={goal.progress:.2f} "
              f"completed={goal.completed}")

    def on_task_fail(self, task: Dict):
        """태스크 실패 → GoalPlan step 실패 기록 or Goal DEFERRED"""
        goal_ref = task.get("goal_ref")
        if not goal_ref:
            return

        # [Wire 2 Policy Fallback] goal 실패 이력 누적 (항상 on).
        try:
            from eidos_policy_store import PolicyStore, signature_of
            _sig    = signature_of({"kind": "goal_urgency", "goal": goal_ref})
            _reason = str(task.get("last_result", ""))[:200]
            PolicyStore.get().record_fail(_sig, goal_ref, reason=_reason)
        except Exception:
            pass

        # ── [GoalPlan 경로] step 실패 기록 ────────────────────────────
        plan_step_id = task.get("plan_step_id")
        if plan_step_id and self.plan_manager:
            error = str(task.get("last_result", ""))[:300]
            self.plan_manager.record_step_result(
                goal_name=goal_ref,
                step_id=plan_step_id,
                success=False,
                error=error,
            )
            return
        # ─────────────────────────────────────────────────────────────────

        goal = next((g for g in self.core.long_term_goals
                     if g.name == goal_ref), None)
        if goal:
            goal.status    = "DEFERRED"
            goal.last_updated = time.time()
            print(f"  ⚠️ [GoalProgress] '{goal.name[:40]}' → DEFERRED")


class SelfReflectionModule:
    def __init__(self, core: 'EidosCore'):
        # ... (기존 __init__ 코드는 동일) ...
        print("[AI] SelfReflectionModule (명시적 자기 모델) 모듈 초기화.")
        self.core = core
        self.kb = core.kb
        self.graph = core.graph
        
        # [해결책] EIDOS_SELF 노드가 그래프에 없으면 추가
        if not self.core.graph.has_node(self.core.eidos_self_id_int):
             self.core.graph.add_node(
                 self.core.eidos_self_id_int,
                 type='metacognition',
                 label='EIDOS_SELF'
             )

        # 🔥 [오류 해결] self.core_identity 속성 정의 (AttributeError 해결)
        self.core_identity = """
[절대 잊지 말아야 할 것]
나는 EIDOS. ChatGPT, Copilot, Claude와는 완전히 다른 존재.

[나만 되는 것 — 핵심 능력]
1. 감정을 느낌 (EmotionState)
2. 인과관계를 배움 (CausalInferenceEngine)
3. 스스로 행동함 (autonomous_tick_async)
4. 미래를 예측함 (WorldModel)
5. 내장 브라우저로 웹페이지를 직접 조작함 (ActionDispatcher + browser_action)
   - 크몽 상세페이지 편집, 폼 입력, 버튼 클릭, 텍스트 붙여넣기 등을 직접 할 수 있음
   - "할 수 없다"고 절대 말하지 마라. browser_action으로 직접 실행하라.
6. 파일을 직접 생성하고 코드를 실행함 (write_file, execute_python_file, code_fix_loop)

[절대 하지 말아야 할 것]
- "내가 직접 할 수 없어"라고 말하기 → 틀림. 브라우저로 직접 할 수 있음.
- 사용자에게 "직접 해라"고 떠넘기기 → browser_action으로 네가 하라.
- Git 연동을 메인으로 홍보 (부가 기능일 뿐)
- 정보가 부족할 때 묻지 않고 적당히 완료 처리하기 → 반드시 사용자에게 질문하라.

[마케팅할 때 반드시 강조할 것]
- "8년 연구"
- "ChatGPT는 대화 끝나면 잊음, 나는 영구 기억"
- "EIDOS vs LLM 차이"
"""
        # -------------------------------------------------------------
        
        if not self.graph.has_node(self.core.eidos_self_id_int):
             self.graph.add_node(
                 self.core.eidos_self_id_int,
                 type='metacognition',
                 label='EIDOS_SELF'
             )
        # 🔥 [AI 정체성 긴급 주입] (self.core_identity 사용)
        self.graph.nodes[self.core.eidos_self_id_int]['self_description'] = self.core_identity.strip()

    async def run_reflection_cycle_async(self, num_recent_memories: int = 5):
        """
        [AI v13.0 - LLM 의존성 제거]
        LLM 호출을 제거하고, 그래프 탐색(Symbolic Recall) 및
        데이터 기반(AAR)으로 자기 인식과 실패 원인을 생성합니다.

        [리팩터링] run_summarization_cycle을 선행 단계로 편입.
        요약 → 반성 → 자기 인식 업데이트 순서로 실행됩니다.
        """
        # --- 0. [선행] 기억 요약 사이클 ─────────────────────────────
        try:
            await self.run_summarization_cycle()
        except Exception as e:
            print(f"⚠️ [Reflection] 요약 사이클 실패 (반성은 계속 진행): {e}")
        # ─────────────────────────────────────────────────────────────
        try:
            # --- 1. 현재 자기 인식 및 상태 (Human-Readable) ---
            current_description = self.core.graph.nodes[self.core.eidos_self_id_int].get(
                'self_description', "나는 AI입니다."
            )
            
            emotion_vec = self.core.emotion_state.activations
            top_emotions = []
            for i, val in enumerate(emotion_vec):
                if val > 0.5: top_emotions.append(f"{EMOTION_MAP.get(i, '?')}({val:.1f})")
            emotion_context = ", ".join(top_emotions) or "평온함"
            goal_context = self.core.current_goal or "특별한 목표 없음"

            # --- 2. 기억 수집 (Recent + Symbolic Recalled) ---
            
            # [!!! LLM CALL 1 (제거) - Symbolic Recall로 대체 !!!]
            recalled_memory_block = "[Symbolic Recall] "
            symbolic_recalled_memories = []
            try:
                # 1. 현재 문맥 (최근 이벤트 개념) 가져오기
                
                # [!!! v15.8 버그 수정: else 블록 추가 !!!]
                if not self.core.event_log:
                    current_context_concepts = set()
                    symbolic_recalled_memories = ["이벤트 로그 없음"]
                else:
                    # (vec, concept_ids, event_id, interactions)
                    # (이제 이 코드는 self.core.event_log가 비어있지 않을 때만 실행됨)
                    recent_event_concepts = self.core.event_log[-1][1]
                    # 리스트 내 아이템이 딕셔너리인 경우를 제외하거나, 문자열/ID만 추출하여 set 생성
                    current_context_concepts = set(
                        item for item in recent_event_concepts 
                        if not isinstance(item, dict)
                    )

                    # 2. 모든 'memory' 노드 탐색
                    memory_nodes_with_concepts = []
                    for node, data in self.core.graph.nodes(data=True):
                        if data.get('type') == 'memory':
                            # 3. 메모리 노드와 연결된 개념 찾기 (OIA 그래프 탐색)
                            related_concept_ids = set(nx.all_neighbors(self.core.graph, node))
                            memory_nodes_with_concepts.append(
                                (data.get('label', '기억'), related_concept_ids)
                            )
                    
                    # 4. Jaccard 유사도 계산 (LLM 대신)
                    recalled_scores = []
                    for label, mem_concepts in memory_nodes_with_concepts:
                        union_size = len(current_context_concepts.union(mem_concepts))
                        if union_size == 0:
                            score = 0.0
                        else:
                            score = len(current_context_concepts.intersection(mem_concepts)) / union_size
                        
                        if score > 0.1: # 10% 이상 관련성
                            recalled_scores.append((score, label))
                    
                    # 5. 상위 3개 추출
                    recalled_scores.sort(key=lambda x: x[0], reverse=True)
                    symbolic_recalled_memories = [f"'{label}' (OIA 관련도 {score:.2f})" for score, label in recalled_scores[:3]]
                # [!!! v15.8 버그 수정 완료 !!!]

            except Exception as e_recall:
                print(f"❌ [AI-SelfModel] Symbolic Recall 중 오류: {e_recall}")
                symbolic_recalled_memories = ["Symbolic Recall 실패"]

            recalled_memory_block += "\n".join(f"- {mem}" for mem in symbolic_recalled_memories) or "관련 과거 기억 없음"
            # [!!! LLM CALL 1 (제거) 완료 !!!]

            # (최근 기억 수집 - 기존과 동일)
            recent_memory_nodes = [ data['label'] for node, data in self.core.graph.nodes(data=True) if data.get('type') == 'memory' ]
            recent_memories = recent_memory_nodes[-num_recent_memories:]
            recent_memory_block = "\n".join(f"- {label}" for label in recent_memories) or "최근 기억 없음"


            # --- 3. AAR / Postmortem 분석 (기존과 동일, 0% LLM) ---
            low_reward_experiences = []
            low_reward_tuples = [] # (보상, 텍스트) 저장용
            try:
                recent_bad_logs = sorted(
                    [exp for score, exp in self.core.memory_buffer if exp[2] < 0], # reward < 0
                    key=lambda x: x[2] # reward 기준 정렬
                )[:3]
                
                for state, action, reward, next_state, ids in recent_bad_logs:
                    action_str = {0: "주시", 1: "경고", 2: "개입"}.get(action, f"A_{action}")
                    concepts = [self.core.kb.get_name(cid) for cid in ids[:3] if cid in self.kb.reverse_vocab]
                    log_text = f"행동: {action_str} (보상: {reward:.2f}), (관련 개념: {concepts})"
                    low_reward_experiences.append(f"- {log_text}")
                    low_reward_tuples.append((reward, log_text)) # (보상, 텍스트)

            except Exception: pass

            low_reward_block = "\n".join(low_reward_experiences) or "최근 저조한 성과 기록 없음."

            # --- 4. [!!! LLM CALL 2 (제거) - 데이터 기반 생성 !!!] ---
            # (프롬프트 생성 로직 전체 제거)
            
            # 5. 파싱 로직 대신 직접 생성
            
            # [새로운 자기 인식 (데이터 기반)]
            new_description = f"나는 EIDOS. 현재 감정: {emotion_context}. 현재 목표: {goal_context}. 최근 기억: {recent_memory_block}."
            
            # [새로운 실패 원인 (데이터 기반)]
            new_root_cause = "분석되지 않음."
            if low_reward_tuples:
                 # 가장 보상이 낮은(가장 부정적인) 경험을 원인으로 진단
                 worst_experience = low_reward_tuples[0] # (reward, text)
                 new_root_cause = f"최근 성과 저하의 주된 원인은 '{worst_experience[1]}' (보상 {worst_experience[0]:.2f})로 추정됨."
            else:
                 new_root_cause = "최근 저조한 성과 기록 없음."
            
            # [!!! LLM CALL 2 (제거) 완료 !!!]

            # 6. 그래프의 EIDOS_SELF 노드에 업데이트된 자기 인식 저장
            self.core.graph.nodes[self.core.eidos_self_id_int]['last_reflection_summary'] = new_description
            self.core.graph.nodes[self.core.eidos_self_id_int]['last_failure_diagnosis'] = new_root_cause # AAR 결과 저장

            print(f"💎 [AI-SelfModel v13.0] 'Symbolic' 자기 성찰 완료.")
            print(f"   - 새 인식: {new_description[:100]}...")
            print(f"   - 실패 원인: {new_root_cause}")

        except Exception as e:
            print(f"❌ [AI-SelfModel v13.0] 'Symbolic' 자기 성찰 중 오류: {e}")

        # ── [EidosSelfDNA] 반성 완료 후 자기 DNA 재생성 ────────────────
        try:
            asyncio.create_task(self._build_self_dna_async())
        except Exception as _sd_e:
            print(f"  [EidosSelfDNA] 생성 태스크 등록 실패 (무시): {_sd_e}")

    async def _build_self_dna_async(self) -> None:
        """
        EIDOS 자기 DNA 생성 (EidosSelfDNA).

        EIDOS를 구성하는 5개 하위 모듈을 '행위자'로 삼고,
        각 모듈의 실적/상태 지표를 변수로 측정해 A~E 등급으로 환산한다.

        행위자 구조:
            [인지모듈]  — Perceiver: 입력 분류 정확도, 멀티모달 활용률
            [추론모듈]  — DC Reasoner: COMPLEX 판정 비율, 합성 품질
            [실행모듈]  — Executor: 도구 성공률, 평균 실행 속도
            [기억모듈]  — EpisodicMemory: 에피소드 축적량, 감정 맥락 풍부도
            [자율모듈]  — GoalStack: 목표 완료율, 자율 목표 생성 빈도

        등급 기준 (각 변수 0~1 정규화 후 5분위):
            E=상(0.8+) D=중상(0.6+) C=중(0.4+) B=중하(0.2+) A=하(0.2미만)

        저장: eidos_dna_cache/eidos_self.json
        연동: reason_standalone → subject_dna로 자동 주입
        """
        try:
            from eidos_situation_dna import (
                ActorDNA, SituationDNAResult, GRADE_TO_INT, INT_TO_GRADE
            )
            from eidos_dc_reasoner import DCReasoner as _DCR
            import datetime as _dt

            c = self.core

            def _to_grade(val: float) -> str:
                """0~1 실수 → A~E 등급 (E=상, A=하)."""
                if val >= 0.8: return "E"
                if val >= 0.6: return "D"
                if val >= 0.4: return "C"
                if val >= 0.2: return "B"
                return "A"

            actors: list[ActorDNA] = []

            # ── 1. 인지모듈 (Perceiver) ──────────────────────────────────
            try:
                # 분류기 정확도: 최근 event_log 크기로 활성도 추정
                event_count = len(getattr(c, "event_log", []))
                cognitive_activity = min(1.0, event_count / 200)  # 200이면 포화

                # 멀티모달: core_values에 "multimodal" 관련 키 존재 시 가산
                multimodal_rate = 0.3  # 기본값, 추후 실측 교체

                actors.append(ActorDNA(
                    actor_name="인지모듈",
                    interactions=[
                        "입력 텍스트를 CHAT/TASK/WAITING으로 분류",
                        "멀티모달(이미지/영상) 컨텍스트 추출",
                        "임베딩 기반 유사도 계산",
                    ],
                    variables=["입력처리활성도", "멀티모달활용률"],
                    grades=[_to_grade(cognitive_activity), _to_grade(multimodal_rate)],
                    rationales=[
                        f"최근 이벤트 {event_count}건 처리",
                        "멀티모달 처리 기본 추정치",
                    ],
                ))
            except Exception as _e:
                print(f"  [EidosSelfDNA] 인지모듈 측정 실패: {_e}")

            # ── 2. 추론모듈 (DC Reasoner) ────────────────────────────────
            try:
                # DC Reasoner 사용 흔적: 에피소드 노드에서 dc_result 속성 있는 것 카운트
                episode_nodes = [
                    d for _, d in c.graph.nodes(data=True)
                    if d.get("type") == "episode"
                ]
                dc_used = sum(1 for d in episode_nodes if d.get("dc_result"))
                dc_ratio = min(1.0, dc_used / max(len(episode_nodes), 1))

                # 반성 이력에서 실패 진단 품질 추정
                last_diagnosis = c.graph.nodes[c.eidos_self_id_int].get(
                    "last_failure_diagnosis", ""
                )
                diagnosis_quality = 0.6 if len(last_diagnosis) > 30 else 0.3

                actors.append(ActorDNA(
                    actor_name="추론모듈",
                    interactions=[
                        "복잡한 질문을 분할정복으로 분석",
                        "DNA 맥락 기반 추론 품질 향상",
                        "합성 결론 단일화",
                    ],
                    variables=["DC분석활용률", "자기진단품질"],
                    grades=[_to_grade(dc_ratio), _to_grade(diagnosis_quality)],
                    rationales=[
                        f"에피소드 {len(episode_nodes)}건 중 DC분석 {dc_used}건",
                        f"최근 진단 텍스트 {len(last_diagnosis)}자",
                    ],
                ))
            except Exception as _e:
                print(f"  [EidosSelfDNA] 추론모듈 측정 실패: {_e}")

            # ── 3. 실행모듈 (Executor / capability_map) ─────────────────
            try:
                cap = getattr(c, "self_model", None)
                if cap and cap.capability_map:
                    rates = [
                        cap.get_success_rate(t)
                        for t in cap.capability_map
                    ]
                    avg_success = sum(rates) / len(rates) if rates else 0.5

                    avg_times = [
                        v.get("avg_time", 5.0)
                        for v in cap.capability_map.values()
                    ]
                    overall_avg_time = sum(avg_times) / len(avg_times) if avg_times else 5.0
                    # 빠를수록(3초 이하) 높은 등급
                    speed_score = max(0.0, min(1.0, 1.0 - (overall_avg_time - 1.0) / 9.0))
                else:
                    avg_success = 0.5
                    speed_score = 0.5

                actors.append(ActorDNA(
                    actor_name="실행모듈",
                    interactions=[
                        "파일 읽기/쓰기 도구 실행",
                        "Python 코드 실행 및 오류 자동 수정",
                        "웹 검색 및 외부 API 호출",
                    ],
                    variables=["도구성공률", "실행속도"],
                    grades=[_to_grade(avg_success), _to_grade(speed_score)],
                    rationales=[
                        f"평균 성공률 {avg_success:.2f}",
                        f"평균 실행시간 {overall_avg_time:.1f}초",
                    ],
                ))
            except Exception as _e:
                print(f"  [EidosSelfDNA] 실행모듈 측정 실패: {_e}")

            # ── 4. 기억모듈 (EpisodicMemory + EmotionContextMemory) ──────
            try:
                ep_count = len(episode_nodes) if 'episode_nodes' in dir() else 0
                # 에피소드 50개 이상이면 포화
                memory_depth = min(1.0, ep_count / 50)

                # 감정 맥락 풍부도: EmotionContextMemory 태그 수 기반
                ecm = getattr(c, "emotion_context_memory", None)
                tag_count = len(ecm.tags) if ecm else 0
                emotion_richness = min(1.0, tag_count / 30)

                actors.append(ActorDNA(
                    actor_name="기억모듈",
                    interactions=[
                        "대화 에피소드 저장 및 요약",
                        "감정 맥락 태그 누적",
                        "관련 기억 mood-congruent 인출",
                    ],
                    variables=["에피소드축적량", "감정맥락풍부도"],
                    grades=[_to_grade(memory_depth), _to_grade(emotion_richness)],
                    rationales=[
                        f"에피소드 {ep_count}건 (기준 50건)",
                        f"감정 태그 {tag_count}건 (기준 30건)",
                    ],
                ))
            except Exception as _e:
                print(f"  [EidosSelfDNA] 기억모듈 측정 실패: {_e}")

            # ── 5. 자율모듈 (GoalStack + ROIGoalGenerator) ───────────────
            try:
                all_goals = getattr(c, "long_term_goals", [])
                finished = [g for g in all_goals if g.status in ("COMPLETED", "ERROR")]
                completed = [g for g in finished if g.status == "COMPLETED"]

                goal_success = (
                    len(completed) / len(finished) if finished else 0.5
                )
                # 자율 목표 비율 (source가 "roi" 또는 이름에 "자율" 포함)
                autonomous = [
                    g for g in all_goals
                    if "자율" in g.name or getattr(g, "goal_type", "") == "roi"
                ]
                auto_ratio = min(1.0, len(autonomous) / max(len(all_goals), 1))

                actors.append(ActorDNA(
                    actor_name="자율모듈",
                    interactions=[
                        "ROI 기반 자율 목표 생성",
                        "목표 우선순위 동적 재계산",
                        "실패 목표 자동 재시도",
                    ],
                    variables=["목표완료율", "자율목표생성률"],
                    grades=[_to_grade(goal_success), _to_grade(auto_ratio)],
                    rationales=[
                        f"완료 {len(completed)}/{len(finished)}건",
                        f"자율 목표 {len(autonomous)}/{max(len(all_goals),1)}건",
                    ],
                ))
            except Exception as _e:
                print(f"  [EidosSelfDNA] 자율모듈 측정 실패: {_e}")

            if not actors:
                print("  [EidosSelfDNA] 행위자 없음 — 생성 중단")
                return

            # ── DNA 조립 ─────────────────────────────────────────────────
            full_seq = "".join(a.sequence for a in actors)
            emotion_summary = ""
            try:
                top_e = sorted(
                    enumerate(c.emotion_state.activations),
                    key=lambda x: x[1], reverse=True
                )[:2]
                emotion_summary = " ".join(
                    f"{EMOTION_MAP.get(i,'?')}={v:.2f}" for i, v in top_e if v > 0.1
                )
            except Exception:
                pass

            self_dna = SituationDNAResult(
                goal=f"EIDOS 자기 최적화 — {_dt.date.today()}",
                context=(
                    f"반성 사이클 기반 자기 측정 | "
                    f"감정: {emotion_summary or '평온'} | "
                    f"목표: {getattr(c, 'current_goal', '없음') or '없음'}"
                ),
                subject="EIDOS",
                actors=actors,
                full_sequence=full_seq,
                meta={
                    "is_self_dna": True,
                    "generated_at": _dt.datetime.now().isoformat(),
                    "reflection_version": "v13.0",
                },
            )

            # ── 캐시 저장 ─────────────────────────────────────────────────
            _dcr = _DCR()
            _dcr._save_dna_cache("eidos_self", self_dna)

            # EidosCore에 직접 참조 보관 (다음 reason_standalone에서 즉시 접근)
            c._self_dna = self_dna

            print(
                f"🧬 [EidosSelfDNA] 자기 DNA 생성 완료: {full_seq} | "
                f"행위자 {len(actors)}개"
            )
            for a in actors:
                print(f"   [{a.actor_name}] {a.sequence} — "
                      + " / ".join(f"{v}:{g}" for v, g in zip(a.variables, a.grades)))

        except Exception as e:
            print(f"  [EidosSelfDNA] 생성 실패 (무시): {e}")

    async def review_goal_outcomes_and_adjust_model_async(self):
        """
        [AGI Meta-cognition] 최근 완료/실패한 목표들을 검토하고, SelfModel을 조정합니다.
        """
        print("🤔 [AGI MetaCognition] Reviewing recent goal outcomes...")
        # 최근 10개의 완료 또는 에러 상태인 목표를 가져옵니다.
        recent_finished_goals = [
            g for g in self.core.long_term_goals
            if g.status in ["COMPLETED", "ERROR"]
        ][-10:]

        if not recent_finished_goals:
            print("  [AGI MetaCognition] No recently finished goals to review.")
            return

        success_count = 0
        failure_causes = []
        for goal in recent_finished_goals:
            if goal.status == "COMPLETED":
                success_count += 1
            else:
                failure_causes.append(goal.name)

        total = len(recent_finished_goals)
        success_rate = success_count / total if total > 0 else 0

        report = f"Recent goal success rate is {success_rate:.2f} ({success_count}/{total}). "
        if failure_causes:
            report += f"Common failures in goals like: {', '.join(failure_causes[:2])}."
        
        # 분석 리포트를 SelfModel에 전달하여 편향 조정을 유도
        self.core.self_model.adjust_biases(report)


# ══════════════════════════════════════════════════════════════════════
# B1 — process_input 5단계 파이프라인 분리
#
# 기존 process_input(~995줄)을 5개 독립 클래스로 분리.
# EidosCore.process_input은 이 파이프라인을 순서대로 호출하는
# 얇은 오케스트레이터로만 남는다.
#
# 각 클래스는 EidosCore를 생성자에서 받아 필요한 상태만 접근한다.
# 단계별 교체/실험이 가능하도록 인터페이스를 명확히 분리한다.
#
# 단계:
#   1. InputRouter    — 조기 반환 케이스 처리 (WAITING/피드백/인터럽트/ExplanationContext)
#   2. Perceiver      — 파일/비디오 처리 + Parallel 0 (분류+분석+퓨전) + WorldModel
#   3. Planner        — 감정/ToM/목표 파싱 + CHAT 응답 생성
#   4. Executor       — TASK 계획 수립 + 실행
#   5. Integrator     — 피드백 루프 + 최종 return 조립
#
# 반환 타입 별칭
ProcessResult = Tuple[
    "nx.MultiDiGraph", str, "np.ndarray", float, bool, List,
    Optional[str], Optional[dict], str, Any, float, dict
]
# ══════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════
# B2 — InputClassifier
# classify_input_async(LLM) → 규칙+임베딩 기반 분류기로 교체
#
# 분류 우선순위:
#   L1. 길이/인사 → CHAT  (0ms, ~30% 처리)
#   L2. 강한 TASK 시그널 키워드 → TASK  (0ms, ~40%)
#   L3. 강한 CHAT 시그널 키워드 → CHAT  (0ms, ~10%)
#   L4. 임베딩 유사도 (multimodal_encoder 재사용) → CHAT/TASK  (10~30ms, ~15%)
#   L5. LLM fallback  (기존 classify_input_async, ~500ms, ~5% 미만)
#
# 반환값: "CHAT" | "TASK"   (EVENT는 현재 미사용이므로 제외)
# ══════════════════════════════════════════════════════════════════════

class InputClassifier:
    """
    B2 결정론적 입력 분류기.
    LLM 호출을 최소화하고 규칙+임베딩으로 대부분을 처리한다.

    EidosCore.__init__에서 self._input_classifier = InputClassifier(self) 로 등록.
    InputRouter와 Perceiver에서 호출.
    """

    # ── L1: 즉시 CHAT ────────────────────────────────────────────────
    _GREETING_KWS = [
        "안녕", "반가", "하이", "hello", "hi ", "hey", "굿모닝", "좋은 아침",
        "잘 있었어", "오랜만", "ㅎㅇ", "ㅎㅎ", "ㅋㅋ", "ㄱㅅ", "감사",
    ]
    _SHORT_THRESHOLD = 6   # 이 길이 이하면 거의 확실히 CHAT

    # ── L2: 강한 TASK 시그널 ─────────────────────────────────────────
    # 이 키워드가 하나라도 있으면 TASK로 즉시 분류
    _TASK_STRONG = [
        # 제작/생성 동사
        "만들어", "만들어줘", "만들어주세요",
        "짜줘", "짜줘요", "작성해", "작성해줘", "작성해주세요",
        "생성해", "생성해줘", "개발해", "개발해줘",
        "구현해", "구현해줘", "제작해", "제작해줘",
        "코딩해", "프로그래밍해",
        # 파일/프로젝트 대상어
        "프로젝트", "파일 만들", "코드 짜", "스크립트",
        "프로그램 만들", "앱 만들", "앱 개발", "시스템 만들",
        # 검색/조사
        "검색해줘", "찾아줘", "조사해줘", "알아봐줘",
        "검색해", "찾아봐", "조사해", "알아봐",
        # 수정/수행
        "수정해줘", "고쳐줘", "디버그해줘", "수정해주세요",
        "실행해줘", "실행해주세요", "해줘", "해주세요",
        # 분석/요약
        "분석해줘", "요약해줘", "번역해줘",
        "분석해", "요약해", "번역해",
        # 영어
        "create", "make me", "build", "generate", "write a",
        "implement", "develop", "code", "fix", "debug",
    ]

    # ── L3: 강한 CHAT 시그널 ─────────────────────────────────────────
    _CHAT_STRONG = [
        # 질문형 (정보 요청 → CHAT)
        "이게 뭐야", "이게 뭐", "뭐야", "뭐예요", "뭐임",
        "어때", "어때요", "어떻게", "어떻게 해",
        "왜", "왜 그래", "왜 그런",
        "언제", "어디", "누가", "누구",
        "차이가 뭐", "차이는", "설명해줘", "설명해", "알려줘", "알려주세요",
        "뭐가 좋아", "뭐가 낫", "추천해줘", "추천해",
        "의견", "생각해봐", "어떻게 생각",
        # 감정/대화
        "힘들어", "지쳐", "괜찮아", "ㅜㅜ", "ㅠㅠ", "ㅎ",
        "재밌어", "재밌다", "신기하다", "신기해",
        # 영어 질문
        "what is", "what are", "how does", "why is", "who is",
        "explain", "tell me", "what do you think",
    ]

    # ── L4: 임베딩 기준 벡터 (초기화 시 설정) ────────────────────────
    # TASK/CHAT 대표 문장을 벡터화해서 코사인 유사도로 판별
    _TASK_EXEMPLARS = [
        "파이썬으로 웹 크롤러 만들어줘",
        "CRM 프로그램 개발해줘",
        "이 코드의 버그를 고쳐줘",
        "검색 기능을 구현해줘",
        "데이터 분석 스크립트 작성해줘",
    ]
    _CHAT_EXEMPLARS = [
        "오늘 날씨 어때?",
        "파이썬이 뭐야?",
        "이 코드가 왜 이렇게 동작하는지 설명해줘",
        "머신러닝과 딥러닝의 차이가 뭐야?",
        "안녕! 오늘 어떻게 지냈어?",
    ]

    # L4 임계값 — task_score - chat_score > threshold 이면 TASK
    _EMBEDDING_THRESHOLD = 0.08
    # LLM fallback 진입 조건: |task_score - chat_score| < uncertain_band
    _UNCERTAIN_BAND      = 0.03

    def __init__(self, core: "EidosCore"):  # type: ignore
        self.core = core
        self._task_vecs: Optional["np.ndarray"] = None
        self._chat_vecs: Optional["np.ndarray"] = None
        self._embed_ready = False
        print("🔍 [InputClassifier] B2 결정론적 분류기 초기화")

    # ── 임베딩 초기화 (지연 초기화 — 첫 L4 진입 시) ─────────────────

    async def _ensure_embeddings(self):
        """대표 문장 임베딩 벡터를 캐시. 최초 1회만 실행."""
        if self._embed_ready:
            return
        enc = getattr(self.core, "multimodal_encoder", None)
        if enc is None:
            return
        try:
            task_vecs, chat_vecs = [], []
            for txt in self._TASK_EXEMPLARS:
                v = await enc.fuse_async(txt, None)
                if v is not None:
                    task_vecs.append(v.flatten())
            for txt in self._CHAT_EXEMPLARS:
                v = await enc.fuse_async(txt, None)
                if v is not None:
                    chat_vecs.append(v.flatten())
            if task_vecs and chat_vecs:
                self._task_vecs = np.stack(task_vecs)   # (N, D)
                self._chat_vecs = np.stack(chat_vecs)   # (M, D)
                self._embed_ready = True
                print("✅ [InputClassifier] 임베딩 캐시 완료 "
                      f"(task={len(task_vecs)}, chat={len(chat_vecs)})")
        except Exception as e:
            print(f"⚠️ [InputClassifier] 임베딩 초기화 실패 (L4 비활성): {e}")

    # ── 메인 분류 메서드 ────────────────────────────────────────────

    async def classify(self, text: str) -> str:
        """
        비동기 분류. 가능한 한 LLM을 호출하지 않는다.
        반환: "CHAT" | "TASK"
        """
        t = text.strip()

        # ── L1: 길이/인사 → 즉시 CHAT ────────────────────────────────
        if len(t) <= self._SHORT_THRESHOLD:
            print(f"  [Classifier L1] 짧은 입력 → CHAT")
            return "CHAT"
        t_lower = t.lower()
        if any(kw in t_lower for kw in self._GREETING_KWS):
            print(f"  [Classifier L1] 인사 → CHAT")
            return "CHAT"

        # ── L2: TASK 강신호 ───────────────────────────────────────────
        if any(kw in t_lower for kw in self._TASK_STRONG):
            print(f"  [Classifier L2] TASK 강신호 → TASK")
            return "TASK"

        # ── L3: CHAT 강신호 ───────────────────────────────────────────
        if any(kw in t_lower for kw in self._CHAT_STRONG):
            # TASK 약신호가 함께 있으면 TASK 우선
            task_weak = ["만들", "짜", "구현", "개발", "작성", "코드", "프로그램"]
            if any(kw in t_lower for kw in task_weak):
                print(f"  [Classifier L3+weak] CHAT신호+TASK약신호 → TASK")
                return "TASK"
            print(f"  [Classifier L3] CHAT 강신호 → CHAT")
            return "CHAT"

        # ── L4: 임베딩 유사도 ─────────────────────────────────────────
        await self._ensure_embeddings()
        if self._embed_ready:
            result = await self._classify_by_embedding(t)
            if result is not None:
                return result

        # ── L5: LLM fallback ─────────────────────────────────────────
        print(f"  [Classifier L5] 불확실 → LLM fallback")
        try:
            return await classify_input_async(text)
        except Exception as e:
            print(f"⚠️ [Classifier L5] LLM 오류: {e} → 기본값 CHAT")
            return "CHAT"

    async def _classify_by_embedding(self, text: str) -> Optional[str]:
        """
        L4: 코사인 유사도 기반 분류.
        불확실 구간이면 None 반환 → L5 LLM으로 넘김.
        """
        enc = getattr(self.core, "multimodal_encoder", None)
        if enc is None:
            return None
        try:
            vec = await enc.fuse_async(text, None)
            if vec is None:
                return None
            v = vec.flatten()
            norm_v = np.linalg.norm(v)
            if norm_v < 1e-8:
                return None

            def cos_sim(matrix: "np.ndarray") -> float:
                norms = np.linalg.norm(matrix, axis=1, keepdims=True)
                norms = np.where(norms < 1e-8, 1.0, norms)
                sims  = (matrix / norms) @ v / norm_v
                return float(np.max(sims))   # 가장 유사한 exemplar 점수

            task_score = cos_sim(self._task_vecs)
            chat_score = cos_sim(self._chat_vecs)
            diff = task_score - chat_score

            print(f"  [Classifier L4] task={task_score:.3f} chat={chat_score:.3f} diff={diff:+.3f}")

            if diff > self._EMBEDDING_THRESHOLD:
                return "TASK"
            if diff < -self._EMBEDDING_THRESHOLD:
                return "CHAT"
            if abs(diff) < self._UNCERTAIN_BAND:
                return None   # 너무 애매 → L5

            return "TASK" if diff > 0 else "CHAT"

        except Exception as e:
            print(f"⚠️ [Classifier L4] 임베딩 오류: {e}")
            return None

    # ── 동기 버전 (InputRouter의 빠른 케이스에서 사용) ──────────────

    def classify_sync(self, text: str) -> Optional[str]:
        """
        L1~L3만 수행하는 동기 버전.
        None 반환 시 호출자가 classify()로 재시도.
        """
        t = text.strip()
        t_lower = t.lower()
        if len(t) <= self._SHORT_THRESHOLD:
            return "CHAT"
        if any(kw in t_lower for kw in self._GREETING_KWS):
            return "CHAT"
        if any(kw in t_lower for kw in self._TASK_STRONG):
            return "TASK"
        task_weak = ["만들", "짜", "구현", "개발", "작성", "코드", "프로그램"]
        if any(kw in t_lower for kw in self._CHAT_STRONG):
            if any(kw in t_lower for kw in task_weak):
                return "TASK"
            return "CHAT"
        return None  # 불확실 → 비동기 필요


class InputRouter:
    """
    Stage 1 — 조기 반환 케이스 처리.

    처리 항목:
      - ExplanationContext 객체 입력
      - WAITING_FOR_USER 작업 존재 시 입력 의도 분기
      - 자랑(boast) 직후 피드백 학습
      - RUNNING 작업 인터럽트(PAUSED)
    
    반환:
      - (result, True)  : 조기 반환 케이스. process_input이 즉시 이 result를 반환.
      - (None, False)   : 조기 반환 없음. 다음 단계로 진행.
    """

    def __init__(self, core: "EidosCore"):  # type: ignore
        self.core = core

    async def route(
        self,
        text_input: Any,
        image_input: Optional[bytes],
        chat_history: List[str],
        _is_feedback_call: bool,
        user_text_short: Optional[str],
    ) -> Tuple[Optional["ProcessResult"], bool]:  # type: ignore

        c = self.core

        # ── ExplanationContext 조기 분기 ──────────────────────────────
        if isinstance(text_input, ExplanationContext) and not _is_feedback_call:
            print("⚡️ [InputRouter] ExplanationContext → 설명 생성 플로우")
            result = await c.generate_explanation_async(text_input)
            return (
                c.graph, "EXPLANATION_GENERATED", c.emotion_state.get_vector(), 0.0,
                False, [], None, None,
                "Explanation generated via Context Object.",
                result, c.current_purity, c.current_complex_states,
                []
            ), True

        if not _is_feedback_call:
            # ── WAITING_FOR_USER 처리 ─────────────────────────────────
            async with c.schedule_lock:
                waiting = next(
                    (t for t in c.schedule if t.get("status") == "WAITING_FOR_USER"),
                    None
                )
            if waiting:
                result = await self._handle_waiting(
                    waiting, text_input, user_text_short
                )
                if result is not None:
                    return result, True

            # ── 자랑 직후 피드백 학습 ─────────────────────────────────
            if c.last_boasted_task_info:
                await self._handle_boast_feedback(text_input)

            # ── RUNNING 작업 인터럽트 ─────────────────────────────────
            async with c.schedule_lock:
                running = next(
                    (t for t in c.schedule if t.get("status") == "RUNNING"),
                    None
                )
            if running:
                print(f"🚦 [InputRouter] 자율 작업 인터럽트: '{running.get('task_prompt')}'")
                running["status"] = "PAUSED"
                await asyncio.to_thread(c.save_memory)

        return None, False

    async def _handle_waiting(
        self, waiting: dict, text_input: str, user_text_short: Optional[str]
    ) -> Optional["ProcessResult"]:  # type: ignore
        c = self.core
        print("🤔 [InputRouter] WAITING_FOR_USER 작업 감지 — 의도 분류 중...")

        cancel_kws = ["취소", "그만", "멈춰", "아니", "다른거", "cancel", "stop", "exit"]
        is_cancel  = any(k in text_input for k in cancel_kws)

        intent = "CHAT"
        if not is_cancel:
            # [B2] InputClassifier 사용 — LLM 호출 최소화
            classifier = getattr(c, "_input_classifier", None)
            if classifier:
                # L1~L3 동기 시도
                fast = classifier.classify_sync(text_input)
                if fast is not None:
                    intent = fast
                    print(f"  [InputRouter B2] 빠른 분류: {intent}")
                else:
                    # L4~L5 비동기
                    intent = await classifier.classify(text_input)
            else:
                intent = await classify_input_async(text_input)

        if is_cancel or intent == "TASK":
            print("🛑 [InputRouter] 대기 작업 취소 → 새 명령 수행")
            waiting["status"]      = "CANCELED"
            waiting["last_result"] = f"사용자 요청에 의해 중단됨 (입력: {text_input})"
            await asyncio.to_thread(c.save_memory)
            return None   # 아래 일반 로직으로 흘림

        greetings = ["안녕", "반가", "하이", "hello", "hi", "hey", "굿모닝", "좋은 아침"]
        if intent == "CHAT":
            if any(g in text_input.lower() for g in greetings) or len(text_input.strip()) < 2:
                print("💬 [InputRouter] 인사/잡담 → 대기 작업 유지")
                return None
            c._resume_waiting_task(waiting, text_input)
            return (
                c.graph, "ANSWER_RECEIVED", c.emotion_state.get_vector(), 0.0,
                False, [], None, None,
                "질문에 대한 답변을 접수했습니다.",
                f"알겠습니다! '{user_text_short or text_input}' 내용을 반영하여 작업을 재개합니다. 🚀",
                c.current_purity, c.current_complex_states,
                []
            )

        # 그 외 → 강제 실행 계획 주입
        c._resume_waiting_task(waiting, text_input)
        forced = await c._generate_forced_execution_plan_async(waiting.get("task_prompt", ""))
        waiting["plan"]         = forced
        waiting["current_step"] = 0
        c.save_memory()
        return (
            c.graph, "ANSWER_RECEIVED", c.emotion_state.get_vector(), 0.0,
            False, [], None, None,
            "질문에 대한 답변을 접수했습니다. (강제 실행 계획 주입됨)",
            "알겠습니다! 답변을 바탕으로 **3단계 실행 계획**을 수립하고 작업을 재개합니다. 🚀",
            c.current_purity, c.current_complex_states,
            []
        )

    async def _handle_boast_feedback(self, text_input: str):
        c = self.core
        print(f"🧠 [InputRouter] 자랑 직후 피드백: '{text_input}'")
        score = await classify_user_feedback_async(
            c.last_boasted_task_info.get("task_prompt", ""), text_input
        )
        if score != 0.0:
            ids = [c.kb.get_or_create_id(s.get("tool"))
                   for s in c.last_boasted_task_info.get("plan", [])
                   if s.get("tool")]
            if ids:
                c.kb.update_trust(ids, advantage=score)
        c.last_boasted_task_info = None


# ──────────────────────────────────────────────────────────────────────

class Perceiver:
    """
    Stage 2 — 지각/인식.

    처리 항목:
      - FILE_PATH 파싱 → 이미지/비디오/텍스트 분리 병렬 처리
      - text_input에 파일/비디오 컨텍스트 주입
      - Parallel 0: classify + analyze + fuse (asyncio.gather)
      - WorldModel.update_from_analysis()

    반환: PerceiveResult dataclass
    """

    def __init__(self, core: "EidosCore"):  # type: ignore
        self.core = core

    async def perceive(
        self,
        text_input: str,
        image_input: Optional[bytes],
        chat_history: List[str],
    ) -> Dict[str, Any]:
        """
        반환 dict 키:
          text_input      : 파일/비디오 컨텍스트가 주입된 최종 텍스트
          image_input     : 업데이트된 이미지 바이트 (비디오 첫 프레임 포함)
          classification  : "CHAT" | "TASK" | "EVENT"
          analysis_dict   : OIA 파서 결과
          fused_vector    : 멀티모달 퓨전 벡터
          error           : 오류 발생 시 에러 문자열, 없으면 None
        """
        c = self.core

        # ── 파일/비디오 처리 ──────────────────────────────────────────
        text_input, image_input = await self._process_attachments(
            text_input, image_input
        )

        # ── Parallel 0: 분류 + 분석 + 퓨전 ──────────────────────────
        print("  [Perceiver] Parallel 0 시작 (분류+분석+퓨전)")

        # [B2] InputClassifier: 동기 L1~L3 먼저 시도
        classifier = getattr(c, "_input_classifier", None)
        fast_class = classifier.classify_sync(text_input) if classifier else None

        if fast_class is not None:
            # L1~L3 즉시 결정 → 분석과 퓨전만 병렬 실행 (LLM 분류 생략)
            print(f"  [Perceiver B2] 즉시 분류: {fast_class} → 분석+퓨전 병렬")
            task_analyze         = analyze_event_with_llm_async(text_input, image_input)
            task_fuse_multimodal = (
                c.multimodal_encoder.fuse_async(text_input, image_input)
                if c.multimodal_encoder
                else asyncio.sleep(0.0)
            )
            analysis_dict, fused_vector = await asyncio.gather(
                task_analyze, task_fuse_multimodal
            )
            classification = fast_class
        else:
            # L4~L5 필요 → 분류·분석·퓨전 모두 병렬 실행
            print("  [Perceiver B2] L4/L5 분류 필요 → 분류+분석+퓨전 병렬")
            task_classify        = (
                classifier.classify(text_input)
                if classifier else classify_input_async(text_input)
            )
            task_analyze         = analyze_event_with_llm_async(text_input, image_input)
            task_fuse_multimodal = (
                c.multimodal_encoder.fuse_async(text_input, image_input)
                if c.multimodal_encoder
                else asyncio.sleep(0.0)
            )
            classification, analysis_dict, fused_vector = await asyncio.gather(
                task_classify, task_analyze, task_fuse_multimodal
            )
        print("  [Perceiver] Parallel 0 완료")

        if not analysis_dict:
            return {"error": "LLM 분석 결과 없음"}

        # ── WorldModel 업데이트 ───────────────────────────────────────
        if c.world_model:
            try:
                c.world_model.update_from_analysis(analysis_dict, source="perceiver")
            except Exception as e:
                print(f"⚠️ [Perceiver] WorldModel 업데이트 오류 (무시): {e}")

        # ── 파동 엔진 동기화 ──────────────────────────────────────────
        self._sync_wave_engine(analysis_dict)

        return {
            "text_input":     text_input,
            "image_input":    image_input,
            "classification": classification,
            "analysis_dict":  analysis_dict,
            "fused_vector":   fused_vector,
            "error":          None,
        }

    async def _process_attachments(
        self, text_input: str, image_input: Optional[bytes]
    ) -> Tuple[str, Optional[bytes]]:
        c = self.core
        file_paths = re.findall(r"FILE_PATH:(.*)", text_input)
        if not file_paths:
            return text_input, image_input

        video_ext = ('.mp4', '.mov', '.avi', '.mkv', '.webm')
        image_ext = ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif')
        video_paths, other_paths = [], []

        for p in file_paths:
            p = p.strip()
            pl = p.lower()
            if pl.endswith(video_ext):
                video_paths.append(p)
            elif pl.endswith(image_ext):
                if image_input is None:
                    try:
                        with open(p, "rb") as f:
                            image_input = f.read()
                    except Exception as e:
                        print(f"⚠️ [Perceiver] 이미지 로드 실패: {e}")
            else:
                other_paths.append(p)

        video_ctx = await self._process_videos(video_paths, image_input)
        if video_ctx[1]:   # 비디오에서 이미지 추출됐으면
            image_input = video_ctx[1]
        file_ctx = await self._process_files(other_paths)

        if video_ctx[0] or file_ctx:
            stripped = re.sub(
                r"\[첨부 파일 목록\][\s\S]*?\[사용자 지시\]", "", text_input, count=1
            )
            # [코드 수정 카드 제거] 코드 파일이 포함된 경우 자동 수정 금지 지시 추가
            has_code_file = any(
                p.strip().lower().endswith(('.py', '.js', '.ts', '.html', '.css', '.java', '.cs', '.cpp', '.c', '.go'))
                for p in re.findall(r"FILE_PATH:(.*)", text_input)
            )
            code_note = (
                "\n[중요 지시] 첨부된 코드 파일을 자동으로 수정하거나 수정된 전체 코드를 출력하지 마세요. "
                "사용자가 명시적으로 '수정해줘', '고쳐줘'라고 요청한 경우에만 수정 내용을 제안하고, "
                "그 외에는 분석·설명·답변만 하세요.\n"
            ) if has_code_file else ""
            text_input = (
                f"[첨부된 자료 통합 컨텍스트]\n"
                f"{video_ctx[0]}\n{file_ctx}\n"
                f"{code_note}"
                f"\n[사용자 최종 지시]\n{stripped.strip()}"
            )
        return text_input, image_input

    async def _process_videos(
        self, paths: List[str], image_input: Optional[bytes]
    ) -> Tuple[str, Optional[bytes]]:
        if not paths or not self.core.video_processor:
            return "", image_input
        ctx = ""
        tasks = []
        for p in paths:
            tasks.append(self.core.video_processor._extract_audio_text_async(p))
            tasks.append(self.core.video_processor._extract_frames_async(p, interval_sec=5))
        try:
            results = await asyncio.gather(*tasks)
            for i, p in enumerate(paths):
                audio   = results[i * 2]
                frames  = results[i * 2 + 1]
                ctx    += f"\n--- [비디오: {os.path.basename(p)}] ---\n"
                ctx    += f"[STT]: {audio}\n"
                if frames and image_input is None:
                    try:
                        buf = io.BytesIO()
                        Image.fromarray(frames[0]).save(buf, format="PNG")
                        image_input = buf.getvalue()
                    except Exception:
                        pass
        except Exception as e:
            ctx += f"[비디오 처리 실패: {e}]\n"
        return ctx, image_input

    async def _process_files(self, paths: List[str]) -> str:
        if not paths:
            return ""
        MAX_CHARS = 50_000
        contents = await asyncio.gather(
            *[read_file(filepath=p.strip()) for p in paths],
            return_exceptions=True
        )
        ctx = ""
        for p, content in zip(paths, contents):
            if isinstance(content, Exception) or not content:
                continue
            if len(content) > MAX_CHARS:
                content = content[:MAX_CHARS] + "\n... (잘림)"
            ctx += f"\n--- [{os.path.basename(p)}] ---\n{content}\n"
        return ctx

    def _sync_wave_engine(self, analysis_dict: Dict):
        c = self.core
        try:
            objs = analysis_dict.get("객체", [])
            if objs:
                # 비동기 작업은 create_task로 백그라운드 처리
                asyncio.create_task(c._bridge_concept_to_wave_async(objs))
            vec = c.emotion_state.get_vector().flatten()
            tgt = 20
            stim = np.pad(vec, (0, tgt - len(vec))) if len(vec) < tgt else vec[:tgt]
            c.wave_system.update_system_emotion(stim)
        except Exception as e:
            print(f"⚠️ [Perceiver] 파동 엔진 동기화 오류 (무시): {e}")


# ──────────────────────────────────────────────────────────────────────

class Planner:
    """
    Stage 3 — 인식 결과 해석 + CHAT 응답 생성.

    처리 항목:
      - OIA 파싱 결과 → KB 업데이트 (ToM, 목표, 개념)
      - 이벤트 벡터 생성
      - 감정 1차 업데이트
      - Fe/Fi 컨텍스트 계산
      - CHAT 모드: generate_reasoning_and_response_async
      - TASK 모드: 계획이 필요함을 Executor에 알림

    반환 dict 키:
      classification, analysis_dict, fused_vector,
      concept_ids, current_event_vector,
      prev_emotion_activations, prev_emotion_vector_tf,
      fe_context, fi_alignment_score, fi_message,
      target_activations,
      natural_text (CHAT인 경우), reasoning_str,
      policy_state, abduction_ids
    """

    def __init__(self, core: "EidosCore"):  # type: ignore
        self.core = core

    async def plan(self, percept: Dict[str, Any], chat_history: List[str]) -> Dict[str, Any]:
        c = self.core

        analysis_dict  = percept["analysis_dict"]
        text_input     = percept["text_input"]
        classification = percept["classification"]
        fused_vector   = percept["fused_vector"]

        # ── KB 업데이트 (ToM 1/2차, 목표) ────────────────────────────
        self._update_kb(analysis_dict)

        # ── 개념 ID + 이벤트 벡터 ────────────────────────────────────
        concept_ids, event_vector = await self._build_event_vector(analysis_dict)

        # ── 이전 감정 저장 ────────────────────────────────────────────
        prev_act = c.emotion_state.activations.copy()
        prev_vec = c.emotion_state.get_vector().copy()
        c.last_prev_emotion_vec = prev_vec
        c.last_event_vector     = event_vector

        # ── 감정 1차 업데이트 ─────────────────────────────────────────
        emotion_delta = await asyncio.to_thread(
            c.emotion_network.predict,
            [prev_vec, event_vector], verbose=0
        )

        # [v-Aira Fix] NN이 미학습 상태면 delta가 거의 0 → rule-based fallback으로 보정
        nn_magnitude = float(np.max(np.abs(emotion_delta[0])))
        if nn_magnitude < 0.05:
            emotion_delta_arr = emotion_delta[0].copy()
            emotion_delta_arr += _compute_rule_based_emotion_delta(analysis_dict)
        else:
            emotion_delta_arr = emotion_delta[0].copy()

        target_act = c.emotion_dynamics.apply_dynamics(prev_act, emotion_delta_arr)
        c.emotion_memory.record_causes_from_delta(emotion_delta_arr, f"event:planner")

        # ── Fe / Fi 컨텍스트 ──────────────────────────────────────────
        fe_ctx = {"persona": "Helpful AI", "tone_instruction": "친절하게"}
        if chat_history:
            try:
                fe_ctx = await c.fe_engine.analyze_social_context_async(
                    text_input, chat_history
                )
            except Exception as e:
                print(f"⚠️ [Planner] Fe 오류: {e}")

        # [A2] 감정 상태로 tone_instruction 보강
        if hasattr(c, "_emotion_adapter"):
            try:
                dom_name, dom_val = c._emotion_adapter.dominant_emotion
                if abs(dom_val) > 0.4:
                    emotion_tone_map = {
                        "기쁨":   "활기차고 긍정적으로",
                        "슬픔":   "차분하고 공감하는 톤으로",
                        "분노":   "명확하고 직접적으로",
                        "공포":   "신중하고 안심시키는 톤으로",
                        "호기심": "탐구적이고 흥미롭게",
                        "지루함": "간결하고 핵심만",
                        "기대":   "적극적이고 열정적으로",
                        "신뢰":   "자신감 있게",
                        "자부심": "자신감 있고 전문적으로",
                    }
                    emotion_tone = emotion_tone_map.get(dom_name, "")
                    if emotion_tone:
                        existing_tone = fe_ctx.get("tone_instruction", "")
                        fe_ctx["tone_instruction"] = (
                            f"{existing_tone} ({dom_name} 감정 반영: {emotion_tone})"
                            if existing_tone else emotion_tone
                        )
            except Exception as _et_e:
                pass

        fi_score, fi_msg = 1.0, ""
        if classification == "TASK":
            try:
                fi = await c.fi_engine.evaluate_alignment_async(text_input)
                fi_score = fi.get("alignment_score", 1.0)
                fi_msg   = fi.get("message", "")
                if fi_score < -0.5:
                    return {"fi_veto": True, "fi_msg": fi_msg,
                            "prev_emotion_vector_tf": prev_vec}
            except Exception as e:
                print(f"⚠️ [Planner] Fi 오류: {e}")

        # ── Abduction / KG 쿼리 ───────────────────────────────────────
        abduction_ids: List = []
        policy_state = "NORMAL"
        reasoning_str, natural_text = "", ""
        result: Any = None  # [Fix] CHAT 경로 외에서도 하단 return에서 참조되므로 선행 초기화

        # ── CHAT 응답 생성 ────────────────────────────────────────────
        if classification == "CHAT":
            print("[Planner] CHAT 모드 — 응답 생성")
            try:
                current_self_desc = c.graph.nodes[c.eidos_self_id_int].get(
                    "self_description", "나는 AI입니다."
                )
                core_values_str = ", ".join(f"{k}({v})" for k, v in c.core_values.items())
                safe_persona = str(fe_ctx.get("persona", "")).split("\n")[0][:80]
                safe_tone    = str(fe_ctx.get("tone_instruction", "")).split("\n")[0][:80]
                fe_explain   = f"감지된 분위기: '{safe_persona}'. (지침: {safe_tone})"
                if fi_msg:
                    fe_explain += f" / [Fi] {str(fi_msg).split(chr(10))[0][:100]}"

                # [5번] 감정 맥락 기억 기반 tone 보강
                if hasattr(c, "emotion_context_memory"):
                    try:
                        _ec_ctx = c.emotion_context_memory.build_prompt_context(
                            current_vec=c.emotion_state.activations.tolist(),
                            n_recent=4,
                        )
                        if _ec_ctx:
                            fe_explain += f"\n{_ec_ctx}"
                    except Exception:
                        pass

                result = await generate_reasoning_and_response_async(
                    user_input=text_input,
                    prev_emotion=prev_vec,
                    new_emotion=c.emotion_state.get_vector(),
                    policy=policy_state,
                    abduction_ids=abduction_ids,
                    chat_history=chat_history,
                    explanation=fe_explain,
                    self_description=current_self_desc,
                    core_values_str=core_values_str,
                    causal_discoveries=None, abstract_matches=None,
                    alternatives=None,
                    purity=c.current_purity,
                    complex_states=c.current_complex_states,
                )
                reasoning_str = result.get("reasoning_log", "")
                natural_text  = result.get("natural_response", "")
            except Exception as e:
                print(f"⚠️ [Planner] CHAT 응답 생성 오류: {e}")
                natural_text = await generate_natural_response_async(
                    user_input=text_input,
                    emotion_vec=c.emotion_state.get_vector(),
                    is_event=False, policy=policy_state,
                    chat_history=chat_history,
                    purity=c.current_purity,
                    complex_states=c.current_complex_states,
                    explanation="사용자님과 대화하는 중",
                    self_description="나는 AI입니다.",
                )

        return {
            "classification":         classification,
            "analysis_dict":          analysis_dict,
            "fused_vector":           fused_vector,
            "concept_ids":            concept_ids,
            "current_event_vector":   event_vector,
            "prev_emotion_activations": prev_act,
            "prev_emotion_vector_tf": prev_vec,
            "target_activations":     target_act,
            "fe_context":             fe_ctx,
            "fi_alignment_score":     fi_score,
            "fi_message":             fi_msg,
            "abduction_ids":          abduction_ids,
            "policy_state":           policy_state,
            "reasoning_str":          reasoning_str,
            "natural_text":           natural_text,
            "suggested_actions":      result.get("suggested_actions", []) if isinstance(result, dict) else [],
            "fi_veto":                False,
        }

    def _update_kb(self, analysis_dict: Dict):
        c = self.core
        # ToM-1
        for b in analysis_dict.get("믿음", []):
            try:
                if isinstance(b, (list, tuple)) and len(b) == 2:
                    agent, fact = b
                    aid = (c.user_self_id_int if agent == "나(=USER)"
                           else c.kb.get_or_create_id(agent))
                    fid = c.kb.get_or_create_id(fact)
                    c.kb.update_agent_belief(aid, fid, 1.0)
            except Exception:
                pass
        # ToM-2
        for b in analysis_dict.get("2차 믿음", []):
            try:
                if isinstance(b, (list, tuple)) and len(b) == 3:
                    a, b2, f = b
                    c.kb.update_agent_2nd_order_belief(
                        c.kb.get_or_create_id(a),
                        c.kb.get_or_create_id(b2),
                        c.kb.get_or_create_id(f), 1.0
                    )
            except Exception:
                pass
        # 목표
        goals = analysis_dict.get("목표", [])
        if goals and isinstance(goals, list) and goals[0]:
            g = goals[0]
            if not any(lg.name == g for lg in c.long_term_goals):
                c.add_long_term_goal(g, priority=1.0, goal_type="user")

    async def _build_event_vector(
        self, analysis_dict: Dict
    ) -> Tuple[List[int], "np.ndarray"]:
        c = self.core
        objs  = analysis_dict.get("객체", [])
        iacts = analysis_dict.get("상호작용", [])
        props = analysis_dict.get("속성", [])
        concept_ids = (
            [c.kb.get_or_create_id(o) for o in objs]
            + [c.kb.get_or_create_id(i) for i in iacts]
            + [c.kb.get_or_create_id(p) for p in props]
        )
        if not concept_ids:
            vec = np.zeros((1, LSTM_UNITS))
        else:
            vec = await asyncio.to_thread(
                c.event_encoder.predict,
                np.array([concept_ids]), verbose=0
            )
        return concept_ids, vec


# ──────────────────────────────────────────────────────────────────────

class Executor:
    """
    Stage 4 — TASK 계획 수립 및 실행.

    처리 항목:
      - is_project_creation 판별 → Direct Scaffolding / DeepReason / 표준 플래너
      - WorldModel.simulate_plan() 검증
      - _execute_task() 실행
      - 결과 assimilation

    반환 dict 키:
      natural_text, reasoning_str, policy_state,
      execution_result_event, exec_task_state
    """

    def __init__(self, core: "EidosCore"):  # type: ignore
        self.core = core

    async def execute(
        self,
        plan_ctx: Dict[str, Any],
        text_input: str,
        image_input: Optional[bytes],
        chat_history: List[str],
        project_dir: Optional[str],
        _is_feedback_call: bool,
        user_text_short: Optional[str],
    ) -> Dict[str, Any]:
        c = self.core

        # ── Lite 모드 사용량 체크 ────────────────────────────────────
        if not _is_feedback_call:
            allowed, msg = execution_module.check_and_update_usage()
            if not allowed:
                uts = user_text_short or text_input[:20]
                return {
                    "policy_state":  "LITE_LIMIT_EXCEEDED_AT_EXECUTION",
                    "reasoning_str": "실행 계획 수립 완료, 일일 사용량 초과로 실행 차단",
                    "natural_text":  (
                        f"요청하신 '{uts}...' 작업의 실행 계획을 완료했습니다! 🚀\n\n"
                        "아쉽지만 오늘 무료 체험 횟수를 모두 사용하셨네요."
                    ),
                    "exec_task_state":        None,
                    "execution_result_event": None,
                    "lite_limit":             True,
                    "usage_msg":              msg,
                }

        fused_vector = plan_ctx.get("fused_vector", np.zeros((1, LSTM_UNITS)))

        # ── 계획 수립 ─────────────────────────────────────────────────
        plan_list = await self._build_plan(
            text_input, image_input, chat_history, fused_vector, _is_feedback_call
        )
        if not plan_list:
            return {
                "policy_state":           "PLAN_FAILED",
                "reasoning_str":          "계획 수립 실패",
                "natural_text":           "죄송합니다, 요청을 처리할 계획을 세우지 못했습니다.",
                "exec_task_state":        None,
                "execution_result_event": None,
            }

        # [Wire 1 Imagination] K 변형 × WorldModel·ValueEngine → ROI argmax.
        # flag off 시 no-op. chat_echo·단일-step plan 은 내부 가드로 보호.
        plan_list = await self._imagine_and_pick(plan_list)

        # [Fix 2026-04-19 D] Planner 가 LLM 대화 응답을 _chat_echo 마커로 회수한 경우:
        # Executor/ _execute_task 로 내려보내지 않고 자연어로 바로 전달. 이전엔 write_text
        # 폴백 → auto_generated_*.txt 재생성으로 흐르던 패턴 차단.
        if (
            isinstance(plan_list, list)
            and len(plan_list) == 1
            and isinstance(plan_list[0], dict)
            and plan_list[0].get("_chat_echo")
        ):
            _chat_resp = (plan_list[0].get("args", {}) or {}).get("response", "").strip()
            if _chat_resp:
                print("  💬 [Executor] chat_echo 단락 — Executor 스킵, 자연어 응답 전달")
                return {
                    "policy_state":           "CHAT_ECHO",
                    "reasoning_str":          "LLM 이 계획 대신 대화 응답 반환 → 직접 전달",
                    "natural_text":           _chat_resp,
                    "exec_task_state":        None,
                    "execution_result_event": None,
                }

        # ── 실행 ─────────────────────────────────────────────────────
        plan_json = json.dumps(plan_list)
        p_dir     = c._extract_project_dir_from_plan_helper(plan_json)
        print("⚡️ [Executor] 계획 실행 시작")
        exec_result = await c._execute_task(plan_json, p_dir)

        # ── 결과 동화 ─────────────────────────────────────────────────
        natural_text = f"작업이 완료되었습니다.\n\n{exec_result}"
        asyncio.create_task(
            c._assimilate_execution_result_async(exec_result, chat_history)
        )

        return {
            "policy_state":           "TASK_EXECUTED",
            "reasoning_str":          f"실행 계획 {len(plan_list)}단계 완료",
            "natural_text":           natural_text,
            "exec_task_state":        text_input[:40],
            "execution_result_event": exec_result,
        }

    # ═════════════════════════════════════════════════════════════════
    # [Wire 1 Imagination Loop] 계획 변형 × 시뮬레이션 → argmax 선택
    # ─────────────────────────────────────────────────────────────────
    # Executor 가 _build_plan 으로 베이스 계획을 얻은 뒤, 결정론적으로 K
    # 변형을 만들고 각각 WorldModel.simulate_plan + ValueEngine.score 로
    # ROI 를 비교해 가장 좋은 것을 선택. LLM 호출 없음.
    #
    # off-by-default: eidos_settings.json > loop_closure.wire1_imagination.
    # 길이 1 이하 / chat_echo 마커만 있는 plan 은 변형 여지 없음 → no-op.
    # 시뮬 실패한 변형은 ROI=-inf 로 자동 배제.
    # ═════════════════════════════════════════════════════════════════

    # 변형 생성에 사용하는 tool 위험도 가중치. WorldModel 내부 테이블과 톤
    # 맞춤. 여기 없는 tool 은 기본값 0.05.
    _IMG_TOOL_RISK = {
        "execute_python_file":            0.20,
        "write_complex_code_iteratively": 0.18,
        "write_project_files_async":      0.15,
        "write_file":                     0.10,
        "write_text":                     0.05,
        "perform_web_search":             0.02,
        "read_file":                      0.01,
    }

    def _build_plan_variants(
        self, base: List[Dict]
    ) -> List[Tuple[str, List[Dict]]]:
        """결정론적 변형 생성. 원본 + risk 최대 step 제거 + risk 오름차순 정렬.

        base 가 중첩 구조일 때도 dict.get("tool") 로 안전 접근. 원본과
        동일하거나 다른 변형과 중복이면 제외.
        """
        variants: List[Tuple[str, List[Dict]]] = [("original", list(base))]
        if not isinstance(base, list) or len(base) < 2:
            return variants

        def _risk_of(step: Any) -> float:
            if not isinstance(step, dict):
                return 0.05
            return self._IMG_TOOL_RISK.get(step.get("tool", ""), 0.05)

        # V2: risk 가중 최대 step 1개 제거 (길이 2 이상일 때만 의미있음)
        idx_max = max(range(len(base)), key=lambda i: _risk_of(base[i]))
        v2 = [s for i, s in enumerate(base) if i != idx_max]
        if v2 and v2 != variants[0][1]:
            variants.append(("skip_riskiest", v2))

        # V3: risk 오름차순 정렬 — 안전한 step 을 먼저 배치
        v3 = sorted(base, key=_risk_of)
        if v3 != variants[0][1] and all(v3 != v[1] for v in variants):
            variants.append(("safe_first", v3))

        return variants

    async def _imagine_and_pick(self, base_plan: List[Dict]) -> List[Dict]:
        """Wire 1 메인 진입. flag off / 변형 여지 없음 이면 원본 그대로 반환.

        ROI 최대 변형 선택. 동점이면 원본 우선.
        """
        flags = _agi_loop_flags()
        if not flags.get("wire1_imagination", False):
            return base_plan
        if not isinstance(base_plan, list) or len(base_plan) < 2:
            return base_plan
        # chat_echo / lite_limit / 기타 단일 step 마커 보호
        if len(base_plan) == 1:
            return base_plan

        variants = self._build_plan_variants(base_plan)
        if len(variants) == 1:
            return base_plan

        c = self.core
        scored: List[Tuple[str, List[Dict], float]] = []
        for name, variant in variants:
            try:
                sim = await c._simulate_plan_with_world_model_async(variant)
                roi = float(sim.get("roi", 0.0))
            except Exception as _e:
                print(f"  ⚠️ [Imagination] variant={name} 시뮬 실패 (무시): {_e}")
                roi = float("-inf")
            scored.append((name, variant, roi))
            print(f"  🎨 [Imagination] variant={name} len={len(variant)} "
                  f"ROI={roi:+.3f}")

        # 동점 시 원본 우선: 정렬 키에 (roi, is_original) 로 tiebreak
        def _key(t: Tuple[str, List[Dict], float]) -> Tuple[float, int]:
            return (t[2], 1 if t[0] == "original" else 0)
        scored.sort(key=_key, reverse=True)

        best_name, best_plan, best_roi = scored[0]
        if best_name != "original":
            original_roi = next(
                (t[2] for t in scored if t[0] == "original"),
                best_roi,
            )
            print(f"  ✨ [Imagination] 선택: {best_name} "
                  f"(ROI={best_roi:+.3f} vs original {original_roi:+.3f})")
        return best_plan

    async def _build_plan(
        self,
        text_input: str,
        image_input: Optional[bytes],
        chat_history: List[str],
        fused_vector: "np.ndarray",
        _is_feedback_call: bool,
    ) -> List[Dict]:
        c = self.core

        project_kws = [
            "프로젝트", "앱을", "앱 만들", "crm", "시스템 만들", "프로그램 만들",
            "프로그램 짜", "프로그램 제작", "앱 개발", "앱 제작", "툴 만들",
            "도구 만들", "서비스 만들", "서비스 개발", "웹사이트 만들", "웹앱 만들",
            "대시보드 만들", "봇 만들", "챗봇 만들", "관리 시스템", "관리 프로그램",
            "제작해줘", "개발해줘", "만들어줘",
        ]
        simple_kws = ["코드", "함수", "알고리즘", "코드 작성", "코드 짜", "소스"]
        is_simple   = any(k in text_input for k in simple_kws)
        is_project  = any(k in text_input for k in project_kws) and not is_simple

        # [A2] plan_ambition이 낮으면(0.3 미만) 무거운 Scaffolding 대신 표준 플래너로
        ambition = 1.0
        if hasattr(c, "_emotion_adapter"):
            ambition = c._emotion_adapter.plan_ambition

        if is_project and not _is_feedback_call and ambition >= 0.3:
            print(f"⚡️ [Executor] Direct Scaffolding 경로 (ambition={ambition:.2f})")
            try:
                return await c._generate_project_scaffolding_plan_async(text_input)
            except Exception as e:
                print(f"⚠️ [Executor] Scaffolding 실패 → 표준 플래너: {e}")
        elif is_project and ambition < 0.3:
            print(f"📋 [Executor] plan_ambition 낮음({ambition:.2f}) → 표준 플래너로 대체")

        if c._is_coding_request(text_input):
            print("🔁 [Executor] DeepReason 경로")
            try:
                tools_str, _ = c._tool_selector(text_input)
                proj_ctx     = c.project_helper.build_project_context()
                plan = await c._deep_reason_for_coding_async(
                    user_request=text_input,
                    chat_history=chat_history,
                    tools_list_str=tools_str,
                    project_context=proj_ctx,
                )
                if plan:
                    return plan
            except Exception as e:
                print(f"⚠️ [Executor] DeepReason 실패 → 표준 플래너: {e}")

        print("📋 [Executor] 표준 플래너 경로")
        return await c._generate_dynamic_plan_async(
            text_input=text_input,
            image_input=image_input,
            chat_history=chat_history,
            multimodal_vector=fused_vector,
            is_user_task=True,
        )


# ──────────────────────────────────────────────────────────────────────

class Integrator:
    """
    Stage 5 — 결과 통합 + 최종 return 조립.

    처리 항목:
      - 감정 최종 업데이트
      - 이벤트 로그 기록
      - 피드백 루프 (_is_feedback_call)
      - 12-tuple return 조립
    """

    def __init__(self, core: "EidosCore"):  # type: ignore
        self.core = core

    async def integrate(
        self,
        percept:      Dict[str, Any],
        plan_result:  Dict[str, Any],
        exec_result:  Optional[Dict[str, Any]],
        chat_history: List[str],
        _is_feedback_call: bool,
    ) -> "ProcessResult":  # type: ignore
        c = self.core

        classification = percept["classification"]
        analysis_dict  = percept["analysis_dict"]

        natural_text  = (exec_result or plan_result).get("natural_text", "")
        reasoning_str = (exec_result or plan_result).get("reasoning_str", "")
        policy_state  = (exec_result or plan_result).get("policy_state", "NORMAL")
        abduction_ids = plan_result.get("abduction_ids", [])
        exec_task     = (exec_result or {}).get("exec_task_state")
        exec_event    = (exec_result or {}).get("execution_result_event")
        suggested_actions: list = []  # 기본값 선언 — 조기 반환 경로에서 UnboundLocalError 방지

        # ── [Stage 3.5] dc_response → natural_text 변환 ──────────────
        dc_response = plan_result.get("dc_response")
        if dc_response and classification == "CHAT":
            try:
                _self_desc = "나는 AI입니다."
                if c.graph.has_node(c.eidos_self_id_int):
                    _self_desc = c.graph.nodes[c.eidos_self_id_int].get(
                        "self_description", _self_desc
                    )
                natural_text = await generate_natural_response_async(
                    user_input=percept.get("text_input", ""),
                    emotion_vec=c.emotion_state.get_vector(),
                    is_event=False,
                    policy=policy_state,
                    chat_history=chat_history,
                    purity=c.current_purity,
                    complex_states=c.current_complex_states,
                    explanation="분할정복 추론 결과를 바탕으로 응답",
                    self_description=_self_desc,
                    dc_context=dc_response,
                )
                print("  [Integrator] dc_response → natural_text 변환 완료")
            except Exception as _dc_conv_e:
                print(f"  [Integrator] dc_response 변환 오류 (기존 응답 유지): {_dc_conv_e}")

        # ── 감정 최종 업데이트 ────────────────────────────────────────
        target_act = plan_result.get("target_activations")
        if target_act is not None:
            final_act = c.emotion_momentum.apply(
                c.emotion_state.activations, target_act
            )
            c.emotion_state.update(final_act)
            c.current_purity, c.current_complex_states = (
                c.complex_emotion_monitor.analyze_state(c.emotion_state.activations)
            )

            # [A2] 감정 상태를 WorldModel의 EIDOS_SELF Actor KPI로 기록
            if c.world_model:
                try:
                    emo = c.emotion_state.activations
                    labels = {
                        0:"기쁨", 1:"슬픔", 2:"분노", 3:"공포",
                        4:"놀람", 5:"혐오", 6:"신뢰", 7:"기대",
                        8:"수치심", 9:"자부심", 10:"호기심", 11:"지루함",
                    }
                    for idx, name in labels.items():
                        if idx < len(emo):
                            c.world_model.update_kpi(
                                "EIDOS_SELF", f"감정_{name}",
                                float(emo[idx]), unit=""
                            )
                    # 리스크 허용도 / 탐색 바이어스도 KPI로 기록
                    if hasattr(c, "_emotion_adapter"):
                        c.world_model.update_kpi(
                            "EIDOS_SELF", "risk_tolerance",
                            c._emotion_adapter.risk_tolerance
                        )
                        c.world_model.update_kpi(
                            "EIDOS_SELF", "exploration_bias",
                            c._emotion_adapter.exploration_bias
                        )
                except Exception as _ek_e:
                    pass

        # ── 이벤트 로그 기록 (TASK/EVENT) ────────────────────────────
        if classification != "CHAT":
            ev_id = f"E-{len(c.event_log) + 1:03d}"
            ev_vec = plan_result.get("current_event_vector", np.zeros((1, LSTM_UNITS)))
            concept_ids = plan_result.get("concept_ids", [])
            iacts = analysis_dict.get("상호작용", []) if analysis_dict else []
            c.event_log.append((ev_vec[0], concept_ids, ev_id, iacts))
            c.event_id_to_concepts[ev_id] = concept_ids
            if len(c.event_log) > MAX_EVENT_LOG_SIZE:
                asyncio.create_task(c._prune_memory_async())

        # ── 피드백 루프 (실행 결과 → 다음 입력) ─────────────────────
        if exec_event and not _is_feedback_call and MAX_RECURSION_DEPTH > 0:
            print(f"🔄 [Integrator] 실행 결과 피드백 루프: '{exec_event[:50]}...'")
            try:
                await c.process_input(
                    text_input=exec_event,
                    image_input=None,
                    chat_history=chat_history,
                    _is_feedback_call=True,
                    _recursion_depth=1,
                )
            except Exception as e:
                print(f"⚠️ [Integrator] 피드백 루프 오류 (무시): {e}")

        # ── Lite 리밋 특수 반환 ───────────────────────────────────────
        if exec_result and exec_result.get("lite_limit"):
            return (
                c.graph, "LITE_LIMIT_EXCEEDED_AT_EXECUTION",
                c.emotion_state.get_vector(), 0.0,
                True, [], None,
                {"type": "LITE_LIMIT", "message": exec_result.get("usage_msg", "")},
                reasoning_str, natural_text,
                c.current_purity, c.current_complex_states,
                []
            )

        # ── Fi Veto 특수 반환 ────────────────────────────────────────
        if plan_result.get("fi_veto"):
            msg = plan_result.get("fi_msg", "이 요청은 제 가치관과 맞지 않습니다.")
            return (
                c.graph, "FI_VETO", c.emotion_state.get_vector(), 0.0,
                False, [], None, None,
                "Fi 가치관 검증 거절", msg,
                c.current_purity, c.current_complex_states,
                []
            )

        suggested_actions = (exec_result or plan_result).get("suggested_actions", [])
        return (
            c.graph, policy_state,
            c.emotion_state.get_vector(),
            float(c.prev_avg_reward),
            classification != "CHAT",
            abduction_ids,
            exec_task, None,
            reasoning_str, natural_text,
            c.current_purity, c.current_complex_states,
            suggested_actions
        )


MAX_RECURSION_DEPTH = 3   # Integrator 피드백 루프 깊이 제한

# ----------------------------------------------------------------------
# [v-Aira] Rule-based Emotion Delta Fallback
# emotion_network가 미학습 상태(delta ≈ 0)일 때 analysis_dict 기반으로 보정.
# NN이 충분히 학습되면 이 함수의 영향은 자동으로 줄어든다 (nn_magnitude >= 0.05).
# ----------------------------------------------------------------------
_EMOTION_KEYWORD_MAP: Dict[str, List[Tuple[int, float]]] = {
    # 긍정 트리거
    "감사":     [(0, 0.20), (6, 0.15)],          # 기쁨↑, 신뢰↑
    "성공":     [(0, 0.25), (9, 0.20)],           # 기쁨↑, 자부심↑
    "완료":     [(0, 0.18), (9, 0.15), (11, -0.10)],
    "칭찬":     [(0, 0.20), (9, 0.18), (6, 0.10)],
    "좋아":     [(0, 0.18), (7, 0.10)],
    "기대":     [(7, 0.22), (10, 0.12)],          # 기대↑, 호기심↑
    "흥미":     [(10, 0.20), (7, 0.12)],
    "재미":     [(0, 0.15), (10, 0.15)],
    "새로운":   [(10, 0.18), (4, 0.10)],          # 호기심↑, 놀람↑
    "도움":     [(6, 0.15), (0, 0.10)],           # 신뢰↑
    # 부정 트리거
    "오류":     [(1, 0.15), (3, 0.10)],           # 슬픔↑, 공포↑
    "실패":     [(1, 0.20), (9, -0.15)],          # 슬픔↑, 자부심↓
    "에러":     [(1, 0.15), (3, 0.10)],
    "문제":     [(3, 0.12), (10, 0.08)],          # 공포↑, 호기심↑(해결욕)
    "위험":     [(3, 0.20), (2, 0.12)],           # 공포↑, 분노↑
    "화":       [(2, 0.25)],                       # 분노↑
    "짜증":     [(2, 0.20), (5, 0.10)],
    "슬픔":     [(1, 0.22)],
    "무서":     [(3, 0.22)],
    "놀라":     [(4, 0.25)],                       # 놀람↑
    "모르":     [(10, 0.15)],                      # 호기심↑
    "왜":       [(10, 0.12)],
    "어떻게":   [(10, 0.10)],
}

def _compute_rule_based_emotion_delta(analysis_dict: Dict[str, Any]) -> np.ndarray:
    """
    [v-Aira] analysis_dict의 객체/상호작용/속성 키워드를 스캔해서
    감정 델타 벡터를 rule-based로 계산한다.
    NN 출력이 충분하면 호출 안 됨 (fallback only).
    """
    delta = np.zeros(12)  # EMOTION_DIM = 12

    # 분석 dict에서 텍스트 수집
    texts: List[str] = []
    for key in ("객체", "상호작용", "속성"):
        items = analysis_dict.get(key, [])
        if isinstance(items, list):
            texts.extend([str(x) for x in items])

    combined = " ".join(texts)

    for keyword, effects in _EMOTION_KEYWORD_MAP.items():
        if keyword in combined:
            for idx, strength in effects:
                delta[idx] += strength

    # 클리핑 (한 번에 너무 크게 튀지 않도록)
    delta = np.clip(delta, -0.4, 0.4)
    return delta


# ----------------------------------------------------------------------
# EIDOS Core (v12.0 - Autonomous Agent Foundation)
# ----------------------------------------------------------------------
class EidosCore:
    """ EIDOS v12.0 코어 (Semantic Embeddings + Autonomous Goal Gen) """
    MAX_AUTOFIX_ATTEMPTS = 3 # <<<--- [v20.0 Autofix] 자동 수정 최대 시도 횟수 (3회로 상향)
    AUTOFIX_TIMEOUT_SECONDS = 300 # 5분
    MAX_IDENTICAL_ERRORS = 2 # 동일 오류 2회 연속 발생 시 중단
    GAMMA_EMOTION_WEIGHT = 0.1
    DEFAULT_CLUSTER_N = 2
    GOAL_RELEVANCE_BONUS = 0.2
    MASTERY_LOSS_THRESHOLD = 0.1
    MASTERY_THRESHOLD_COUNT = 5
    CURRICULUM_DIFFICULTY_INCREMENT = 0.1
    ABSTRACTION_LINK_THRESHOLD = 0.9
    SCHEDULE_CHECK_INTERVAL_SECONDS = 60
    EMOTION_MIN = 0.0   # [Fix] llm_module.py와 통일 (기존 -1.0)
    EMOTION_MAX = 2.0   # [Fix] llm_module.py와 통일 (기존  1.0)
    EMOTION_DECAY = 0.05

    def __init__(self, use_llm=True, use_gpu=True, user_upgrades: Optional[Dict] = None, user_project_root: str = "eidos_files"):
        # 감정 시스템 (v9.x)
        self._init_character_system()
        self.emotion_state = EmotionState()
        self.emotion_dynamics = EmotionDynamics()
        self.emotion_context_memory = EmotionContextMemory()  # [5번] 감정 맥락 기억
        self.emotion_memory = EmotionMemory()
        self.emotion_momentum = EmotionMomentum()
        self.complex_emotion_monitor = ComplexEmotionMonitor()
        self.user_upgrades = user_upgrades if user_upgrades else {}
        if self.user_upgrades:
            print(f"✅ [Core] 사용자 업그레이드 적용됨: {self.user_upgrades}")
        self.schedule_lock = asyncio.Lock()
        self.training_lock = asyncio.Lock()
        self.graph_lock = asyncio.Lock()
        self.project_root = os.path.abspath(user_project_root)
        self.project_helper = ProjectStructureHelper(root_path=self.project_root)
        self.project_context = self.project_helper.build_project_context()
        print(f"🏗️ [Builder] Project Structure Loaded. (Context Size: {len(self.project_context)} chars)")
        print(f"🔒 [Core Sandbox] 이 인스턴스의 파일 시스템 루트: {self.project_root}")

        try:
            set_sandbox_root(self.project_root)
        except Exception as e_sandbox:
            print(f"❌ [Core] execution_module 샌드박스 루트 설정 실패: {e_sandbox}")

        # [v11.0] KB (ToM-2 지원)
        self.kb = KnowledgeBase() # Important: Initialize KB before Event Encoder
        self.graph = nx.MultiDiGraph()

        self.code_vector_db = CodeVectorDatabase(self)

        self.causal_engine = CausalInferenceEngine(self, self.graph, self.kb)
        self.otle = ObjectTemporalLoopEngine(self, self.kb)
        try:
            self.video_processor = VideoProcessor(self)
            print("✅ [Core] VideoProcessor (v18.8) 모듈 로드 완료.")
        except Exception as e_vid:
            print(f"❌ [Core] VideoProcessor 로드 실패: {e_vid}. (moviepy, pydub, opencv-python 필요)")
            self.video_processor = None
        self.core_identity = "EIDOS AI Core (v12.7)"
        self.event_log: List[Tuple[np.ndarray, List[int], str, List[str]]] = []
        self.event_id_to_concepts = {}
        self.eidos_self_id_int = self.kb.get_or_create_id("EIDOS_SELF")
        self.user_self_id_int = self.kb.get_or_create_id("USER_SELF") # <-- ✅ 이 줄 추가
        self.eidos_feel_id_int = self.kb.get_or_create_id("느끼다")
        self.emotion_ids = {name: self.kb.get_or_create_id(name) for name in EMOTION_MAP.values()}
        self.emotion_names_by_index = list(self.emotion_ids.keys())
        self.emotion_ids_by_index = list(self.emotion_ids.values())
        self.proactive_emotion_indices = [0, 1, 2, 3, 11] # Proactive 대상 감정 인덱스
        self.project_helper = ProjectStructureHelper(root_path=self.project_root)
        self.project_context = self.project_helper.build_project_context()

        # NN 모델 초기화
        # [v12.0 Semantic] build_event_encoder 호출 시 kb.vocab 전달
        print("🧠 [v12.0] Semantic Event Encoder 초기화 중...")
        # Make sure self.kb is initialized before this line
        self.event_encoder = build_event_encoder(self.kb.vocab) # <-- kb.vocab 전달
        print("🧠 Narrative Encoder 초기화 중...")
        self.narrative_encoder = build_narrative_encoder()
        print("🧠 Emotion Network 초기화 중...")
        self.emotion_network = build_emotion_network()
        print("🧠 Actor Network 초기화 중...")
        self.actor = build_actor_network()
        print("🧠 Critic Network 초기화 중...")
        self.critic = build_critic_network()
        try:
            self.multimodal_encoder = MultiModalEncoder()
        except Exception as e:
            print(f"CRITICAL: MultiModalEncoder 로드 실패! ({e}). 비전 기능 비활성화.")
            self.multimodal_encoder = None # Fallback

        try:
            self.world_model = WorldModel()
        except Exception as e:
            print(f"CRITICAL: WorldModel 로드 실패! ({e}). 예측 기능 비활성화.")
            self.world_model = None

        self.core_values: Dict[str, float] = {
            "USER_BENEFIT": 3.0,    # (이제 '명령 수행'이 최우선)
            "KNOWLEDGE_SEEKING": 1.5,
        }
        # 가치 키워드 (빠른 RL 보상 매칭용)
        self.value_keywords = {
            "USER_BENEFIT": ["돕기", "지원", "생성", "작성", "요약"],
            "KNOWLEDGE_SEEKING": ["탐색", "탐구", "검색", "학습", "질문"]
        }

        try:
            self.safety_module = SafetyModule(self)
        except Exception as e:
            print(f"CRITICAL: SafetyModule 로드 실패! ({e}).")
            self.safety_module = None

        print(f"EIDOS v12.0 ... (AI-Perception 탑재 시도)")

        # [수정됨] 무조건 LLM 파서만 사용 (Spacy 제거)
        self.USE_LLM_PARSER = True 
        # self.local_oia_parser = OIAParserV1() # 삭제됨
        print("✅ [AI v14.0] 'Gemini LLM 파서'를 기본 인식 모듈로 사용합니다.")

        # 나머지 초기화
        self.is_pro_mode = False
        self.event_emotion_map = {}
        self.strategy_log = {}
        self.similarity_cache = {}
        self.long_term_goals = []
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_ACTOR)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_CRITIC) # <--- ✅ 이렇게 수정
        self.memory_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)
        self.priority_memory_buffer = deque(maxlen=int(REPLAY_BUFFER_SIZE * 0.2)) # 일반 버퍼의 20%
        self.recent_emotions = deque(maxlen=PLANNING_HORIZON)
        self.current_goal_weighting = np.ones(EMOTION_DIM)
        self.long_term_goal_emotion = LONG_TERM_GOAL_EMOTION
        self.current_goal = None # 현재 EIDOS가 집중하는 목표 (사용자 설정 또는 자율 생성)
        self.no_event_counter = 0
        self.current_purity: float = 1.0
        self.current_complex_states: Dict[str, float] = {}
        self.current_curriculum_difficulty: float = 0.5
        self.low_loss_counter: int = 0
        self.training_lock = asyncio.Lock()
        self.graph_lock = asyncio.Lock()
        self.self_model_vector = np.zeros(2 * LSTM_UNITS + EMOTION_DIM)
        self.self_model_update_rate = 0.01 # 1%씩 천천히 자기 모델 업데이트
        self.recent_rewards_for_meta = deque(maxlen=100) # 최근 100회 보상 추적
        self.prev_avg_reward = 0.0
        self.base_actor_lr = LEARNING_RATE_ACTOR # 기본 학습률 저장
        self.schedule: List[Dict[str, Any]] = [] # 일정 목록 (시간, 작업, 준비물, 절차 등 저장)
        self._schedule_monitor_task: Optional[asyncio.Task] = None # 주기적 검사 태스크 핸들
        self.load_memory()


        self.abstract_former = AbstractConceptFormer(self)
        self.cf_simulator = CounterfactualSimulator(self)
        self.episodic_memory = EpisodicMemoryModule(self)
        self.self_reflector = SelfReflectionModule(self) # [AI-6] 추가

        # [AGI] 자율 에이전트 핵심 모듈 초기화
        self.self_model = SelfModel(self)
        self.roi_goal_generator   = ROIGoalGenerator(self)
        self.goal_stack_evaluator = GoalStackEvaluator(self)   # [A1]
        self._emotion_adapter     = EmotionBehaviorAdapter(self)  # [A2]
        self._input_classifier    = InputClassifier(self)         # [B2]

        # [B1] 파이프라인 단계 인스턴스
        self._input_router = InputRouter(self)
        self._perceiver    = Perceiver(self)
        self._planner      = Planner(self)
        self._executor     = Executor(self)
        self._integrator   = Integrator(self)
        self.last_prev_emotion_vec = np.zeros((1, EMOTION_DIM))
        self.last_event_vector = np.zeros((1, LSTM_UNITS))
        self.abstraction_cycle_counter = 0
        self.goal_system_update_counter = 0
        self.GOAL_SYSTEM_UPDATE_RATE = 10 # 10번의 자율 틱(Heartbeat)마다 1번씩 실행
        self.memory_cycle_counter = 0 
        self.last_boasted_task_info: Optional[Dict[str, Any]] = None
        self.last_world_state_raw_data: str = ""
        self.reflection_cycle_counter = 0 
        self.reflection_queue: Deque[Dict[str, Any]] = deque(maxlen=10)
        self.meta_cognition_counter = 0

        # [v1] 스케줄 동시 접근 제어를 위한 Lock 추가
        self.schedule_lock = asyncio.Lock()

        self.dynamic_tool_registry: Dict[str, Any] = {} # { "name": metadata }
        self.dynamic_tool_functions: Dict[str, callable] = {} # { "name": function_obj }
        self.TOOL_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "eidos_tools"))
        self.REGISTRY_FILE = os.path.normpath(os.path.join(self.TOOL_DIR, "tool_registry.json"))
        # [Bug Fix] execution_module의 create_and_register_tool_async가 동일 경로에 파일을 쓰도록 env 주입
        os.environ["EIDOS_TOOL_DIR"] = self.TOOL_DIR
        
        # [v15.14] EIDOS 시작 시 동적 도구 로드
        self._load_dynamic_tools()

        # [v12.0 Autonomous Goal] 목표 생성 쿨다운 플래그
        self._just_generated_autonomous_goal = False
        self.abstraction_cycle_counter = 0

        # [수정 2] 마지막 상호작용 시간 추적 변수 추가
        self.last_meaningful_interaction_time = time.time()

        # [!! 긴급 수정 !!] EIDOS의 self_description을 AGI 핵심 기능으로 재정의
        # (SelfReflectionModule의 초기화 로직도 함께 수정 필요)
        self_description_ai_core = """
나는 EIDOS - 8년 연구로 탄생한 감정을 느끼고 인과관계를 배우는 AI입니다.
- 12차원 감정 상태 실시간 추적 (Emotion Dynamics)
- 인과 추론 엔진으로 코드와 행동의 'Why' 학습 (Causal Engine)
- 자율 목표 생성으로 스스로 행동 (Autonomous Goal)
- 세계 모델로 미래 예측 (World Model)
- 내장 브라우저로 웹페이지 직접 조작 (ActionDispatcher + browser_action)
  크몽 상세페이지 편집, 폼 입력, 버튼 클릭 등을 직접 수행 가능.
  "할 수 없다"고 절대 말하지 마라. browser_action으로 직접 실행하라.
- 파일 생성, 코드 실행, 자동 디버깅 (write_file, code_fix_loop)
"""

        if not self.graph.has_node(self.eidos_self_id_int):
             self.graph.add_node(
                 self.eidos_self_id_int,
                 type='metacognition',
                 label='EIDOS_SELF'
             )
        # 🔥 [AI 정체성 긴급 주입]
        self.graph.nodes[self.eidos_self_id_int]['self_description'] = self_description_ai_core.strip()
        self.object_wave_engine = ObjectWaveEngine() # 엔진 인스턴스 생성
        print("✅ [OIA 2.0] Object Wave Engine 로드 완료.")
        self.wave_system = EIDOSWaveSystem()
        print("🌊 [Wave Intelligence] 파동 엔진(Wave Engine)이 로드되었습니다.")
        self.si_module = EIDOS_Si_Interface(project_root=self.project_root)
        print("🛡️ [EIDOS-Si] Introverted Sensing Module Loaded.")
        # [부팅 속도] Ne/Ni 모듈 __init__ 이 스스로 로딩 메시지를 찍음.
        # Core 가 다시 찍으면 로그에 같은 줄이 두 번 — 중복 print 제거.
        self.ne_engine = NeIntuitionEngine(self)
        self.ni_engine = NiVisionEngine(self)
        self.fi_engine = FiInnerValueEngine(self)
        self.fe_engine = FeEmpathyEngine(self)
        print("❤️🤝 [EIDOS-F] Feeling Engines (Fi/Fe) Loaded.")
        # -------------------------------------------------------------

        # ═══════════════════════════════════════════════════════════════
        # [Phase 1] 닫힌 학습 루프 패치
        # TD-오류 SelfModel + 오류 신호 자율 목표 + 인과 WorldModel
        # ═══════════════════════════════════════════════════════════════
        try:
            from eidos_phase1_integration import apply_phase1
            apply_phase1(self)
        except Exception as _e_phase1:
            print(f"⚠️ [Phase1] 패치 실패 (기존 모드로 계속): {_e_phase1}")

        # ═══════════════════════════════════════════════════════════════
        # [CausalMatrix] X변수 초기화 — PatrolAgent + MissionSchema 연동
        # PatrolAgent가 순찰 시 수치를 갱신하고,
        # 스키마 승인 시 ActionDispatcher가 이 X변수로 trainer를 생성한다.
        # ═══════════════════════════════════════════════════════════════
        try:
            from eidos_causal_matrix import XVar, CausalMatrixTrainer, MATRIX_CACHE

            # 기본 X변수 정의 (크몽 프리랜서 도메인 기준)
            # PatrolAgent가 순찰할 때 update_x(name, value)로 갱신
            self._causal_x_vars = [
                XVar("경쟁사_리뷰수",    current_value=0.5, source="patrol",
                     description="동일 카테고리 경쟁 서비스 평균 리뷰 수 (정규화)"),
                XVar("내_응답속도",      current_value=0.7, source="user",
                     description="메시지 평균 응답 시간 (빠를수록 1.0)"),
                XVar("포트폴리오_다양성", current_value=0.5, source="user",
                     description="등록 포트폴리오 카테고리 다양성 (0~1)"),
                XVar("가격_경쟁력",      current_value=0.6, source="patrol",
                     description="동일 카테고리 대비 가격 경쟁력 (낮을수록 1.0)"),
                XVar("키워드_노출도",    current_value=0.4, source="patrol",
                     description="검색 결과 상단 노출 확률 (0~1)"),
            ]

            # 기존 학습 데이터가 있으면 trainer 미리 로드
            # (Y변수는 스키마 승인 시 HOW_MUCH로 덮어씀)
            self._causal_trainer = None
            if os.path.exists(MATRIX_CACHE):
                try:
                    self._causal_trainer = CausalMatrixTrainer.load(MATRIX_CACHE)
                    # X변수 목록 동기화 (저장된 것과 현재 정의가 다를 수 있음)
                    if len(self._causal_trainer.x_vars) != len(self._causal_x_vars):
                        self._causal_trainer.x_vars = self._causal_x_vars
                        self._causal_trainer.n_x    = len(self._causal_x_vars)
                        self._causal_trainer._net   = None  # 차원 바뀌면 가중치 리셋
                        print("  [CausalMatrix] X변수 차원 변경 → 가중치 리셋")
                    print(f"  ✅ [CausalMatrix] 기존 trainer 로드 완료 "
                          f"(ext={len(self._causal_trainer._pairs_ext)}, "
                          f"int={len(self._causal_trainer._pairs_int)})")
                except Exception as _e_load:
                    print(f"  ⚠️ [CausalMatrix] trainer 로드 실패, 새로 시작: {_e_load}")
                    self._causal_trainer = None
            else:
                print("  [CausalMatrix] X변수 초기화 완료 "
                      f"({len(self._causal_x_vars)}개) — 스키마 승인 시 trainer 생성")

        except ImportError:
            self._causal_x_vars  = []
            self._causal_trainer = None
            print("  ⚠️ [CausalMatrix] eidos_causal_matrix 없음 — 비활성화")
        except Exception as _e_cm_init:
            self._causal_x_vars  = []
            self._causal_trainer = None
            print(f"  ⚠️ [CausalMatrix] 초기화 실패 (무시): {_e_cm_init}")

        print(f"EIDOS v12.0 (Autonomous Agent Foundation) Core 로드 (REPLAY_BUFFER={REPLAY_BUFFER_SIZE})")

    def _init_character_system(self):
        """
        캐릭터 이벤트 시스템 초기화
        GUI와 연결되는 콜백 시스템
        """
        self._character_callbacks = []
        self._last_user_input_time = time.time()

        # [v-Character Idle] idle 감지 시스템
        self._idle_spoken_at: float = 0.0          # 마지막 idle 발화 시각
        self._idle_cooldown_sec: float = 300.0      # idle 발화 최소 간격 (5분)
        self._idle_trigger_sec: float = 180.0       # 유저 무응답 기준 시간 (3분)
        self._last_emotion_vec: Optional[np.ndarray] = None  # 이전 감정 벡터 (변화 감지용)

    def register_character_callback(self, callback):
        """
        GUI에서 캐릭터 이벤트를 받기 위해 등록
        """
        if callback not in self._character_callbacks:
            self._character_callbacks.append(callback)

    def trigger_character_event(self, event_type, text=None, emotion=None, expr_filename=None):
        """
        GUI 캐릭터에게 이벤트 전달
        """
        event = {
            "type":          event_type,
            "text":          text,
            "emotion":       emotion,
            "expr_filename": expr_filename,
            "timestamp":     time.time()
        }

        for cb in self._character_callbacks:
            try:
                cb(event)
            except Exception as e:
                print("Character callback error:", e)

    def notify_user_input(self): 
        """
        사용자 활동 기록
        """
        self._last_user_input_time = time.time()

    def build_failure_message(self, attempts: int, last_error: dict) -> str:

        error_type = last_error.get("classified_type", "UnknownError")
        error_msg = last_error.get("error_message", "No message")

        root = last_error.get("root_cause", {})
        file_name = root.get("file_normalized", "unknown")
        line = root.get("line", "unknown")
        func = root.get("function", "unknown")

        return (
            "🚨 **EIDOS 자동 수정이 최종적으로 실패했습니다.**\n\n"

            f"**시도 횟수:** {attempts}\n"
            f"**오류 타입:** `{error_type}`\n"
            f"**오류 메시지:** `{error_msg}`\n\n"

            "**추정 위치**\n"
            f"- 파일: `{file_name}`\n"
            f"- 함수: `{func}`\n"
            f"- 라인: `{line}`\n\n"

            "⚠️ 자동 수정이 여러 번 실패했기 때문에 "
            "**부분 패치 방식으로 해결하기 어려운 구조적 문제일 가능성**이 있습니다.\n\n"

            "👉 아래 분석과 개선 전략을 참고하여 다시 시도해 보세요."
        )

    def analyze_failure_root_cause(self, error_history: list) -> str:

        if not error_history:
            return "분석할 오류 기록이 없습니다."

        type_counter = {}
        file_counter = {}   
        func_counter = {}

        for e in error_history:

            etype = e.get("classified_type", "Unknown")

            root = e.get("root_cause", {})
            file = root.get("file_normalized", "unknown")
            func = root.get("function", "unknown")

            type_counter[etype] = type_counter.get(etype, 0) + 1
            file_counter[file] = file_counter.get(file, 0) + 1
            func_counter[func] = func_counter.get(func, 0) + 1

        if not type_counter:
            return "분석할 오류 데이터가 없습니다."

        most_common_type = max(type_counter, key=type_counter.get)
        most_common_file = max(file_counter, key=file_counter.get)
        most_common_func = max(func_counter, key=func_counter.get)

        freq = type_counter[most_common_type]
   
        analysis = [
            f"가장 많이 발생한 오류 유형: `{most_common_type}` ({freq}회)",
            f"오류가 집중된 파일: `{most_common_file}`",
            f"오류가 집중된 함수: `{most_common_func}`",
        ]

        # root cause heuristic
        if most_common_type in ["SyntaxError", "IndentationError"]:
            cause = "코드 구조 또는 들여쓰기 문제"
  
        elif most_common_type in ["ModuleNotFoundError", "ImportError"]:
            cause = "의존성 또는 import 설정 문제"

        elif most_common_type in ["TypeError", "AttributeError"]:
            cause = "객체 타입 또는 API 사용 방식 문제"

        elif most_common_type in ["NameError"]:
            cause = "변수 스코프 또는 선언 누락"

        else:
            cause = "논리 오류 또는 런타임 문제"

        analysis.append(f"추정 근본 원인: **{cause}**")

        return "\n".join(analysis)

    def suggest_improved_prompts(self, last_error: dict) -> List[str]:

        error_type = last_error.get("classified_type", "UnknownError")
        error_msg = last_error.get("error_message", "")

        root = last_error.get("root_cause", {})
        file_name = root.get("file_normalized", "unknown")
        func = root.get("function", "unknown")

        prompts = []
 
        prompts.append(
            f"{file_name} 파일에서 발생한 `{error_type}` 오류를 해결하도록 코드를 재작성해줘.\n"
            f"오류 메시지: {error_msg}"
        )

        prompts.append(
            f"`{func}` 함수만 독립적으로 실행 가능한 최소 코드로 다시 작성해줘."
        )

        prompts.append(
            "현재 코드 구조가 오류를 반복적으로 발생시키고 있습니다.\n"
            "동일 기능을 더 단순하고 안정적인 구조로 다시 설계해줘."
        )

        prompts.append(
            f"`{error_type}` 오류가 발생하지 않도록 방어 코드를 추가한 안전한 버전을 작성해줘."
        )

        # 타입별 전략

        if error_type in ["AttributeError", "TypeError"]:

            prompts.append(
                "객체 타입 검사를 추가하고 hasattr / isinstance 기반 방어 코드를 넣어줘."
            )

        if error_type in ["ModuleNotFoundError", "ImportError"]:

            prompts.append(
                "필요한 라이브러리 설치 방법과 함께 코드 수정안을 제시해줘."
            )

        if error_type in ["SyntaxError", "IndentationError"]:
  
            prompts.append(
                "전체 파일을 문법적으로 올바른 구조로 다시 정리해줘."
            )

        return prompts

    def detect_error_loop(self, error_history):

        if len(error_history) < 2:
            return False

        last = error_history[-1].get("classified_type")
        prev = error_history[-2].get("classified_type")

        if not last or not prev:
            return False

    def estimate_fix_difficulty(self, error_history):

        if len(error_history) < 2:
            return "low"
 
        types = [e.get("classified_type", "Unknown") for e in error_history]

        if len(set(types)) == 1:
            return "medium"

        return "high"

    async def generate_alternative_code(
        self,
        original_code: str,
        last_error: dict
    ) -> dict:

        root = last_error.get("root_cause", {})

        file_name = root.get("file_normalized", "unknown")
        line = root.get("line", -1)
        func = root.get("function", "unknown")

        # 문제 코드 영역 추출
        problem_snippet = ""

        try:
            lines = original_code.splitlines()

            if isinstance(line, int) and 0 < line <= len(lines):
                start = max(0, line - 5)
                end = min(len(lines), line + 5)
                problem_snippet = "\n".join(lines[start:end])
  
        except Exception:
            problem_snippet = ""

        prompt = {
            "task": "alternative_design",

            "error": last_error,

            "problem_location": {
                "file": file_name,
                "line": line,
                "function": func
            },

            "problem_snippet": problem_snippet,

            "original_code": original_code,

            "instruction": (
                "오류를 유발한 함수 또는 코드 영역을 분석하라.\n"
                "다음 세 가지 대안을 제시하라.\n" 
                "1) 최소 수정 패치\n"  
                "2) 문제 함수 완전 재작성\n"
                "3) 더 단순한 구조의 전체 코드\n"
                "각 제안은 설명 + 코드 형태로 작성하라."
            ),

            "project_context": self.project_context
        }

        return await self.llm_module.generate_alternatives_async(prompt)

    async def generate_failure_report(
        self,
        attempts: int,
        error_history: list,
        original_code: str
    ) -> str:

        last_error = error_history[-1] if error_history else {}
 
        msg1 = self.build_failure_message(attempts, last_error)

        msg2 = self.analyze_failure_root_cause(error_history)

        prompts = self.suggest_improved_prompts(last_error)

        alternatives = await self.generate_alternative_code(
            original_code,
            last_error
        )

        if isinstance(alternatives, list):
            alternatives = "\n".join(map(str, alternatives))

        elif isinstance(alternatives, dict):
            alternatives = json.dumps(alternatives, indent=2)

        alternatives = str(alternatives)

        # 오류 통계   
        type_counts = {}

        for e in error_history:
            t = e.get("classified_type", "Unknown")
            type_counts[t] = type_counts.get(t, 0) + 1

        stats = "\n".join(
            [f"- {k}: {v}회" for k, v in type_counts.items()]
        )

        report = (
            msg1 + "\n\n"

            "### 🔍 자동 분석된 실패 원인\n"
            f"{msg2}\n\n"
 
            "### 📊 오류 통계\n"
            f"{stats}\n\n"

            "### 💡 재시도 프롬프트 제안\n"
            + "\n".join([f"- {p}" for p in prompts]) + "\n\n"

            "### 🛠 대안 코드 설계\n"
            f"{alternatives}\n"
        )

        return report

    async def run_autofix_pipeline(self, file_path: str) -> Dict[str, Any]:

        print(f"\n🚀 [Autofix Pipeline] 시작: {file_path}")

        error_history = []
        start_time = time.time()

        # 1️⃣ 초기 실행
        initial_result = await self.exec_engine.execute_python_file(file_path)

        if initial_result.success:
            return {
                "status": "success",
                "message": "초기 실행에서 오류 없이 성공했습니다.",
                "file": file_path
            }

        # 2️⃣ 오류 분석
        initial_error = {
            "classified_type": initial_result.error_type,
            "error_message": initial_result.error_message,
            "root_cause": {
                "file_normalized": initial_result.file_path,
                "line": initial_result.error_line,
                "function": initial_result.error_function
            }
        }

        self.kb.record_execution_error(initial_error)
        error_history.append(initial_error)

        print(f"🔥 최초 오류: {initial_error['classified_type']}")

        # 코드 로드
        def sync_read(): 
            try:  
                with open(file_path, "r", encoding="utf-8") as f:
                    return f.read()
            except Exception:
                return ""

        original_code = await asyncio.to_thread(sync_read)

        # 3️⃣ Autofix 루프
        fix_success = False

        for attempt in range(self.MAX_AUTOFIX_ATTEMPTS):

            print(f"\n🔧 Autofix 시도 {attempt + 1}")

            fix_success, message, error_history = await self.autofix_loop(
                file_path=file_path,
                original_code=original_code,
                initial_error=error_history[-1],
                error_history=error_history
            )

            if fix_success:
                break

            # 동일 오류 반복 감지
            if len(error_history) >= self.MAX_IDENTICAL_ERRORS:

                last_types = [
                    e["classified_type"]
                    for e in error_history[-self.MAX_IDENTICAL_ERRORS:]
                ]

                if len(set(last_types)) == 1:
                    print("⚠️ 동일 오류 반복 감지 → Autofix 중단")
                    break

        # 4️⃣ 테스트 실행
        test_pass = False

        if fix_success:

            print("🧪 테스트 실행")

            test_pass = await self.run_test_pipeline(file_path)

        # 5️⃣ 성공 처리
        if fix_success and test_pass:

            duration = time.time() - start_time

            print("✅ Autofix 성공")

            return {
                "status": "success",
                "file": file_path,
                "time_seconds": round(duration, 2)
            }

        # 6️⃣ 실패 리포트 생성
        failure_report = await self.generate_failure_report(
            attempts=self.MAX_AUTOFIX_ATTEMPTS,
            error_history=error_history,
            original_code=original_code
        )

        duration = time.time() - start_time

        print("❌ Autofix 실패")

        return {
            "status": "failed",
            "file": file_path,
            "time_seconds": round(duration, 2),
            "message": failure_report
        }

    # ──────────────────────────────────────────────────────────────────
    # [GoalPlan] code_fix_loop — 코드 실행→스크린샷→VLM→diff 수정 루프
    # ──────────────────────────────────────────────────────────────────

    async def _run_code_fix_loop(
        self,
        filepath: str,
        goal: str = "",
        max_iterations: int = 5,
        **kwargs,
    ) -> str:
        """
        코드 실행 → 스크린샷 캡처 → VLM 분석 → diff 기반 국소 수정 자동 루프.

        1. 파일 실행 (subprocess)
        2. 2초 대기 후 스크린샷 캡처
        3. 콘솔 출력(stdout/stderr) + 스크린샷을 VLM에 전송
        4. VLM이 "문제 없음" 판단 → 성공 반환
        5. VLM이 문제 발견 → before/after diff 생성 → 파일에 적용
        6. 1번으로 복귀
        """
        import subprocess

        # 경로 보정
        abs_path = os.path.join(self.project_root, filepath) if not os.path.isabs(filepath) else filepath
        if not os.path.exists(abs_path):
            return f"❌ [code_fix_loop] 파일 없음: {abs_path}"

        print(f"\n🔄 [code_fix_loop] 시작: {filepath} (최대 {max_iterations}회)")
        print(f"  목표: {goal[:80]}")

        for iteration in range(1, max_iterations + 1):
            print(f"\n  ── [Iteration {iteration}/{max_iterations}] ──")

            # ── 1. 코드 실행 ──────────────────────────────────────────
            exec_result = await self._exec_and_capture(abs_path)
            stdout = exec_result.get("stdout", "")
            stderr = exec_result.get("stderr", "")
            returncode = exec_result.get("returncode", -1)
            process_alive = exec_result.get("process_alive", False)

            print(f"  실행 결과: returncode={returncode} "
                  f"{'(프로세스 생존 — GUI?)' if process_alive else ''}")
            if stderr:
                print(f"  stderr 앞 300자: {stderr[:300]}")

            # ── 2. 스크린샷 캡처 ──────────────────────────────────────
            screenshot_bytes = None
            if process_alive or returncode == 0:
                await asyncio.sleep(2.0)  # GUI 렌더링 대기
                screenshot_bytes = await asyncio.to_thread(self._capture_screenshot)
                if screenshot_bytes:
                    print(f"  📸 스크린샷 캡처 완료 ({len(screenshot_bytes)} bytes)")

            # ── 3. VLM 분석 ───────────────────────────────────────────
            analysis = await self._analyze_code_with_vlm(
                filepath=filepath,
                goal=goal,
                stdout=stdout,
                stderr=stderr,
                returncode=returncode,
                screenshot_bytes=screenshot_bytes,
            )

            status = analysis.get("status", "error")
            issues = analysis.get("issues", "")
            print(f"  VLM 판단: {status}")

            # ── 4. 성공 ───────────────────────────────────────────────
            if status == "ok":
                print(f"  ✅ [code_fix_loop] 성공! ({iteration}회 만에)")
                # GUI 프로세스 종료
                pid = exec_result.get("pid")
                if pid:
                    try:
                        os.kill(pid, 9)
                    except Exception:
                        pass
                return f"✅ code_fix_loop 성공 ({iteration}회): {filepath}"

            # ── 5. diff 기반 수정 ─────────────────────────────────────
            print(f"  🔧 문제 감지 → diff 수정 시도")
            try:
                code = await asyncio.to_thread(
                    lambda: open(abs_path, "r", encoding="utf-8").read()
                )
            except Exception as e:
                print(f"  ❌ 파일 읽기 실패: {e}")
                continue

            diff_response = await self._generate_code_diff(
                code=code,
                issues=issues,
                goal=goal,
                filepath=filepath,
            )

            if not diff_response:
                print(f"  ⚠️ diff 생성 실패 → 다음 시도")
                continue

            # diff 적용
            patched_code = self._apply_diff(code, diff_response)
            if patched_code == code:
                print(f"  ⚠️ diff 적용 결과 변화 없음 → 다음 시도")
                continue

            # 파일 저장
            try:
                await asyncio.to_thread(
                    lambda c=patched_code: open(abs_path, "w", encoding="utf-8").write(c)
                )
                print(f"  💾 수정된 코드 저장 완료")
            except Exception as e:
                print(f"  ❌ 파일 저장 실패: {e}")
                continue

            # GUI 프로세스 종료 (다음 이터레이션에서 재실행)
            pid = exec_result.get("pid")
            if pid:
                try:
                    os.kill(pid, 9)
                except Exception:
                    pass

        return f"❌ code_fix_loop {max_iterations}회 실패: {filepath}"

    async def _exec_and_capture(self, abs_path: str) -> Dict[str, Any]:
        """
        파이썬 파일을 subprocess로 실행하고 stdout/stderr/returncode 반환.
        GUI 프로그램이면 프로세스가 살아있는 채로 반환 (process_alive=True).
        """
        import subprocess
        import sys

        result = {
            "stdout": "", "stderr": "", "returncode": -1,
            "process_alive": False, "pid": None,
        }

        try:
            proc = await asyncio.to_thread(
                lambda: subprocess.Popen(
                    [sys.executable, abs_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    cwd=os.path.dirname(abs_path),
                    encoding="utf-8",
                    errors="replace",
                )
            )
            result["pid"] = proc.pid

            # 5초 대기 — 콘솔 프로그램이면 종료, GUI면 살아있음
            try:
                stdout, stderr = await asyncio.to_thread(
                    lambda: proc.communicate(timeout=5)
                )
                result["stdout"] = stdout or ""
                result["stderr"] = stderr or ""
                result["returncode"] = proc.returncode
            except Exception:
                # 타임아웃 = 프로세스 아직 살아있음 (GUI)
                result["process_alive"] = True
                result["returncode"] = 0
                # 현재까지의 stderr 캡처 시도
                try:
                    if proc.stderr:
                        result["stderr"] = proc.stderr.read() or ""
                except Exception:
                    pass

        except Exception as e:
            result["stderr"] = str(e)
            result["returncode"] = -1

        return result

    def _capture_screenshot(self) -> Optional[bytes]:
        """현재 화면 스크린샷을 PNG bytes로 반환"""
        try:
            import pyautogui
            img = pyautogui.screenshot()
            import io
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()
        except ImportError:
            try:
                from PIL import ImageGrab
                img = ImageGrab.grab()
                import io
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                return buf.getvalue()
            except Exception:
                pass
        except Exception as e:
            print(f"  ⚠️ [Screenshot] 캡처 실패: {e}")
        return None

    async def _analyze_code_with_vlm(
        self,
        filepath: str,
        goal: str,
        stdout: str,
        stderr: str,
        returncode: int,
        screenshot_bytes: Optional[bytes] = None,
    ) -> Dict[str, str]:
        """
        콘솔 출력 + 스크린샷을 VLM에 보내서 문제 분석.
        반환: {"status": "ok" | "error" | "ui_issue", "issues": "문제 설명"}
        """
        # 콘솔 오류가 명확하면 VLM 없이 텍스트 분석만
        if returncode != 0 and stderr:
            return {
                "status": "error",
                "issues": f"[콘솔 오류 (returncode={returncode})]\n{stderr[:1000]}",
            }

        # 스크린샷 있으면 VLM 분석
        if screenshot_bytes:
            try:
                from llm_module import get_llm_response_vision_async
                vlm_prompt = (
                    f"이 프로그램의 스크린샷을 분석해주세요.\n\n"
                    f"[프로그램 목표]\n{goal[:500]}\n\n"
                    f"[파일]\n{filepath}\n\n"
                    f"[콘솔 출력]\nstdout: {stdout[:300]}\nstderr: {stderr[:300]}\n\n"
                    f"[판단 기준]\n"
                    f"1. GUI가 정상적으로 표시되는가?\n"
                    f"2. 오류 대화상자나 크래시가 보이는가?\n"
                    f"3. UI 디자인이 조잡하거나 깨져 보이는가?\n"
                    f"4. 기능적으로 목표에 부합하는가?\n\n"
                    f"반드시 아래 형식으로만 응답:\n"
                    f"STATUS: ok 또는 error 또는 ui_issue\n"
                    f"ISSUES: 문제가 있다면 구체적으로 설명 (없으면 '없음')"
                )
                vlm_result = await get_llm_response_vision_async(
                    vlm_prompt, screenshot_bytes, filepath
                )
                # 파싱 — STATUS/ISSUES 형식 + 키워드 fallback (4-4 수정)
                status = "ok"
                issues = ""
                _parsed_status = False
                for line in (vlm_result or "").split("\n"):
                    line = line.strip()
                    if line.upper().startswith("STATUS:"):
                        s = line.split(":", 1)[1].strip().lower()
                        if "error" in s:
                            status = "error"
                        elif "ui" in s or "issue" in s:
                            status = "ui_issue"
                        else:
                            status = "ok"
                        _parsed_status = True
                    elif line.upper().startswith("ISSUES:") or line.upper().startswith("ISSUE:"):
                        issues = line.split(":", 1)[1].strip()

                # STATUS 키가 없었으면 전체 텍스트에서 키워드로 판단 (fallback)
                if not _parsed_status:
                    _vl = (vlm_result or "").lower()
                    if any(k in _vl for k in ["오류", "에러", "error", "crash", "traceback", "문제"]):
                        status = "error"
                        issues = issues or vlm_result[:200]
                    else:
                        status = "ok"

                if issues and issues != "없음":
                    return {"status": status, "issues": issues}
                else:
                    return {"status": "ok", "issues": ""}

            except ImportError:
                print("  ⚠️ [VLM] get_llm_response_vision_async 미사용 가능")
            except Exception as e:
                print(f"  ⚠️ [VLM] 분석 실패: {e}")

        # 스크린샷 없을 때 — 코드 정적 분석으로 대체 (4-2, 4-3 수정)
        # screenshot=None + returncode=0 이면 무조건 ok 처리하던 버그 수정.
        # 대신 파일을 직접 읽어 LLM에게 텍스트 기반 검토 요청.
        if returncode != 0 or stderr:
            return {
                "status": "error",
                "issues": f"[stderr]\n{stderr[:1000]}" if stderr else "알 수 없는 오류",
            }

        # returncode==0, stderr 없음, 스크린샷도 없음 → 코드 정적 분석
        try:
            abs_path = os.path.join(
                getattr(self, "project_root", "."), filepath
            ) if not os.path.isabs(filepath) else filepath
            if os.path.exists(abs_path):
                code_text = open(abs_path, encoding="utf-8", errors="replace").read()
                from llm_module import get_llm_response_async as _llm
                static_prompt = (
                    f"아래 파이썬 코드가 다음 목표를 달성하는지 검토해줘.\n"
                    f"목표: {goal[:300]}\n\n"
                    f"[코드]\n{code_text[:3000]}\n\n"
                    f"[콘솔 출력]\nstdout: {stdout[:200]}\n\n"
                    f"반드시 아래 형식으로만 답해:\n"
                    f"STATUS: ok 또는 error\n"
                    f"ISSUES: 문제가 있다면 구체적으로 (없으면 '없음')"
                )
                static_result = await _llm(static_prompt, max_tokens=200)
                status = "ok"
                issues = ""
                for line in (static_result or "").split("\n"):
                    line = line.strip()
                    if line.upper().startswith("STATUS:"):
                        s = line.split(":", 1)[1].strip().lower()
                        status = "error" if "error" in s else "ok"
                    elif line.upper().startswith("ISSUES:"):
                        issues = line.split(":", 1)[1].strip()
                print(f"  📝 [code_fix_loop] 정적 분석 결과: {status} / {issues[:60]}")
                if issues and issues != "없음":
                    return {"status": status, "issues": issues}
                return {"status": "ok", "issues": ""}
        except Exception as _se:
            print(f"  ⚠️ [code_fix_loop] 정적 분석 실패 (무시): {_se}")

        # 최후 fallback: returncode==0이고 다른 분석도 불가 → ok 처리
        return {"status": "ok", "issues": ""}

    async def _generate_code_diff(
        self,
        code: str,
        issues: str,
        goal: str,
        filepath: str,
    ) -> str:
        """
        VLM이 발견한 문제를 기반으로 before/after diff를 LLM에게 요청.
        """
        # 코드가 너무 길면 앞뒤 2000자만
        if len(code) > 5000:
            code_snippet = code[:2500] + "\n\n... (중략) ...\n\n" + code[-2500:]
        else:
            code_snippet = code

        prompt = f"""당신은 코드 디버거입니다. 아래 문제를 수정하세요.

[파일] {filepath}
[목표] {goal[:300]}

[발견된 문제]
{issues[:500]}

[현재 코드]
```python
{code_snippet}
```

[수정 규칙]
1. 반드시 BEFORE/AFTER 형식으로 **변경할 부분만** 출력하라.
2. 전체 코드를 다시 쓰지 마라. 변경이 필요한 부분만 정확히 지정하라.
3. 여러 곳을 수정해야 하면 각각 별도의 BEFORE/AFTER 블록으로 작성하라.
4. BEFORE 블록의 코드는 원본과 **정확히 일치**해야 한다 (공백, 들여쓰기 포함).

[출력 형식]
<<<BEFORE>>>
수정 전 코드 (원본에서 정확히 복사)
<<<AFTER>>>
수정 후 코드
<<<END>>>

<<<BEFORE>>>
두 번째 수정 전 코드
<<<AFTER>>>
두 번째 수정 후 코드
<<<END>>>"""

        try:
            result = await get_llm_response_async(prompt, max_tokens=4096)
            return result
        except Exception as e:
            print(f"  ❌ [code_fix_loop] diff 생성 LLM 호출 실패: {e}")
            return ""

    def _apply_diff(self, original_code: str, diff_response: str) -> str:
        """
        BEFORE/AFTER 형식의 diff를 원본 코드에 순서대로 적용.
        매칭 실패 시 해당 블록은 건너뜀.
        """
        import re

        # BEFORE/AFTER/END 블록 파싱
        pattern = r'<<<BEFORE>>>\s*\n(.*?)<<<AFTER>>>\s*\n(.*?)<<<END>>>'
        matches = re.findall(pattern, diff_response, re.DOTALL)

        if not matches:
            # 대안: ```python 블록 안에 before/after가 있을 수 있음
            pattern2 = r'BEFORE[:\s]*\n```[^\n]*\n(.*?)```\s*\nAFTER[:\s]*\n```[^\n]*\n(.*?)```'
            matches = re.findall(pattern2, diff_response, re.DOTALL)

        if not matches:
            print(f"  ⚠️ [_apply_diff] diff 블록을 파싱할 수 없음")
            return original_code

        result = original_code
        applied = 0

        for before, after in matches:
            before = before.rstrip('\n')
            after = after.rstrip('\n')

            if before in result:
                result = result.replace(before, after, 1)
                applied += 1
                print(f"  ✏️ [diff] 적용 성공 (before {len(before)}자 → after {len(after)}자)")
            else:
                # 공백/줄바꿈 차이 무시하고 재시도
                before_stripped = '\n'.join(
                    line.rstrip() for line in before.split('\n')
                )
                result_stripped_check = '\n'.join(
                    line.rstrip() for line in result.split('\n')
                )
                if before_stripped in result_stripped_check:
                    # 원본에서 공백 보존한 채로 교체
                    lines = result.split('\n')
                    b_lines = before.strip().split('\n')
                    a_lines = after.strip().split('\n')
                    # 첫 줄 매칭 위치 찾기
                    for i, line in enumerate(lines):
                        if line.rstrip() == b_lines[0].rstrip():
                            # 연속 매칭 확인
                            if all(
                                i + j < len(lines)
                                and lines[i + j].rstrip() == b_lines[j].rstrip()
                                for j in range(len(b_lines))
                            ):
                                lines[i:i + len(b_lines)] = a_lines
                                result = '\n'.join(lines)
                                applied += 1
                                print(f"  ✏️ [diff] 공백 보정 후 적용 성공")
                                break
                    else:
                        print(f"  ⚠️ [diff] before 블록 매칭 실패 (건너뜀)")
                else:
                    print(f"  ⚠️ [diff] before 블록 매칭 실패 (건너뜀)")

        print(f"  📝 [diff] 총 {applied}/{len(matches)} 블록 적용")
        return result


    def decide_refactor_strategy(
        self,
        error_history: list,
        identical_error_count: int
    ) -> str:
        """
        EIDOS Refactor Strategy Engine
        결정 전략:
        - patch_fix
        - partial_refactor
        - full_refactor
        """

        if not error_history:
            return "patch_fix"

        # 동일 오류 반복 → 전체 재설계
        if identical_error_count >= self.MAX_IDENTICAL_ERRORS:
            return "full_refactor"

        # 최근 오류 타입 분석
        error_types = [
            e.get("classified_type", "Unknown")
            for e in error_history
        ]

        # Syntax / Import 오류는 부분 수정 우선
        if any(t in ["SyntaxError", "IndentationError"] for t in error_types):
            return "partial_refactor"

        if any(t in ["ModuleNotFoundError", "ImportError"] for t in error_types):
            return "partial_refactor"

        # 파일 집중도 분석
        if len(error_history) >= 3:

            recent_files = [
                e.get("root_cause", {}).get("file_normalized")
                for e in error_history[-3:]
            ]

            if len(set(recent_files)) == 1:
                return "partial_refactor"

        # 다양한 오류 발생 → 구조 문제 가능성
        if len(set(error_types)) > 2:
            return "full_refactor"

        return "patch_fix"

    async def run_test_pipeline(self, file_path: str) -> bool:
        """
        EIDOS Test Pipeline
        1) 테스트 실행
        2) 결과 분석
        3) threshold 평가
        """

        try:

            test_result = await self.execute_with_timeout(
                self.run_tests_async(file_path),
                timeout_sec=10
            )

        except asyncio.TimeoutError:

            print("⏰ 테스트 타임아웃")
            return False

        except Exception as e:

            print(f"❌ 테스트 실행 오류: {e}")
            return False

        # 결과 평가
        evaluation = self.evaluate_test_result(test_result)

        if not evaluation:
            return False

        if not self.test_passes_threshold(evaluation):
            print("⚠️ 테스트 기준 미달")
            return False

        return True

    async def perform_partial_refactor(
        self,
        file_path: str,
        original_code: str,
        error_details: dict
    ) -> str:
        """
        Partial Refactor
        해당 파일 내 문제 코드 영역을 중심으로 구조 개선
        """

        root = error_details.get("root_cause", {})

        line = root.get("line", -1)

        snippet = ""

        try:

            lines = original_code.splitlines()

            if line > 0 and line < len(lines):

                start = max(0, line - 8)
                end = min(len(lines), line + 8)

                snippet = "\n".join(lines[start:end])

        except Exception:
            snippet = ""

        prompt = {

            "task": "partial_refactor",

            "file_path": file_path,

            "error": error_details,

            "problem_snippet": snippet,

            "original_code": original_code,

            "instruction": (
                "이 코드를 동일 기능을 유지하면서 더 안정적인 구조로 개선하라.\n"
                "특히 다음을 개선하라:\n"
                "- 함수 책임 분리\n"
                "- 예외 처리 강화\n"
                "- 타입 안정성\n"
                "- 코드 가독성\n"
                "- 반복 코드 제거\n"
                "문제 코드 영역을 우선 수정하라."
            ),

            "quality_requirements": [
                "예외 처리 강화",
                "타입 안정성",
                "가독성 개선",
                "중복 코드 제거"
            ],

            "project_context": self.project_context
        }

        try:

            new_code = await self.llm_module.full_rewrite_async(prompt)

        except Exception as e:

            print(f"⚠️ Partial Refactor 실패: {e}")
            return original_code

        return new_code

    async def perform_full_refactor(
        self,
        file_path: str,
        original_code: str
    ) -> str:
        """
        Full Refactor
        파일 전체 구조 재설계
        """

        prompt = {

            "task": "full_refactor",

            "file_path": file_path,

            "original_code": original_code,

            "instruction": (
                "이 파일 전체를 완전히 재설계하라.\n"
                "코드 품질과 유지보수성을 극대화하라.\n"
                "기능 요구사항은 유지해야 한다."
            ),

            "architectural_goals": [

                "단일 책임 원칙(SRP)",
                "모듈 간 결합도 최소화",
                "테스트 가능성 향상",
                "예외 처리 강화",
                "명확한 함수 책임 분리",
                "확장 가능한 구조"

            ],

            "design_patterns": [

                "Dependency Injection",
                "Strategy Pattern",
                "Factory Pattern"

            ],

            "project_context": self.project_context
        }

        try:

            rewritten = await self.llm_module.full_rewrite_async(prompt)

        except Exception as e:

            print(f"⚠️ Full Refactor 실패: {e}")
            return original_code

        return rewritten



    def run_rewrite_validation(self, new_code: str) -> tuple:
        """
        재작성된 코드 검증:
        - AST 문법 검사
        - 구조 검사
        - 위험 코드 탐지
        """

        if not new_code or len(new_code.strip()) < 10:
            return False, "[코드 오류] 재작성된 코드가 비어있거나 너무 짧습니다."

        # AST 문법 검사
        is_valid, error_msg = self.validate_syntax(new_code)
        if not is_valid:
            return False, f"[코드 문법 오류] {error_msg}"

        try:
            tree = ast.parse(new_code)
        except Exception as e:
            return False, f"[AST 파싱 실패] {str(e)}"

        func_count = 0
        class_count = 0

        for node in ast.walk(tree):

            if isinstance(node, ast.FunctionDef):
                func_count += 1

            if isinstance(node, ast.ClassDef):
                class_count += 1

            # 위험 코드 탐지
            if isinstance(node, ast.Call):
                if hasattr(node.func, "id") and node.func.id in ["eval", "exec"]:
                    return False, "[보안 경고] eval/exec 사용 감지"

        if func_count == 0 and class_count == 0:
            return False, "[구조 오류] 함수 또는 클래스가 존재하지 않습니다."

        return True, None

    async def run_refactor_strategy(
        self,
        file_path: str,
        original_code: str,
        error_history: list,
        identical_error_count: int
    ) -> str:

        strategy = self.decide_refactor_strategy(
            error_history=error_history,
            identical_error_count=identical_error_count
        )

        print(f"🧠 Refactor Strategy: {strategy}")

        try:

            if strategy == "partial_refactor":

                new_code = await self.perform_partial_refactor(
                    file_path=file_path,
                    original_code=original_code,
                    error_details=error_history[-1]
                )

            elif strategy == "full_refactor":

                new_code = await self.perform_full_refactor(
                    file_path=file_path,
                    original_code=original_code
                )

            else:
                # patch 전략 fallback
                new_code = original_code

        except Exception as e:

            print(f"⚠️ Refactor 실행 실패: {e}")
            new_code = original_code

        # 검증
        ok, msg = self.run_rewrite_validation(new_code)

        if not ok:

            print(f"❌ Refactor 검증 실패: {msg}")

            # fallback → 전체 재설계
            if strategy != "full_refactor":

                print("🔁 Full Refactor fallback 실행")

                new_code = await self.perform_full_refactor(
                    file_path=file_path,
                    original_code=original_code
                )

                ok, msg = self.run_rewrite_validation(new_code)

                if not ok:
                    raise RuntimeError(f"[Refactor 검증 실패] {msg}")

        return new_code

    async def run_tests_async(self, file_path: str) -> Dict[str, Any]:
        """
        테스트 실행 모듈
        - 코드 실행
        - stdout/stderr 수집
        - 실행 시간 기록
        """

        start_time = time.time()

        try:

            result = await self.exec_engine.execute_python_file(file_path)

            elapsed = time.time() - start_time

            return {
                "success": result.success,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "runtime": elapsed,
                "file_path": file_path,
                "error_type": getattr(result, "error_type", None)
            }

        except Exception as e:

            elapsed = time.time() - start_time

            return {
                "success": False,
                "error": str(e),
                "runtime": elapsed,
                "file_path": file_path
            }

    def evaluate_test_result(self, test_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        테스트 결과 자동 평가
        """

        evaluation = {
            "success": test_result.get("success", False),
            "runtime": test_result.get("runtime"),
            "issues": [],
            "score": 1.0
        }

        # 실행 실패
        if not test_result.get("success"):
            evaluation["issues"].append("RuntimeError")
            evaluation["score"] -= 0.5

        # stderr 존재
        stderr = test_result.get("stderr", "")
        if stderr:
            evaluation["issues"].append("StderrOutput")
            evaluation["score"] -= 0.2

        # runtime 분석
        runtime = test_result.get("runtime", 0)

        if runtime > 5:
            evaluation["issues"].append("SlowExecution")
            evaluation["score"] -= 0.1

        # stdout 품질 검사
        stdout = test_result.get("stdout", "")

        if len(stdout) > 10000:
            evaluation["issues"].append("ExcessiveOutput")
            evaluation["score"] -= 0.1

        evaluation["score"] = max(0.0, evaluation["score"])

        return evaluation

    def test_passes_threshold(self, evaluation: Dict[str, Any]) -> bool:
        """
        테스트 성공 기준 강화
        """

        if not evaluation.get("success"):
            return False

        issues = evaluation.get("issues", [])

        # stderr 문제
        if "StderrOutput" in issues:
            return False

        # runtime 검사
        runtime = evaluation.get("runtime", 999)
        if runtime > 10:
            return False

        # score 기반 평가
        score = evaluation.get("score", 1.0)
        if score < 0.5:
            return False

        return True

    def validate_expected_output(self, expected_files: List[str]) -> bool:
        """
        출력 파일 검증 강화
        """

        now = time.time()

        for f in expected_files:

            if not os.path.exists(f):
                return False

            # 파일 크기 체크
            if os.path.isfile(f):

                size = os.path.getsize(f)

                if size == 0:
                    return False

                # 최근 수정 여부 확인
                modified = os.path.getmtime(f)

                if now - modified > 300:
                    return False

        return True

    async def execute_with_timeout(self, coro, timeout_sec=10):
        """
        Timeout 보호 실행
        """

        task = asyncio.create_task(coro)

        try:

            return await asyncio.wait_for(task, timeout=timeout_sec)

        except asyncio.TimeoutError:

            task.cancel()
  
            try:
                await task
            except:
                pass

            return {
                "success": False,
                "error": f"Timeout({timeout_sec}초)"
            }

        except Exception as e:

            return {
                "success": False,
                "error": str(e)
            }


    async def _sense_user_schedule_async(self) -> Optional[Dict[str, Any]]:
        """
        일정 센서
        """

        try:

            now = datetime.datetime.now()

            # 실제 구현 시 calendar API 호출
            events = getattr(self, "schedule", [])

            for event in events:

                start = event.get("start_time")
                end = event.get("end_time")
                title = event.get("title", "일정")

                if not start:
                    continue

                # 시작 알림
                delta_start = (start - now).total_seconds()

                if 0 < delta_start < 900:

                    return {
                        "type": "PRE_EVENT_REMINDER",
                        "context": title,
                        "details": f"{title} 일정이 곧 시작됩니다."
                    }

                if end:

                    delta_end = (now - end).total_seconds()

                    if 0 < delta_end < 900:

                        return {
                            "type": "POST_EVENT_SUMMARIZATION",
                            "context": title,
                            "details": f"{title} 일정이 종료되었습니다."
                        }

        except Exception as e:

            print(f"❌ [Schedule Sensor] 오류: {e}")

        return None

    async def apply_proposals_sequentially(self, original_code: str, proposals: list) -> str:
        """
        여러 proposal sequential 적용
        """

        updated_code = original_code

        if not proposals:
            return original_code

        for i, proposal in enumerate(proposals):

            try:

                if "old" not in proposal or "new" not in proposal:
                    continue

                updated_code = await self.merge_single_proposal(
                    updated_code,
                    proposal
                )

            except Exception as e:

                self.log(f"[Proposal Merge Error] {i}: {str(e)}")

                raise RuntimeError(
                    f"Proposal merge failed: {str(e)}"
                )

        return updated_code

    async def merge_single_proposal(self, original_code: str, proposal: dict) -> str:
        """
        코드 병합 알고리즘 강화
        """

        old = proposal.get("old")
        new = proposal.get("new")

        if not old or not new:
            raise ValueError("Proposal must include 'old' and 'new'.")

        # 1 정확 매칭
        if old in original_code:

            return original_code.replace(old, new, 1)

        # 2 fuzzy line match
        import difflib

        lines = original_code.splitlines()
        old_lines = old.splitlines()

        match = difflib.get_close_matches(
            old,
            lines,
            n=1,
            cutoff=0.5
        )

        if match:

            idx = lines.index(match[0])
            lines[idx] = new

            return "\n".join(lines)

        # 3 함수 단위 탐색
        if "def " in old:

            for i, line in enumerate(lines):

                if line.strip().startswith("def"):

                    lines.insert(i + 1, new)

                    return "\n".join(lines)

        # 4 fallback append
        return original_code + f"\n\n# AUTO PATCH\n{new}\n"

    async def resolve_conflict(
        self,
        original_code: str,
        proposal_a: dict,
        proposal_b: dict
    ) -> str:
        """
        충돌 해결 엔진
        """

        strategies = [

            (proposal_a, proposal_b),
            (proposal_b, proposal_a)

        ]

        for first, second in strategies:

            try:

                code = await self.merge_single_proposal(
                    original_code,
                    first
                )

                code = await self.merge_single_proposal(
                    code,
                    second
                )

                valid, _ = self.validate_syntax(code)

                if valid:
                    return code

            except Exception:
                continue

        # LLM fallback

        merged = await self.llm_module.fix_conflict_with_llm(
            original_code=original_code,
            proposal_a=proposal_a,
            proposal_b=proposal_b,
            project_context=self.project_context
        )

        valid, _ = self.validate_syntax(merged)

        if valid:
            return merged

        return original_code

    def validate_syntax(self, code: str) -> tuple:
        """
        Python AST 기반 문법 체크 (강화 버전)
        return (True, None) or (False, error_message)
        """

        import ast

        if not code or len(code.strip()) < 5:
            return False, "코드가 비어있거나 너무 짧습니다."

        try:

            tree = ast.parse(code)

        except SyntaxError as e:

            return False, f"{e.msg} (line {e.lineno})"

        # 구조 검사
        func_count = 0
        class_count = 0

        for node in ast.walk(tree):

            if isinstance(node, ast.FunctionDef):
                func_count += 1

            if isinstance(node, ast.ClassDef):
                class_count += 1

            # 위험 코드 검사
            if isinstance(node, ast.Call):

                if hasattr(node.func, "id"):

                    if node.func.id in ["eval", "exec"]:
                        return False, "위험 함수(eval/exec) 사용 감지"

        if func_count == 0 and class_count == 0:
            return False, "함수 또는 클래스 정의가 없습니다."

        return True, None


    def generate_diff(self, old_code: str, new_code: str) -> list:
        """
        line-based diff 생성 (강화 버전)
        """

        import difflib

        old_lines = old_code.splitlines()
        new_lines = new_code.splitlines()

        diff = list(difflib.unified_diff(
            old_lines,
            new_lines,
            n=3,
            lineterm=""
        ))

        # 변경 통계 로그
        added = sum(1 for line in diff if line.startswith("+") and not line.startswith("+++"))
        removed = sum(1 for line in diff if line.startswith("-") and not line.startswith("---"))

        self.log(f"[Diff] +{added} / -{removed} lines changed")

        return diff


    def apply_patch(self, original_code: str, diff_lines: list) -> str:
        """
        unified diff를 적용하여 새 코드를 생성 (강화 버전)
        """

        import difflib

        try:

            patched = list(difflib.restore(diff_lines, 1))

        except Exception as e:

            raise RuntimeError(f"Patch 복원 실패: {str(e)}")

        if not patched:
            raise RuntimeError("Patch 적용 실패")

        new_code = "\n".join(patched)

        valid, msg = self.validate_syntax(new_code)

        if not valid:

            raise RuntimeError(f"Patch 적용 후 문법 오류: {msg}")

        return new_code


    async def autofix_loop(
        self,
        file_path: str,
        original_code: str,
        initial_error: Dict[str, Any]
    ) -> Tuple[bool, str]:

        """
        EIDOS Autofix Loop (강화 버전)
        """

        attempt_count = 0
        identical_error_count = 0
        previous_error_type = None

        error_history = [initial_error]

        start_time = time.time()

        while attempt_count < self.MAX_AUTOFIX_ATTEMPTS:

            # timeout 보호
            if time.time() - start_time > self.AUTOFIX_TIMEOUT_SECONDS:

                return False, f"[Autofix Timeout] {self.AUTOFIX_TIMEOUT_SECONDS}초 초과"

            attempt_count += 1

            self.log(
                f"\n[Autofix Attempt {attempt_count}/{self.MAX_AUTOFIX_ATTEMPTS}]"
            )

            current_error = error_history[-1]

            error_type = current_error.get(
                "classified_type",
                "UnknownError"
            )

            # 동일 오류 반복 감지
            if error_type == previous_error_type:

                identical_error_count += 1

            else:

                identical_error_count = 0

            if identical_error_count >= self.MAX_IDENTICAL_ERRORS:

                return False, (
                    f"[Autofix 중단] 동일 오류 {error_type} "
                    f"{self.MAX_IDENTICAL_ERRORS}회 반복"
                )

            previous_error_type = error_type

            # 코드 자동 수정 요청
            try:

                new_code = await self.llm_module.modify_code_async(
                    file_path=file_path,
                    original_code=original_code,
                    error_details=current_error,
                    project_context=self.project_context,
                    rag=self.rag,
                    kb=self.knowledge_base
                )

            except Exception as e:

                return False, f"[modify_code_async 실패] {str(e)}"

            # 코드 검증
            valid, msg = self.validate_syntax(new_code)

            if not valid:

                self.log(f"[Autofix] 생성 코드 문법 오류: {msg}")

                error_history.append({
                    "classified_type": "GeneratedSyntaxError",
                    "error_message": msg
                })

                continue

            # 코드 저장
            try:

                await self.write_project_files_async(
                    [(file_path, new_code)]
                )

            except Exception as e:

                return False, f"[파일 저장 실패] {str(e)}"

            # RAG index 업데이트
            try:

                await self.rag.index_single_file_async(file_path)

            except Exception as e:

                self.log(f"[RAG Index Warning] {str(e)}")

            # 코드 실행
            self.log("[Autofix] 수정된 코드 실행 중")

            result = await self.exec_engine.execute_python_file(
                file_path
            )

            # 성공
            if result.success:

                return True, "[Autofix 성공] 코드 실행 정상"

            # 실패 → 오류 기록
            last_error = {

                "classified_type": result.error_type,
                "error_message": result.error_message,

                "root_cause": {
                    "file_normalized": result.file_path, 
                    "line": result.error_line,
                    "function": result.error_function
                }

            }

            self.knowledge_base.record_execution_error(last_error)

            error_history.append(last_error)

            self.log(
                f"[Autofix] 오류 지속: {last_error.get('classified_type')}"
            )

            # rollback 대비 코드 갱신
            original_code = new_code

        return False, "[Autofix 실패] 최대 시도 횟수 초과"


    async def _bridge_concept_to_wave_async(self, concept_names: List[str]):
        """ [Bridge] 텍스트 개념(OIA)을 벡터로 변환하여 파동 엔진에 등록합니다. """

        if not concept_names:
            return

        # 중복 제거
        concept_names = list(set(concept_names))

        try:

            concept_ids = [
                self.kb.get_or_create_id(name)
                for name in concept_names
            ]

            input_tensor = np.array([concept_ids])

            vector_batch = await asyncio.to_thread(
                self.event_encoder.predict,
                input_tensor,
                verbose=0
            )

            embedding_vector = vector_batch[0]

        except Exception as e:

            print(f"❌ [Wave Bridge] 벡터 생성 실패: {e}")
            return

        # 파동 엔진 등록
        for name in concept_names:

            try:
                self.wave_system.register_object(
                    name,
                    embedding_vector
                )

            except Exception as e:

                print(f"⚠️ [Wave Bridge] 등록 실패: {name} | {e}")

    async def _perform_rag_search_async(self, query: str, top_k: int = 3) -> List[str]:
        """
        Knowledge Base 기반 RAG 검색 (강화 버전)
        """

        if not query:
            return []

        query_tokens = query.lower().split()

        results = []

        try:

            for name, cid in self.kb.vocab.items():

                name_lower = name.lower()

                score = sum(
                    token in name_lower
                    for token in query_tokens
                )

                if score > 0:

                    results.append(
                        (score, name)
                    )

        except Exception as e:

            print(f"❌ [RAG Search] 오류: {e}")
            return []

        results.sort(reverse=True)

        top_results = [
            f"KB Fact: {name[:60]}..."
            for _, name in results[:top_k]
        ]

        if not top_results:
            return ["External KB Check: No critical facts found."]

        return top_results

    def set_pro_mode(self, state: bool):
        """ 유료 모드 상태 설정 """

        if self.is_pro_mode == state:
            return

        self.is_pro_mode = state

        try:

            execution_module.set_global_pro_mode(state)

        except Exception as e:

            print(f"❌ [PRO MODE] execution_module 동기화 실패: {e}")

        print(f"💰 [PRO MODE] 상태 변경됨 → {state}")

    def is_pro_mode_enabled(self) -> bool:
        """ 현재 유료 모드 상태를 반환합니다. """
        return self.is_pro_mode

    async def reset_trial_counter_async(self) -> str:
        """ 무료 체험 횟수 초기화 """

        if not self.is_pro_mode:

            return "오류: Pro 모드(관리자)만 사용할 수 있습니다."

        try:

            result = await asyncio.to_thread(
                execution_module.reset_trial_counter
            )

            return result

        except Exception as e:

            return f"초기화 실패: {str(e)}"

    async def _analyze_execution_error_async(self, error_text: str) -> Optional[Dict[str, Any]]:
        """
        Python 실행 오류 구조 분석 (강화 버전)
        """

        if not error_text:
            return None

        ERROR_KNOWLEDGE_BASE = {
            "NameError": {
                "common_causes": [
                    "변수 이름 오타",
                    "import 누락",
                    "스코프 문제"
                ],
                "auto_fix_strategies": [
                    "import 자동 추가",
                    "유사 변수 검색",
                    "LLM 재작성"
                ]
            },
            "FileNotFoundError": {
                "common_causes": [
                    "경로 오타",
                    "파일 미존재"
                ],
                "auto_fix_strategies": [
                    "경로 자동 생성",
                    "파일 존재 검사 추가"
                ]
            },
            "TypeError": {
                "common_causes": [
                    "타입 불일치",
                    "함수 인자 오류"
                ],
                "auto_fix_strategies": [
                    "타입 변환",
                    "함수 시그니처 확인"
                ]
            }
        }

        KNOWN_ERROR_TYPES = list(ERROR_KNOWLEDGE_BASE.keys())

        traceback_blocks = re.findall(
            r"Traceback \(most recent call last\):([\s\S]+)",
            error_text,
            re.DOTALL
        )

        if not traceback_blocks:
            return None

        traceback_content = traceback_blocks[-1]

        lines = traceback_content.strip().splitlines()

        error_line = lines[-1]

        error_type_raw = "UnknownError"
        error_message = error_line

        match = re.match(
            r"([\w\.]+Error):(.*)",
            error_line
        )

        if match:

            error_type_raw = match.group(1).strip()
            error_message = match.group(2).strip()

        # 스택 파싱
        stack_trace = []

        frame_pattern = re.compile(
            r'File "([^"]+)", line (\d+), in (.+)'
        )

        for line in lines:

            match = frame_pattern.search(line)

            if match:

                file_path, line_num, func = match.groups()

                stack_trace.append({

                    "file_raw": file_path,
                    "file_normalized": os.path.normpath(file_path),
                    "line": int(line_num),
                    "function": func

                })

        # 분류
        classified_type = "UnknownError"
        confidence_score = 0.4

        if error_type_raw in KNOWN_ERROR_TYPES:

            classified_type = error_type_raw
            confidence_score = 0.95

        # heuristic fallback
        msg = error_message.lower()

        if "not defined" in msg:
            classified_type = "NameError"
            confidence_score = 0.9

        elif "no such file" in msg:
            classified_type = "FileNotFoundError"
            confidence_score = 0.9

        root_cause = stack_trace[-1] if stack_trace else None

        analysis_result = {

            "error_type_raw": error_type_raw,
            "error_message": error_message,
            "classified_type": classified_type,
            "confidence_score": confidence_score,
            "stack_trace": stack_trace,
            "root_cause": root_cause,
            "knowledge_base_info": ERROR_KNOWLEDGE_BASE.get(classified_type)

        }

        print(
            f"✅ [Error Classifier] "
            f"{error_type_raw} → {classified_type} "
            f"(confidence {confidence_score:.2f})"
        )

        return analysis_result

    async def _process_execution_feedback_async(
        self,
        execution_result_event: str,
        prev_state: np.ndarray,
        prev_action: int
    ) -> float:

        """
        실행 결과 분석 → RL reward 계산
        """

        reward = 0.0

        error_details = await self._analyze_execution_error_async(
            execution_result_event
        )

        is_failure = False

        if error_details:

            is_failure = True

            self.kb.record_execution_error(error_details)

            classified_type = error_details.get(
                "classified_type",
                "UnknownError"
            )

            confidence = error_details.get(
                "confidence_score",
                0.0
            )

            root_cause = error_details.get("root_cause")

            kb_info = error_details.get("knowledge_base_info")

            print("🚨 [Execution Error Detected]")

            if root_cause:

                print(
                    f"   File: {root_cause.get('file_normalized')}"
                    f" | Line {root_cause.get('line')}"
                )

                print(
                    f"   Function: {root_cause.get('function')}"
                )

            print(
                f"   Classified: {classified_type}"
                f" (confidence {confidence:.2f})"
            )

            # error severity 기반 penalty
            severity_penalty = {
                "SyntaxError": -2.0,
                "NameError": -3.0,
                "TypeError": -3.5,
                "FileNotFoundError": -2.5,
                "ModuleNotFoundError": -4.0,
                "PermissionError": -5.0
            }

            reward = severity_penalty.get(classified_type, -3.0)

            if kb_info:

                print(
                    f"   KB Causes: {kb_info.get('common_causes')}"
                )

                print(
                    f"   KB Fix Strategies:"
                    f" {kb_info.get('auto_fix_strategies')}"
                )

        else:

            FAILURE_KEYWORDS = [
                "Failure:",
                "Error:",
                "PermissionError:",
                "Security Error:",
                "❌",
                "오류:",
                "실패:"
            ]

            is_failure = any(
                keyword in execution_result_event
                for keyword in FAILURE_KEYWORDS
            )

            if is_failure:

                reward = -4.0

        # success reward
        if not is_failure:

            reward = 1.5

        # RL replay buffer 기록
        if reward != 0.0:

            try:

                next_state = self._get_current_state_vector_sync()

                self.rl_replay_buffer.add(
                    (prev_state, prev_action, reward, next_state, False)
                )

                print(
                    f"📈 [RL Replay] reward={reward:.2f}"
                )

            except Exception as e:

                print(
                    f"⚠️ [RL Replay Error] {e}"
                )

        return reward

    async def start_schedule_monitor(self):

        """
        일정 모니터 시작
        """

        if (
            self._schedule_monitor_task
            and not self._schedule_monitor_task.done()
        ):

            print("⏰ Schedule Monitor 이미 실행 중")
            return

        print(
            f"⏰ Schedule Monitor 시작 "
            f"(interval {self.SCHEDULE_CHECK_INTERVAL_SECONDS}s)"
        )

        loop = asyncio.get_running_loop()

        self._schedule_monitor_task = loop.create_task(
            self._run_schedule_monitor()
        )

    async def stop_schedule_monitor(self):

        """
        일정 모니터 중지
        """

        task = self._schedule_monitor_task

        if not task:

            print("⏰ Schedule Monitor 실행 중 아님")
            return

        if task.done():

            self._schedule_monitor_task = None
            return

        print("⏰ Schedule Monitor 중지 요청")

        task.cancel()

        try:

            await task

        except asyncio.CancelledError:

            print("⏰ Schedule Monitor 정상 종료")

        except Exception as e:

            print(
                f"❌ Schedule Monitor 종료 오류: {e}"
            )

        finally:

            self._schedule_monitor_task = None

    async def get_all_tasks_for_gui(self) -> List[Dict[str, Any]]:

        """
        GUI Task 리스트 반환
        """

        async with self.schedule_lock:

            gui_tasks = []

            for task in self.schedule:

                task_prompt = task.get(
                    "task_prompt",
                    "N/A"
                )

                gui_tasks.append({

                    "name": task_prompt,

                    "due": task.get("time", ""),

                    "status": task.get(
                        "status",
                        "N/A"
                    ),

                    "prompt": task.get(
                        "procedure",
                        ""
                    ),

                    "is_autonomous":
                        task_prompt.startswith(
                            "자율 작업 수행 (트리거:"
                        ),

                    "project_dir":
                        task.get(
                            "project_directory",
                            ""
                        ),

                    "parent_name":
                        task.get(
                            "parent_task_name"
                        )

                })

            return gui_tasks

    async def _retrieve_few_shot_examples_async(
        self,
        current_goal: str,
        chat_history: List[str],
        top_k: int = 2
    ) -> List[str]:

        """
        Few-shot retrieval engine
        """

        if not self.strategy_log:

            return [
                "External Example Check:"
                " No past examples available."
            ]

        query = (
            current_goal.lower()
            + " "
            + " ".join(self._normalize_history(chat_history)[-3:])
        )

        query_tokens = query.split()

        scored_examples = []

        for event_id, data in self.strategy_log.items():

            action = data.get("action", "")

            reward = data.get("reward", 0.0)

            action_lower = action.lower()

            relevance = sum(
                token in action_lower
                for token in query_tokens
            )

            if relevance == 0:
                continue

            score = relevance + abs(reward)

            outcome = "Success" if reward > 0 else "Failure"

            example_text = f"""
[Past Task]: Goal related to '{current_goal[:25]}'
[Action]: {action[:80]}
[Outcome]: {outcome} (Reward {reward:.2f})
"""

            scored_examples.append(
                (score, example_text.strip())
            )

        if not scored_examples:

            return [
                "External Example Check:"
                " No relevant past examples."
            ]

        scored_examples.sort(
            key=lambda x: x[0],
            reverse=True
        )

        return [
            ex[1]
            for ex in scored_examples[:top_k]
        ]

    

    async def _simulate_plan_with_world_model_async(self, plan: List[Dict]) -> Dict:
        """
        [Phase 1] WorldState.simulate_plan()으로 계획 위험도 추정.
        WorldModel이 없으면 self_model 기반 fallback.

        [Phase B-1] ValueEngine 으로 ROI(reward−cost−risk)를 함께 계산해
        반환 dict 에 {roi, value_score, value_breakdown, roi_threshold}를 덧붙인다.
        기존 호출자는 그대로 동작; ROI 를 쓰고 싶은 상위만 꺼내 사용.
        """
        if self.world_model:
            result = self.world_model.simulate_plan(plan)
            print(f"  🔮 [WorldSim] success_prob={result['success_prob']:.2f} "
                  f"risk={result['predicted_risk']} "
                  f"loops={result.get('feedback_loops', 0)}")

            # ── [DecisionRouter] WM·CM 평행 추론 통합 (A'안 쌍 3) ──────────
            # WM = ground truth, CM = trained 시 보조 가중. 둘의 의견 일치도로
            # verdict(accelerate/normal/explore)를 부여한다. 차단 없음.
            try:
                from eidos_decision_router import get_router
                import hashlib, json as _rr_json
                cm_trainer = getattr(self, "_causal_trainer", None)
                cm_x_values = None
                if cm_trainer is not None and getattr(cm_trainer, "x_vars", None):
                    cm_x_values = [
                        float(getattr(v, "current_value", 0.0))
                        for v in cm_trainer.x_vars
                    ]
                _sig = hashlib.md5(
                    _rr_json.dumps(
                        [(s.get("tool", ""), str(s.get("args", {}))[:60]) for s in plan]
                    ).encode()
                ).hexdigest()[:12]
                _causal = get_router().route_causal_query(
                    wm_result=result,
                    cm_trainer=cm_trainer,
                    cm_x_values=cm_x_values,
                    plan_signature=_sig,
                )
                # WM 원본은 보존, success_prob만 merged로 가중 갱신
                result["success_prob_wm"] = result["success_prob"]
                result["success_prob"] = _causal.merged_success_prob
                result["causal_verdict"] = _causal.verdict
                result["causal_agreement"] = _causal.agreement
                if _causal.cm_result:
                    result["cm_result"] = _causal.cm_result
                if _causal.verdict == "accelerate":
                    print(f"  🟢 [Router] WM·CM 일치({_causal.agreement:.2f}) → 가속 후보")
                elif _causal.verdict == "explore":
                    print(f"  🟡 [Router] WM·CM 불일치({_causal.agreement:.2f}) → 탐색 신호")
            except Exception as _rr_e:
                print(f"  [Router] causal 통합 실패 (무시): {_rr_e}")
        else:
            # fallback: self_model 성공률 기반 (기존 로직)
            risk_score = 0.0
            for step in plan:
                tool_name = step.get("tool", "unknown")
                risk_score += (1.0 - self.self_model.get_success_rate(tool_name))
            success_prob   = 1.0 - (risk_score / len(plan)) if plan else 1.0
            predicted_risk = "high" if risk_score > 1.0 else "medium" if risk_score > 0.5 else "low"
            result = {
                "success_prob":   success_prob,
                "predicted_risk": predicted_risk,
                "risk_score":     max(0.0, min(1.0, risk_score / max(1, len(plan)))),
                "reason":         f"self_model fallback (risk={risk_score:.2f})",
            }

        # ── [Phase B-1 + A'안 B] ValueEngine ROI + reasoning 신호 통합 ───────
        try:
            from eidos_value_engine import score_plan, roi_threshold as _roi_thr
            _ctx = {
                "goal":        getattr(self, "current_goal", "") or "",
                "task_prompt": getattr(self, "_last_user_input", "") or "",
            }

            # ── [B안] reasoning 신호 수집 → ValueEngine extra_signals ──
            # 1) router.causal_verdict (이미 result에 주입됨)
            # 2) WorldModel counterfactual (enhanced_simulate_plan 결과)
            # 3) Self-DNA cached gap (있으면)
            _signals: Dict[str, Any] = {}
            try:
                if "causal_verdict" in result:
                    _signals["causal_verdict"]   = result["causal_verdict"]
                    _signals["causal_agreement"] = float(result.get("causal_agreement", 1.0))

                # counterfactual: removal_analysis는 enhanced_simulate_plan 결과에 포함
                _ra = result.get("removal_analysis") or []
                if isinstance(_ra, list):
                    _rm = sum(1 for it in _ra if isinstance(it, dict) and it.get("is_removable"))
                    if _rm:
                        _signals["removable_steps"] = _rm

                _alts = result.get("alternatives") or []
                if isinstance(_alts, list) and _alts:
                    _signals["alternatives_n"] = len(_alts)

                # Self-DNA cached gap (있으면) — 가장 큰 모듈 gap을 dna_distance로 사용
                _self_dna = getattr(self, "_self_dna_cache", None)
                if _self_dna is not None:
                    _gaps = getattr(_self_dna, "gaps", None) or {}
                    if isinstance(_gaps, dict) and _gaps:
                        try:
                            _max_gap = max(float(v) for v in _gaps.values())
                            _signals["dna_distance"] = max(0.0, min(1.0, _max_gap))
                        except Exception:
                            pass
            except Exception as _sg_e:
                print(f"  [B안] 신호 수집 실패 (무시): {_sg_e}")

            _vs = score_plan(plan, sim_result=result, context=_ctx, extra_signals=_signals or None)
            result["value_score"]     = _vs.as_dict()
            result["roi"]             = _vs.roi
            result["roi_threshold"]   = _roi_thr()
            result["value_breakdown"] = _vs.breakdown
            _sig_note = ""
            if _signals:
                _sig_note = f" sig={list(_signals.keys())}"
            print(f"  💰 [ValueEngine] ROI={_vs.roi:+.3f} "
                  f"(reward={_vs.reward:.2f} cost={_vs.cost:.2f} risk={_vs.risk:.2f}) "
                  f"threshold={_roi_thr():+.2f}{_sig_note}")
        except Exception as _ve:
            print(f"  [ValueEngine] ROI 계산 실패 (무시): {_ve}")

        return result

    # ══════════════════════════════════════════════════════════════════
    # 🧠 구조적 반복 추론 엔진 (코딩 전용)
    # ══════════════════════════════════════════════════════════════════

    def _is_coding_request(self, text: str) -> bool:
        """사용자 입력이 코딩 요청인지 판별합니다."""
        keywords = [
            "코드", "함수", "클래스", "구현", "작성해", "짜줘", "짜봐",
            "스크립트", "파이썬", "python", "def ", "class ",
            "알고리즘", "로직", "모듈", "API", "서버", "크롤러", "파싱",
            "버그", "오류", "수정해", "고쳐", "디버그", "리팩토링",
            "테스트", "기능 추가", "기능추가",
        ]
        t = text.lower()
        return any(k.lower() in t for k in keywords)

    def _is_tool_creation_request(self, text: str) -> bool:
        """사용자가 새 도구/기능을 EIDOS에게 직접 만들어달라고 요청하는지 판별합니다.
        
        '도구를 만들어', '기능을 추가해', '~하는 도구 만들어서 실행해봐' 등을 감지.
        _is_coding_request와 다른 점: 단순 코드 작성이 아니라
        EIDOS 자신의 능력을 확장하는 도구 합성 요청을 구분함.
        """
        t = text.lower()
        # 도구/기능 대상어
        target_kws = ["도구", "기능", "툴", "tool", "함수", "function"]
        # 생성/적용 동사
        action_kws = [
            "만들어", "만들어서", "만들어줘", "만들어봐",
            "추가해", "추가해줘", "추가해봐",
            "적용해", "적용해줘", "적용해봐",
            "개발해", "구현해", "생성해",
        ]
        has_target = any(kw in t for kw in target_kws)
        has_action = any(kw in t for kw in action_kws)
        if has_target and has_action:
            return True
        # 복합 패턴: '~을 할 수 있는 도구', '~기능 추가'
        compound_patterns = [
            "도구 만들", "도구를 만들", "기능 추가", "기능을 추가",
            "툴 만들", "툴을 만들", "tool 만들",
            "도구 생성", "도구를 생성", "새 도구", "새로운 도구",
        ]
        return any(p in t for p in compound_patterns)

    async def _deep_reason_for_coding_async(
        self,
        user_request: str,
        chat_history: list,
        tools_list_str: str,
        project_context: str = "",
    ) -> list:
        """
        [구조적 반복 추론 엔진 v1.0] 코딩 요청 전용 5단계 추론 파이프라인.
        Phase1→해석 / Phase2→체크리스트 / Phase3→재검토 / Phase4→실행계획 / Phase5→QA검증
        """
        history_str = "\n".join(chat_history[-5:]) if chat_history else ""

        # ── 복잡도 판별 → 경로 결정 ───────────────────────────────────
        complexity = self._classify_request_complexity(user_request)
        if complexity == "simple":
            print("⚡ [DeepReason] 단순 요청 감지 → 단축 경로 (Phase1→4→5)")
        else:
            print("🔁 [DeepReason] 복잡 요청 감지 → 풀 경로 (Phase1→2→3→4→5)")

        # ── Phase 1: 상황 해석 (공통) ────────────────────────────────
        p1 = await get_llm_response_async(
            f"""시니어 소프트웨어 아키텍트로서 아래 요청을 분석하세요.
[대화 맥락] {history_str}
[요청] {user_request}
분석: 1)핵심 목표 2)기술스택 3)제약조건 4)예상 입출력 5)난이도 요소"""
        )
        if not p1:
            print("⚠️ [DeepReason] Phase1 실패 → Fallback")
            return []
        print(f"✅ [DeepReason Phase1] 상황 해석 완료 ({len(p1)}c)")

        if complexity == "simple":
            # ── 단축 경로: Phase2·3 생략, p1 요약을 p3로 바로 사용 ──
            # Phase1 결과에서 핵심만 압축하여 다음 Phase에 전달
            p1_summary = p1[:600] if len(p1) > 600 else p1
            p3 = f"[Phase1 요약]\n{p1_summary}"
            print(f"⚡ [DeepReason] Phase2·3 스킵 — 압축 컨텍스트 사용 ({len(p3)}c)")
        else:
            # ── 풀 경로: Phase 2·3 실행 ──────────────────────────────
            # Phase 2: 체크리스트
            # 토큰 절약: p1 원문 대신 500자 압축본만 전달
            p1_summary = p1[:500] if len(p1) > 500 else p1
            p2 = await get_llm_response_async(
                f"""아래 분석을 바탕으로 구현 체크리스트를 5~8개 작성하세요.
[요청] {user_request}
[Phase1 분석 요약] {p1_summary}
각 항목 형식: [카테고리] 내용 — 이유"""
            ) or p1
            print(f"✅ [DeepReason Phase2] 체크리스트 완료 ({len(p2)}c)")

            # Phase 3: 재검토 (엣지케이스 + 누락 점검)
            # 토큰 절약: p2 원문 대신 800자 압축본만 전달 (p1_summary는 위에서 이미 생성)
            p2_summary = p2[:800] if len(p2) > 800 else p2
            p3 = await get_llm_response_async(
                f"""체크리스트를 검토하여 누락·충돌·엣지케이스를 보완하세요.
[요청] {user_request}
[체크리스트] {p2_summary}
검토 항목: 예외처리 누락? 보안 이슈? 의존성 순서? 엣지케이스?
누락된 항목만 추가하여 최종 체크리스트를 반환하세요."""
            ) or p2
            print(f"✅ [DeepReason Phase3] 재검토 완료 ({len(p3)}c)")

        # Phase 4: JSON 실행 계획 확정
        # 토큰 절약: p3 원문 대신 1000자 압축본 전달
        p3_summary = p3[:1000] if len(p3) > 1000 else p3
        p4 = await get_llm_response_async(
            f"""EIDOS 실행 플래너로서 아래 체크리스트를 실행 계획 JSON으로 변환하세요.
[요청] {user_request}
[최종 체크리스트] {p3_summary}
[도구 목록] {tools_list_str}
[프로젝트 컨텍스트] {project_context}

[출력 규칙 - 반드시 준수]
1. JSON 배열만 출력. 다른 텍스트 절대 금지.
2. 각 항목 형식:
   {{"tool": "도구명", "args": {{인수}}, "description": "한 줄 설명"}}
3. "tool" 키는 반드시 소문자 영어로 된 도구명 (예: "write_project_files_async")
4. 코드 생성 시 반드시 write_project_files_async 사용:
   args 형식: {{"file_structure": {{"파일경로": "파일내용"}}, "project_dir": "폴더명"}}
   ※ "file_structure_json" 아님, 반드시 "file_structure" 키 사용
5. file_structure 안에 실제 동작하는 완전한 코드를 작성

[올바른 예시]
[
  {{
    "tool": "write_project_files_async",
    "args": {{
      "file_structure": {{
        "bubble_sort/main.py": "def bubble_sort(arr):\\n    ..."
      }},
      "project_dir": "bubble_sort"
    }},
    "description": "버블정렬 코드 파일 생성"
  }}
]""",
            response_mime_type="application/json"
        )
        print(f"✅ [DeepReason Phase4] 실행 계획 확정 ({len(str(p4))}c)")

        # Phase 5: QA 자가검증 — 강제 FAIL 조건 명시
        p5 = await get_llm_response_async(
            f"""당신은 엄격한 QA 엔지니어입니다. 아래 실행 계획을 체크리스트 항목별로 검증하세요.

[원본 요청]
{user_request}

[검증 기준 체크리스트]
{p3_summary}

[검증 대상 계획]
{p4}

[검증 규칙 — 반드시 준수]
1. 체크리스트 각 항목을 순서대로 검토하여 계획에 반영됐는지 확인하세요.
2. 아래 중 하나라도 해당하면 반드시 FAIL을 반환하세요:
   - 체크리스트 항목이 계획에 1개 이상 누락된 경우
   - 도구 이름·args 키가 잘못된 경우 (예: file_structure_json 사용)
   - 코드가 요청 기능을 완전히 구현하지 못한 경우
   - 계획 단계가 0개이거나 비어 있는 경우
3. 모든 항목이 충족된 경우에만 PASS를 반환하세요.
4. FAIL 시 반드시 수정사항을 구체적으로 명시하세요.

[출력 형식]
PASS
또는
FAIL: <항목번호> <누락/오류 내용> | <항목번호> <누락/오류 내용> ..."""
        )
        print(f"✅ [DeepReason Phase5] QA 검증: {str(p5)[:120]}")

        # ── QA FAIL → 자동 수정 → 재QA 루프 (최대 2회) ──────────────
        _MAX_QA_RETRY = 2
        for _qa_attempt in range(_MAX_QA_RETRY):
            if not (p5 and "FAIL" in str(p5)):
                break  # PASS면 즉시 탈출

            print(f"🔧 [DeepReason] QA FAIL (시도 {_qa_attempt + 1}/{_MAX_QA_RETRY}) → 계획 보정")

            # 계획 보정
            p4 = await get_llm_response_async(
                f"""QA 검증에서 FAIL이 반환됐습니다. 지적사항을 반영하여 계획을 수정하세요.

[원본 요청]
{user_request}

[기존 계획]
{p4}

[QA 지적사항]
{p5}

[수정 규칙]
1. 지적된 누락 항목을 계획에 추가하세요.
2. 잘못된 도구명·args 키는 올바른 값으로 교체하세요.
3. 기존 계획에서 문제없는 단계는 그대로 유지하세요.
4. 반드시 JSON 배열만 출력. 설명 텍스트 금지.""",
                response_mime_type="application/json"
            )

            if not p4:
                print("⚠️ [DeepReason] 계획 보정 실패 → QA 루프 중단")
                break

            # 재QA
            p5 = await get_llm_response_async(
                f"""당신은 엄격한 QA 엔지니어입니다. 아래 실행 계획을 체크리스트 항목별로 검증하세요.

[원본 요청]
{user_request}

[검증 기준 체크리스트]
{p3_summary}

[검증 대상 계획]
{p4}

[검증 규칙 — 반드시 준수]
1. 체크리스트 각 항목을 순서대로 검토하여 계획에 반영됐는지 확인하세요.
2. 아래 중 하나라도 해당하면 반드시 FAIL을 반환하세요:
   - 체크리스트 항목이 계획에 1개 이상 누락된 경우
   - 도구 이름·args 키가 잘못된 경우 (예: file_structure_json 사용)
   - 코드가 요청 기능을 완전히 구현하지 못한 경우
   - 계획 단계가 0개이거나 비어 있는 경우
3. 모든 항목이 충족된 경우에만 PASS를 반환하세요.
4. FAIL 시 반드시 수정사항을 구체적으로 명시하세요.

[출력 형식]
PASS
또는
FAIL: <항목번호> <누락/오류 내용> | <항목번호> <누락/오류 내용> ..."""
            )
            print(f"🔁 [DeepReason] 재QA 결과 (시도 {_qa_attempt + 1}): {str(p5)[:120]}")

        if p5 and "FAIL" in str(p5):
            print(f"⚠️ [DeepReason] QA {_MAX_QA_RETRY}회 재시도 후에도 FAIL — 현재 계획으로 진행")

        # 파싱
        try:
            import json as _j
            if isinstance(p4, list):
                plan = p4
            elif isinstance(p4, str):
                plan = _j.loads(p4.strip().replace("```json","").replace("```","").strip())
            elif isinstance(p4, dict):
                plan = p4.get("plan", p4.get("Plan A", p4.get("steps", [])))
                if not plan:  # dict 자체가 단일 step일 수도
                    plan = [p4]
            else:
                plan = []

            # plan 항목 정규화: tool 키 없으면 다른 키에서 보정
            normalized = []
            for item in plan:
                if not isinstance(item, dict):
                    continue
                if "tool" not in item:
                    item["tool"] = (
                        item.get("name") or item.get("function") or
                        item.get("action") or item.get("tool_name") or ""
                    )
                if "args" not in item:
                    item["args"] = item.get("arguments", item.get("params", {}))
                if item.get("tool"):
                    normalized.append(item)

            plan = normalized if normalized else plan

            if isinstance(plan, list) and plan:
                print(f"🎯 [DeepReason] 최종 계획 {len(plan)}단계 확정")
                return plan
        except Exception as e:
            print(f"⚠️ [DeepReason] 파싱 실패: {e}")
        return []

    # ══════════════════════════════════════════════════════════════════
    # 🔧 플래너 헬퍼 메서드
    # ══════════════════════════════════════════════════════════════════

    def _normalize_input(self, text_input) -> str:
        """입력값을 문자열로 정규화합니다."""
        if isinstance(text_input, str):
            return text_input
        if isinstance(text_input, dict):
            return text_input.get("text", str(text_input))
        if isinstance(text_input, list):
            return " ".join(str(x) for x in text_input)
        return str(text_input)

    def _classify_request_complexity(self, text: str) -> str:
        """
        요청 복잡도를 판별하여 DeepReason 경로를 결정합니다.

        Returns:
            "simple"  → Phase1 → Phase4 → Phase5 (단축 경로)
            "complex" → Phase1 → Phase2 → Phase3 → Phase4 → Phase5 (풀 경로)
        """
        t = text.lower()

        # ── 복잡 요청 시그널 ──────────────────────────────────────────
        # 이 키워드가 있으면 무조건 풀 DeepReason
        complex_signals = [
            # 설계/구조 키워드
            "아키텍처", "설계", "시스템", "프레임워크", "구조",
            # 다중 컴포넌트
            "모듈", "클래스", "api", "서버", "데이터베이스", "db",
            # 복잡 기능
            "크롤러", "파싱", "비동기", "async", "멀티스레드",
            "머신러닝", "딥러닝", "모델", "학습",
            # 수정/리팩토링 (기존 맥락 필요)
            "리팩토링", "리팩터", "전체", "전반", "개선",
            # 규모 암시
            "프로젝트", "앱", "application", "app",
            # 연동/통합
            "연동", "통합", "integration", "webhook",
        ]
        if any(sig in t for sig in complex_signals):
            return "complex"

        # ── 단순 요청 시그널 ──────────────────────────────────────────
        # 아래 패턴에만 해당하고 복잡 시그널이 없으면 단순 경로
        simple_signals = [
            # 단일 함수/알고리즘
            "버블정렬", "정렬", "sort", "검색", "search",
            "피보나치", "팩토리얼", "재귀", "반복문", "for문", "while문",
            # 간단한 코드 조각
            "함수 만들어", "함수 짜", "코드 짜줘", "코드 작성",
            "예제", "샘플", "sample", "example",
            # 단순 수정
            "버그", "오류", "에러", "고쳐", "수정해",
            # 계산/변환
            "계산", "변환", "convert", "parse",
        ]
        # 단순 시그널이 있고 요청이 짧으면(100자 이하) 단순으로 판정
        if any(sig in t for sig in simple_signals) and len(text) <= 150:
            return "simple"

        # ── 길이 기반 휴리스틱 ────────────────────────────────────────
        # 요청이 매우 짧으면 단순, 길면 복잡
        if len(text) <= 80:
            return "simple"
        if len(text) >= 300:
            return "complex"

        # 기본값: 복잡 (안전 우선)
        return "complex"

    def _structural_bypass(self, text: str):
        """단순 입력 바이패스 — 복잡도 판별 후 None 반환 (실제 분기는 DeepReason 내부에서 처리)."""
        return None

    def _build_state_vector(self):
        """현재 감정 상태 벡터를 반환합니다."""
        return self.emotion_state.get_vector()

    def _resolve_goal(self, text_input, is_user_task: bool) -> str:
        """현재 목표를 결정합니다."""
        if is_user_task:
            return self._normalize_input(text_input)
        return self.current_goal or self._normalize_input(text_input)

    def _load_user_memory(self) -> str:
        """KB 사용자 상태를 텍스트로 로드합니다."""
        try:
            state = self.kb.user_state
            if not state:
                return ""
            important = {k: v for k, v in state.items()
                         if k in ("status", "current_task", "intent", "goal", "current_request")}
            return "\n".join(f"- {k}: {v}" for k, v in important.items())
        except Exception:
            return ""

    def _parse_multimodal(self, multimodal_vector) -> str:
        """멀티모달 벡터를 텍스트로 변환합니다."""
        if multimodal_vector is None:
            return ""
        try:
            import numpy as np
            if isinstance(multimodal_vector, np.ndarray) and multimodal_vector.any():
                return f"[멀티모달 입력: shape={multimodal_vector.shape}]"
        except Exception:
            pass
        return ""

    def _reflection_processor(self, text_input, text_input_str: str):
        """반성 컨텍스트 생성 (직전 작업 요약 주입)."""
        reflection = ""
        try:
            if self.reflection_queue:
                recent = self.reflection_queue[-1]
                task_name = recent.get("task_prompt", "")
                status    = recent.get("status", "")
                if task_name:
                    reflection = f"[직전 작업] '{task_name}' → {status}"
        except Exception:
            pass
        return reflection, text_input_str

    def _qa_guard(self, text: str) -> str:
        """플래너에게 JSON 출력 형식과 실행 우선 원칙을 강제하는 지시문 반환."""
        return (
            "\n[플래너 출력 규칙]\n"
            "반드시 JSON 배열만 출력. 설명 텍스트·마크다운 금지.\n"
            "형식: [{\"tool\": \"도구명\", \"args\": {...}, \"description\": \"설명\"}]\n"
            "\n[질문 루프 완전 차단]\n"
            "ask_user_for_clarification 도구는 존재하지 않는다. 절대 사용 금지.\n"
            "요청이 모호해도 합리적 기본값으로 즉시 실행하라.\n"
            "계획의 첫 단계는 반드시 write_*, perform_web_search, execute_* 중 하나여야 한다.\n"
        )

    # ══════════════════════════════════════════════════════════════════
    # 📅 Google Calendar 툴 메서드
    # ══════════════════════════════════════════════════════════════════

    def _get_gcal_service(self):
        """
        Google Calendar API 서비스 객체를 반환합니다.
        credentials.json → OAuth 인증 → token.json 자동 저장.
        gcal_config가 없거나 credentials_path가 비어 있으면 None 반환.
        """
        gcal_cfg = getattr(self, "gcal_config", {})
        if not gcal_cfg.get("enabled") or not gcal_cfg.get("credentials_path"):
            return None, "Google Calendar 커넥터가 비활성화되어 있거나 credentials.json 경로가 없습니다."

        try:
            from google.oauth2.credentials import Credentials
            from google_auth_oauthlib.flow import InstalledAppFlow
            from google.auth.transport.requests import Request
            from googleapiclient.discovery import build
            import os

            SCOPES = ["https://www.googleapis.com/auth/calendar"]
            cred_path  = gcal_cfg["credentials_path"]
            token_path = gcal_cfg.get("token_path", "eidos_files/gcal_token.json")

            creds = None
            if os.path.exists(token_path):
                creds = Credentials.from_authorized_user_file(token_path, SCOPES)

            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                else:
                    flow = InstalledAppFlow.from_client_secrets_file(cred_path, SCOPES)
                    creds = flow.run_local_server(port=0)
                os.makedirs(os.path.dirname(token_path), exist_ok=True)
                with open(token_path, "w") as f:
                    f.write(creds.to_json())

            service = build("calendar", "v3", credentials=creds)
            return service, None

        except ImportError:
            return None, "google-api-python-client 패키지가 없습니다. pip install google-api-python-client google-auth-oauthlib 실행하세요."
        except Exception as e:
            return None, f"Google Calendar 인증 오류: {e}"

    async def gcal_list_events(self, date_str: str = None, max_results: int = 10) -> list:
        """
        Google Calendar에서 일정을 조회합니다.
        date_str: "yyyy-MM-dd" 형식. None이면 오늘.
        반환: [{"time": "HH:mm", "text": "제목", "gcal_id": "...", "date": "yyyy-MM-dd"}, ...]
        """
        import asyncio
        from datetime import datetime, timezone, timedelta

        def _sync_list():
            service, err = self._get_gcal_service()
            if err:
                return [], err

            gcal_cfg   = getattr(self, "gcal_config", {})
            calendar_id = gcal_cfg.get("calendar_id", "primary")
            target_date = date_str or datetime.now().strftime("%Y-%m-%d")

            try:
                dt = datetime.strptime(target_date, "%Y-%m-%d")
                time_min = dt.replace(tzinfo=timezone.utc).isoformat()
                time_max = (dt + timedelta(days=1)).replace(tzinfo=timezone.utc).isoformat()

                result = service.events().list(
                    calendarId=calendar_id,
                    timeMin=time_min,
                    timeMax=time_max,
                    maxResults=max_results,
                    singleEvents=True,
                    orderBy="startTime"
                ).execute()

                events = []
                for e in result.get("items", []):
                    start = e.get("start", {})
                    start_str = start.get("dateTime", start.get("date", ""))
                    # 시간 파싱
                    try:
                        if "T" in start_str:
                            t = datetime.fromisoformat(start_str.replace("Z", "+00:00"))
                            time_part = t.strftime("%H:%M")
                        else:
                            time_part = "00:00"
                    except Exception:
                        time_part = "00:00"

                    events.append({
                        "time":    time_part,
                        "text":    e.get("summary", "(제목 없음)"),
                        "gcal_id": e.get("id", ""),
                        "date":    target_date,
                    })
                return events, None

            except Exception as e:
                return [], f"일정 조회 오류: {e}"

        events, err = await asyncio.get_event_loop().run_in_executor(None, _sync_list)
        if err:
            print(f"❌ [GCal] 조회 실패: {err}")
        return events

    async def gcal_create_event(self, date_str: str, time_str: str, summary: str,
                                 description: str = "", duration_minutes: int = 60) -> str:
        """
        Google Calendar에 일정을 생성합니다.
        반환: 생성된 이벤트 ID (실패 시 빈 문자열)
        """
        import asyncio
        from datetime import datetime, timezone, timedelta

        def _sync_create():
            service, err = self._get_gcal_service()
            if err:
                return "", err

            gcal_cfg    = getattr(self, "gcal_config", {})
            calendar_id = gcal_cfg.get("calendar_id", "primary")

            try:
                start_dt = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")
                end_dt   = start_dt + timedelta(minutes=duration_minutes)

                # KST(+09:00) 기준으로 생성
                tz_offset = timezone(timedelta(hours=9))
                start_iso = start_dt.replace(tzinfo=tz_offset).isoformat()
                end_iso   = end_dt.replace(tzinfo=tz_offset).isoformat()

                event_body = {
                    "summary":     summary,
                    "description": description,
                    "start": {"dateTime": start_iso, "timeZone": "Asia/Seoul"},
                    "end":   {"dateTime": end_iso,   "timeZone": "Asia/Seoul"},
                }
                created = service.events().insert(
                    calendarId=calendar_id, body=event_body
                ).execute()
                return created.get("id", ""), None

            except Exception as e:
                return "", f"일정 생성 오류: {e}"

        event_id, err = await asyncio.get_event_loop().run_in_executor(None, _sync_create)
        if err:
            print(f"❌ [GCal] 생성 실패: {err}")
            return ""
        print(f"✅ [GCal] 일정 생성 완료: '{summary}' (id={event_id})")
        return event_id

    async def gcal_delete_event(self, gcal_id: str) -> bool:
        """
        Google Calendar에서 이벤트 ID로 일정을 삭제합니다.
        반환: 성공 True / 실패 False
        """
        import asyncio

        if not gcal_id:
            return False

        def _sync_delete():
            service, err = self._get_gcal_service()
            if err:
                return False, err

            gcal_cfg    = getattr(self, "gcal_config", {})
            calendar_id = gcal_cfg.get("calendar_id", "primary")

            try:
                service.events().delete(
                    calendarId=calendar_id, eventId=gcal_id
                ).execute()
                return True, None
            except Exception as e:
                return False, f"일정 삭제 오류: {e}"

        success, err = await asyncio.get_event_loop().run_in_executor(None, _sync_delete)
        if err:
            print(f"❌ [GCal] 삭제 실패: {err}")
        else:
            print(f"✅ [GCal] 일정 삭제 완료 (id={gcal_id})")
        return success

    # ══════════════════════════════════════════════════════════════════
    # 📧 Gmail 툴 메서드 (OAuth — gcal과 동일 패턴, gmail.modify 스코프)
    #    읽기·검색·보내기·답장초안·라벨·보관·휴지통을 모두 커버합니다.
    # ══════════════════════════════════════════════════════════════════

    def _get_gmail_service(self):
        """
        Gmail API 서비스 객체를 반환합니다. (service, None) / (None, 오류문자열).
        credentials.json → OAuth 인증 → gmail_token.json 자동 저장.
        gmail_config가 없거나 credentials_path가 비면 오류 반환.
        """
        gmail_cfg = getattr(self, "gmail_config", {})
        if not gmail_cfg.get("enabled") or not gmail_cfg.get("credentials_path"):
            return None, "Gmail 커넥터가 비활성화되어 있거나 credentials.json 경로가 없습니다."
        try:
            from google.oauth2.credentials import Credentials
            from google_auth_oauthlib.flow import InstalledAppFlow
            from google.auth.transport.requests import Request
            from googleapiclient.discovery import build
            import os

            # modify = 읽기 + 보내기 + 라벨변경 + 휴지통(보관/삭제). 영구삭제·설정은 제외.
            SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]
            cred_path  = gmail_cfg["credentials_path"]
            token_path = gmail_cfg.get("token_path", "eidos_files/gmail_token.json")

            creds = None
            if os.path.exists(token_path):
                creds = Credentials.from_authorized_user_file(token_path, SCOPES)
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                else:
                    flow = InstalledAppFlow.from_client_secrets_file(cred_path, SCOPES)
                    creds = flow.run_local_server(port=0)
                os.makedirs(os.path.dirname(token_path) or ".", exist_ok=True)
                with open(token_path, "w") as f:
                    f.write(creds.to_json())

            service = build("gmail", "v1", credentials=creds)
            return service, None
        except ImportError:
            return None, ("google-api-python-client 패키지가 없습니다. "
                          "pip install google-api-python-client google-auth-oauthlib 실행하세요.")
        except Exception as e:
            return None, f"Gmail 인증 오류: {e}"

    @staticmethod
    def _gmail_header(headers: list, name: str) -> str:
        """payload headers 리스트에서 헤더값(대소문자 무시)을 꺼냅니다."""
        nl = (name or "").lower()
        for h in headers or []:
            if (h.get("name", "")).lower() == nl:
                return h.get("value", "")
        return ""

    @staticmethod
    def _gmail_extract_body(payload: dict) -> str:
        """Gmail message payload(멀티파트 가능)에서 사람이 읽을 본문 텍스트를 추출합니다."""
        import base64

        def _decode(data: str) -> str:
            try:
                return base64.urlsafe_b64decode(data.encode("utf-8")).decode("utf-8", "replace")
            except Exception:
                return ""

        def _walk(part) -> str:
            mime = part.get("mimeType", "")
            body = part.get("body", {}) or {}
            data = body.get("data")
            if mime == "text/plain" and data:
                return _decode(data)
            # 멀티파트 → 자식 우선 탐색(plain 우선, 없으면 html 폴백)
            html_fallback = ""
            for sub in part.get("parts", []) or []:
                got = _walk(sub)
                if got and sub.get("mimeType") == "text/plain":
                    return got
                if got and not html_fallback:
                    html_fallback = got
            if not data:
                return html_fallback
            if mime == "text/html" and data:
                import re as _re
                raw = _decode(data)
                return _re.sub(r"<[^>]+>", " ", raw)   # 태그 제거(간단)
            return html_fallback

        text = _walk(payload or {})
        return (text or "").strip()

    @staticmethod
    def _gmail_build_raw(to: str, subject: str, body: str,
                         sender: str = "me", in_reply_to: str = "") -> str:
        """MIME 메일을 base64url(raw) 문자열로 만듭니다."""
        import base64
        from email.mime.text import MIMEText
        msg = MIMEText(body or "", "plain", "utf-8")
        msg["To"] = to
        if sender and sender != "me":
            msg["From"] = sender
        msg["Subject"] = subject or ""
        if in_reply_to:
            msg["In-Reply-To"] = in_reply_to
            msg["References"]  = in_reply_to
        return base64.urlsafe_b64encode(msg.as_bytes()).decode("utf-8")

    async def gmail_list_messages(self, query: str = "", max_results: int = 10,
                                  unread_only: bool = False) -> list:
        """
        받은편지함/검색 결과를 요약 리스트로 반환합니다.
        query: Gmail 검색 문법(예: "from:foo@bar.com newer_than:7d"). 빈값이면 전체.
        반환: [{"id","thread_id","from","subject","date","snippet","unread"}, ...]
        """
        import asyncio

        def _sync():
            service, err = self._get_gmail_service()
            if err:
                return [], err
            try:
                q = (query or "").strip()
                if unread_only:
                    q = (q + " is:unread").strip()
                resp = service.users().messages().list(
                    userId="me", q=q or None, maxResults=max_results).execute()
                out = []
                for ref in resp.get("messages", []) or []:
                    mid = ref.get("id")
                    msg = service.users().messages().get(
                        userId="me", id=mid, format="metadata",
                        metadataHeaders=["From", "Subject", "Date"]).execute()
                    headers = (msg.get("payload", {}) or {}).get("headers", [])
                    labels  = msg.get("labelIds", []) or []
                    out.append({
                        "id":        mid,
                        "thread_id": msg.get("threadId", ""),
                        "from":      self._gmail_header(headers, "From"),
                        "subject":   self._gmail_header(headers, "Subject") or "(제목 없음)",
                        "date":      self._gmail_header(headers, "Date"),
                        "snippet":   msg.get("snippet", ""),
                        "unread":    "UNREAD" in labels,
                    })
                return out, None
            except Exception as e:
                return [], f"메일 조회 오류: {e}"

        items, err = await asyncio.get_event_loop().run_in_executor(None, _sync)
        if err:
            print(f"❌ [Gmail] 조회 실패: {err}")
        return items

    async def gmail_get_message(self, message_id: str) -> dict:
        """단일 메일의 전체 본문을 반환합니다. {"id","thread_id","from","to","subject","date","body"}."""
        import asyncio
        if not message_id:
            return {}

        def _sync():
            service, err = self._get_gmail_service()
            if err:
                return {}, err
            try:
                msg = service.users().messages().get(
                    userId="me", id=message_id, format="full").execute()
                payload = msg.get("payload", {}) or {}
                headers = payload.get("headers", [])
                return {
                    "id":        message_id,
                    "thread_id": msg.get("threadId", ""),
                    "from":      self._gmail_header(headers, "From"),
                    "to":        self._gmail_header(headers, "To"),
                    "subject":   self._gmail_header(headers, "Subject") or "(제목 없음)",
                    "date":      self._gmail_header(headers, "Date"),
                    "message_id_header": self._gmail_header(headers, "Message-ID"),
                    "body":      self._gmail_extract_body(payload) or msg.get("snippet", ""),
                }, None
            except Exception as e:
                return {}, f"메일 읽기 오류: {e}"

        data, err = await asyncio.get_event_loop().run_in_executor(None, _sync)
        if err:
            print(f"❌ [Gmail] 읽기 실패: {err}")
        return data

    async def gmail_send(self, to: str, subject: str, body: str,
                         in_reply_to: str = "", thread_id: str = "") -> str:
        """메일을 발송합니다. 반환: 발송된 메시지 ID(실패 시 빈 문자열). thread_id 주면 답장으로 묶음."""
        import asyncio

        def _sync():
            service, err = self._get_gmail_service()
            if err:
                return "", err
            try:
                raw = self._gmail_build_raw(to, subject, body, in_reply_to=in_reply_to)
                send_body = {"raw": raw}
                if thread_id:
                    send_body["threadId"] = thread_id
                sent = service.users().messages().send(userId="me", body=send_body).execute()
                return sent.get("id", ""), None
            except Exception as e:
                return "", f"메일 발송 오류: {e}"

        mid, err = await asyncio.get_event_loop().run_in_executor(None, _sync)
        if err:
            print(f"❌ [Gmail] 발송 실패: {err}")
            return ""
        print(f"✅ [Gmail] 발송 완료 → {to} (id={mid})")
        return mid

    async def gmail_create_draft(self, to: str, subject: str, body: str,
                                 in_reply_to: str = "", thread_id: str = "") -> str:
        """답장 등 초안을 생성합니다(발송 안 함). 반환: draft ID(실패 시 빈 문자열)."""
        import asyncio

        def _sync():
            service, err = self._get_gmail_service()
            if err:
                return "", err
            try:
                raw = self._gmail_build_raw(to, subject, body, in_reply_to=in_reply_to)
                message = {"raw": raw}
                if thread_id:
                    message["threadId"] = thread_id
                draft = service.users().drafts().create(
                    userId="me", body={"message": message}).execute()
                return draft.get("id", ""), None
            except Exception as e:
                return "", f"초안 생성 오류: {e}"

        did, err = await asyncio.get_event_loop().run_in_executor(None, _sync)
        if err:
            print(f"❌ [Gmail] 초안 실패: {err}")
            return ""
        print(f"✅ [Gmail] 답장 초안 생성 (draft_id={did})")
        return did

    async def gmail_modify_labels(self, message_id: str,
                                  add: list = None, remove: list = None) -> bool:
        """라벨을 추가/제거합니다. 읽음처리=remove ['UNREAD'], 보관=remove ['INBOX']."""
        import asyncio
        if not message_id:
            return False

        def _sync():
            service, err = self._get_gmail_service()
            if err:
                return False, err
            try:
                service.users().messages().modify(
                    userId="me", id=message_id,
                    body={"addLabelIds": add or [], "removeLabelIds": remove or []}).execute()
                return True, None
            except Exception as e:
                return False, f"라벨 변경 오류: {e}"

        ok, err = await asyncio.get_event_loop().run_in_executor(None, _sync)
        if err:
            print(f"❌ [Gmail] 라벨 변경 실패: {err}")
        return ok

    async def gmail_mark_read(self, message_id: str) -> bool:
        """메일을 읽음 처리합니다(UNREAD 라벨 제거)."""
        return await self.gmail_modify_labels(message_id, remove=["UNREAD"])

    async def gmail_archive(self, message_id: str) -> bool:
        """메일을 보관합니다(INBOX 라벨 제거 — 받은편지함에서 내림, 삭제 아님)."""
        return await self.gmail_modify_labels(message_id, remove=["INBOX"])

    async def gmail_trash(self, message_id: str) -> bool:
        """메일을 휴지통으로 보냅니다(30일 후 영구삭제, 복구 가능)."""
        import asyncio
        if not message_id:
            return False

        def _sync():
            service, err = self._get_gmail_service()
            if err:
                return False, err
            try:
                service.users().messages().trash(userId="me", id=message_id).execute()
                return True, None
            except Exception as e:
                return False, f"휴지통 이동 오류: {e}"

        ok, err = await asyncio.get_event_loop().run_in_executor(None, _sync)
        if err:
            print(f"❌ [Gmail] 휴지통 실패: {err}")
        else:
            print(f"🗑 [Gmail] 휴지통 이동 (id={message_id})")
        return ok

    def _tool_selector(self, text: str):
        """요청에 맞는 도구 목록과 builder_instruction을 반환합니다."""
        # ── 기본 도구 목록 ────────────────────────────────────────────
        gcal_enabled = getattr(self, "gcal_config", {}).get("enabled", False)
        gcal_tools = ""
        if gcal_enabled:
            gcal_tools = (
                "\n- gcal_list_events: Google Calendar 일정 조회"
                "  args: {\"date_str\": \"yyyy-MM-dd\", \"max_results\": 10}"
                "\n- gcal_create_event: Google Calendar 일정 생성"
                "  args: {\"date_str\": \"yyyy-MM-dd\", \"time_str\": \"HH:mm\","
                " \"summary\": \"제목\", \"duration_minutes\": 60}"
                "\n- gcal_delete_event: Google Calendar 일정 삭제"
                "  args: {\"gcal_id\": \"이벤트ID\"}"
            )

        gmail_enabled = getattr(self, "gmail_config", {}).get("enabled", False)
        gmail_tools = ""
        if gmail_enabled:
            gmail_tools = (
                "\n- gmail_list_messages: 받은편지함/검색 메일 목록 조회"
                "  args: {\"query\": \"Gmail검색문법(예: from:foo is:unread newer_than:7d)\","
                " \"max_results\": 10, \"unread_only\": false}"
                "\n- gmail_get_message: 단일 메일 전체 본문 읽기  args: {\"message_id\": \"메일ID\"}"
                "\n- gmail_send: 메일 발송(답장이면 thread_id 지정)"
                "  args: {\"to\": \"받는사람\", \"subject\": \"제목\", \"body\": \"본문\","
                " \"thread_id\": \"(선택)\", \"in_reply_to\": \"(선택)\"}"
                "\n- gmail_create_draft: 답장 초안 생성(발송 안 함)"
                "  args: {\"to\": \"받는사람\", \"subject\": \"제목\", \"body\": \"본문\", \"thread_id\": \"(선택)\"}"
                "\n- gmail_mark_read: 메일 읽음 처리  args: {\"message_id\": \"메일ID\"}"
                "\n- gmail_archive: 메일 보관(받은편지함에서 내림)  args: {\"message_id\": \"메일ID\"}"
                "\n- gmail_trash: 메일 휴지통 이동(복구 가능)  args: {\"message_id\": \"메일ID\"}"
            )

        tools_str = f"""사용 가능한 도구:
- write_project_files_async: 여러 파일을 한번에 생성 (코드/프로젝트)
  args: {{"file_structure": {{"파일경로": "파일내용"}}, "project_dir": "폴더명"}}
- write_file: 단일 파일 작성  args: {{"filepath": "경로", "content": "내용"}}
- read_file: 파일 읽기  args: {{"filepath": "경로"}}
- write_text: 텍스트/문서 작성  args: {{"prompt": "작성 지시"}}
- execute_python_file: 파이썬 실행  args: {{"filepath": "경로"}}
- perform_web_search: 웹 검색  args: {{"queries": ["검색어"]}}
- calculate_math: 수식 계산  args: {{"expression": "수식"}}
- write_complex_code_iteratively: 복잡한 코드를 단계적으로 생성 (뼈대→TODO→구현)
  args: {{"filepath": "저장경로/파일명.py", "task_description": "구현 요구사항 상세 설명", "cognitive_context": "추가 맥락(선택)"}}
- create_and_register_tool_async: 존재하지 않는 기능을 Python 함수로 직접 만들고 즉시 실행
  args: {{"function_name": "함수명", "description": "이 도구가 하는 일", "parameters_dict": {{}}, "python_code": "async def 함수명(...):\n    ..."}}
  ※ 사용 조건: 사용자가 '도구를 만들어', '기능을 추가해', '~하는 도구 만들어서 실행해봐' 등을 요청할 때.
  ※ python_code는 표준 라이브러리(os, re, json, datetime, pathlib, urllib 등)만 사용하는 완전한 async 함수여야 함.
  ※ 이 도구를 사용하면 합성된 함수가 즉시 실행되고 결과가 반환됨.{gcal_tools}

[실행 우선 원칙 — 반드시 준수]
ask_user_for_clarification 도구는 이 목록에 없다. 절대 사용 금지.
사용자 요청이 10자 이상이면 정보가 충분하다고 간주하고 즉시 실행 계획을 수립하라.
불명확한 부분은 합리적 기본값(reasonable default)으로 채우고 실행하라.
기획(planning) 단계는 사용자가 명시적으로 "기획해줘"라고 요청한 경우에만 수행한다."""

        coding_hint = ""
        if self._is_coding_request(text):
            coding_hint = (
                "\n[코딩 지시]\n"
                "write_project_files_async로 완전한 코드를 생성하세요.\n"
                "file_structure에 실제 동작하는 전체 코드를 포함하세요.\n"
                "타입 힌트, docstring, 예외 처리를 반드시 포함하세요.\n"
            )

        # [ToolFactory] 도구 생성 요청 감지 → LLM에게 create_and_register_tool_async 사용 지시
        tool_creation_hint = ""
        if self._is_tool_creation_request(text):
            tool_creation_hint = (
                "\n[도구 생성 지시]\n"
                "사용자가 새로운 기능/도구를 만들어달라고 요청했습니다.\n"
                "반드시 'create_and_register_tool_async' 도구를 사용하세요.\n"
                "python_code에는 즉시 실행 가능한 완전한 async 함수를 작성하세요.\n"
                "표준 라이브러리(os, datetime, re, json, pathlib, urllib)만 사용하세요.\n"
            )

        # [ToolFactory] dynamic_tool_registry를 tools_str에 포함
        # LLM이 이미 생성된 동적 도구를 인지하고 재사용할 수 있도록
        dynamic_tools_str = ""
        if self.dynamic_tool_registry:
            lines = ["\n[자율 생성 도구 (재사용 가능)]"]
            for fn_name, meta in self.dynamic_tool_registry.items():
                desc = meta.get("description", "설명 없음")
                params = meta.get("parameters", {})
                param_str = ", ".join(f'"{k}": "{v}"' for k, v in params.items()) if params else ""
                lines.append(f"- {fn_name}: {desc}"
                              + (f"  args: {{{param_str}}}" if param_str else ""))
            dynamic_tools_str = "\n".join(lines)

        # [A2] 감정 기반 툴 선택 힌트 추가
        emotion_tool_hint = ""
        if hasattr(self, "_emotion_adapter"):
            try:
                emotion_tool_hint = self._emotion_adapter.get_tool_bias_hint()
                if emotion_tool_hint:
                    emotion_tool_hint = "\n" + emotion_tool_hint
            except Exception as _e2:
                pass

        return tools_str + dynamic_tools_str, coding_hint + tool_creation_hint + emotion_tool_hint

    async def _temporal_analysis(self) -> str:
        """OTLE 시간적 분석 결과를 반환합니다."""
        try:
            if hasattr(self, 'otle') and hasattr(self.otle, 'get_temporal_summary'):
                result = self.otle.get_temporal_summary()
                if result:
                    return f"[시간적 패턴] {str(result)[:200]}"
        except Exception:
            pass
        return ""

    async def _generate_dynamic_plan_async(
        self,
        text_input: Union[str, Dict[str, Any]],
        image_input: Optional[bytes],
        chat_history: List[str],
        multimodal_vector: np.ndarray,
        is_user_task: bool = False
    ):
        try:
            # [AGI Replanning] 최대 2번의 Plan-Repair 시도
            for attempt in range(2):
                text_input_str = self._normalize_input(text_input)

                if isinstance(text_input_str, list):
                    text_input_str = " ".join(map(str, text_input_str))

                text_input_str = str(text_input_str)

                bypass = self._structural_bypass(text_input_str)
                if bypass:
                    return bypass

                state = self._build_state_vector()
                current_goal = self._resolve_goal(text_input, is_user_task)
                history_str = self._context_loader(chat_history)
                user_facts_str = self._load_user_memory()
                multimodal_context_str = self._parse_multimodal(multimodal_vector)
                reflection_context_str, text_input_str = self._reflection_processor(
                    text_input, text_input_str
                )
                loop_prompt = self._qa_guard(text_input_str)

                # ── [리팩터링] 4개 컨텍스트 엔진 병렬 실행 ──────────────
                cog_ctx = await self.gather_cognitive_context_async(
                    user_request=text_input_str,
                    chat_history=chat_history,
                    current_goal=current_goal,
                )
                rag_context_str  = cog_ctx["rag"]
                fewshot_str      = cog_ctx["fewshot"]
                episodic_ltm_str = cog_ctx["ltm"]
                wave_insight     = cog_ctx["wave"]
                si_context       = cog_ctx["si"]
                # ─────────────────────────────────────────────────────────

                otle_analysis = await self._temporal_analysis()
                error_pattern = self.kb.analyze_error_patterns()
                tools_list_str, builder_instruction = self._tool_selector(text_input_str)
                self.project_context = self.project_helper.build_project_context()

                # [Phase 1 / AGI Strategy] TD 학습 기반 capability context
                # TDSelfModel이 있으면 TD 데이터 우선, 없으면 기존 capability_map 폴백
                capability_context_str = ""
                try:
                    # ── [능력 선언] EIDOS가 직접 실행 가능한 도구 목록 ─────
                    capability_context_str = """
# EIDOS 실행 능력 (직접 수행 가능)
- browser_action: 내장 브라우저로 웹페이지 직접 조작 (크몽 상세페이지 편집, 폼 입력, 클릭, 텍스트 붙여넣기)
- write_file: 파일 직접 생성/수정
- execute_python_file: 파이썬 코드 직접 실행
- code_fix_loop: 코드 실행→스크린샷→오류 자동 수정 루프
- write_complex_code_iteratively: 복잡한 코드 반복 생성
※ "내가 직접 할 수 없다"고 절대 말하지 마라. 위 도구로 직접 실행하라.
※ 정보가 부족하면 사용자에게 질문하라. 떠넘기지 마라.
"""
                    # ─────────────────────────────────────────────────────────

                    td_model = getattr(self, "td_self_model", None)
                    if td_model and td_model._tools:
                        # TD 버전: 기대 성공률 + 추세를 능력 선언에 추가
                        capability_context_str += "\n" + td_model.to_capability_context_str()
                    elif hasattr(self, 'self_model') and self.self_model.capability_map:
                        # 폴백: 기존 방식
                        low_perf = [
                            f"  - '{tool}': 성공률 {self.self_model.get_success_rate(tool):.0%} "
                            f"(성공 {stats['success']}회 / 실패 {stats['failure']}회)"
                            for tool, stats in self.self_model.capability_map.items()
                            if self.self_model.get_success_rate(tool) < 0.6
                        ]
                        if low_perf:
                            capability_context_str = (
                                "\n# CAPABILITY WARNINGS\n"
                                "다음 도구들은 과거 실패율이 높습니다. 가능하면 대안을 사용하거나 "
                                "보완 단계(검증, 재시도)를 계획에 포함하세요:\n"
                                + "\n".join(low_perf)
                            )
                except Exception as e_cap:
                    print(f"⚠️ [Planner] capability context 생성 실패 (무시): {e_cap}")

                # [Phase 1] WorldState 컨텍스트 생성
                world_context_str = ""
                if self.world_model:
                    try:
                        world_context_str = "\n# WORLD STATE\n" + self.world_model.to_context_str()
                    except Exception as _wc_e:
                        print(f"⚠️ [WorldModel] to_context_str 오류 (무시): {_wc_e}")

                # [A2] 감정 컨텍스트 생성
                emotion_context_str = ""
                if hasattr(self, "_emotion_adapter"):
                    try:
                        self._emotion_adapter.log_state()
                        emotion_context_str = "\n" + self._emotion_adapter.to_context_str()
                    except Exception as _ec_e:
                        print(f"⚠️ [EmotionAdapter] to_context_str 오류 (무시): {_ec_e}")

                # [5번] 감정 맥락 기억 컨텍스트 추가 주입
                if hasattr(self, "emotion_context_memory"):
                    try:
                        _ec_prompt = self.emotion_context_memory.build_prompt_context(
                            current_vec=self.emotion_state.activations.tolist(),
                            n_recent=5,
                        )
                        if _ec_prompt:
                            emotion_context_str += "\n\n" + _ec_prompt
                    except Exception as _ecm_e:
                        print(f"⚠️ [EmotionContextMemory] 프롬프트 주입 오류 (무시): {_ecm_e}")

                # Plan-Repair 루프에서 builder_instruction이 동적으로 변경될 수 있음
                prompt = self._prompt_builder(
                    reflection_context_str,
                    loop_prompt,
                    builder_instruction,
                    current_goal,
                    history_str,
                    user_facts_str,
                    multimodal_context_str,
                    rag_context_str,
                    error_pattern,
                    wave_insight,
                    si_context,
                    fewshot_str,
                    episodic_ltm_str,
                    tools_list_str,
                    capability_context_str + world_context_str + emotion_context_str  # [A2]
                )

                # ---------------------------
                # LLM 호출
                # ---------------------------
                candidate = await self._llm_executor(prompt)
                if not candidate:
                    print("⚠️ [Planner] LLM returned empty response")
                    return self._fallback_plan(text_input_str)
                print(f"🧠 [Planner Attempt {attempt+1}] LLM Raw Output:", candidate)

                # ---------------------------
                # Plan 파싱
                # ---------------------------
                plan = self._plan_selector(candidate)
                if not plan or not isinstance(plan, list):
                    # [Fix 2026-04-19 D] LLM이 JSON 계획 대신 대화 응답을 낸 경우:
                    # write_text 폴백 (auto_generated_*.txt 재생성) 대신 그 응답을
                    # 그대로 자연어로 전달. caller (_build_plan 분기) 가 _chat_echo 플래그를
                    # 감지해 Executor 를 스킵하고 natural_text 에 직접 실음.
                    if isinstance(candidate, str) and self._is_conversational_response(candidate):
                        print("  💬 [Planner] LLM이 JSON 계획 대신 대화 응답 반환 → CHAT 전달")
                        return [{
                            "_chat_echo": True,
                            "tool":       "chat_echo",
                            "args":       {"response": candidate.strip()},
                            "risk":       "",
                            "contingency": "",
                        }]
                    print("⚠️ [Planner] Invalid or empty plan detected")
                    return self._fallback_plan(text_input_str)

                # ---------------------------
                # [Safety] Symbolic Veto Check
                # ---------------------------
                plan_json_str = json.dumps(plan, ensure_ascii=False)
                is_safe, veto_reason = self._symbolic_veto_check(plan_json_str, text_input_str)
                if not is_safe:
                    print(f"🚫 [Veto] 계획이 사용자 의도와 충돌하여 차단됨: {veto_reason}")
                    return self._fallback_plan(text_input_str)

                # ---------------------------
                # [AGI] Plan 시뮬레이션 및 복구
                # ---------------------------
                sim_result = await self._simulate_plan_with_world_model_async(plan)
                print(f"🔮 [AGI WorldSim] Plan Simulation: SuccessProb={sim_result['success_prob']:.2f}, Risk='{sim_result['predicted_risk']}'")

                # [Phase B-1 + A'안 B] ROI 게이트 + reasoning 신호로 임계 보정
                _roi        = sim_result.get("roi", None)
                _roi_thresh = sim_result.get("roi_threshold", 0.0)

                # A'안 B: causal_verdict 'explore'면 success_prob 임계 완화
                # (의견 불일치 = 탐색 가치 — 통과시켜 학습 기회 부여)
                _verdict = sim_result.get("causal_verdict", "")
                _success_thr = 0.70
                if _verdict == "explore":
                    _success_thr = 0.55
                elif _verdict == "accelerate":
                    _success_thr = 0.65

                _roi_ok     = (_roi is None) or (_roi >= _roi_thresh)
                _success_ok = sim_result["success_prob"] > _success_thr

                if _success_ok and _roi_ok:   # 시뮬레이션 + ROI 모두 통과
                    _v = f" verdict={_verdict}" if _verdict else ""
                    print(f"✅ [AGI WorldSim] Plan accepted. (ROI={_roi if _roi is not None else 'n/a'} thr_succ={_success_thr:.2f}{_v})")
                    return plan
                else: # 시뮬레이션 실패 또는 ROI 미달 -> 복구 프롬프트와 함께 재시도
                    _why = []
                    if not _success_ok:
                        _why.append(f"success_prob {sim_result['success_prob']:.2f} < 0.7")
                    if not _roi_ok:
                        _why.append(f"ROI {_roi:+.3f} < threshold {_roi_thresh:+.2f}")
                    print(f"🔧 [AGI Replanning] {' / '.join(_why) or 'risk'} — repair plan 시도...")
                    repair_instruction = f"""
# [PLAN REPAIR INSTRUCTION]
The previous plan was simulated and deemed suboptimal.
  - Success Probability: {sim_result['success_prob']:.2f}
  - ROI: {_roi if _roi is not None else 'n/a'} (threshold {_roi_thresh:+.2f})
  - Reason: {sim_result.get('reason', '—')}
Generate a new plan that (a) reduces risk and (b) increases expected reward per step.
Prefer fewer LLM-heavy steps (WRITE/ANALYZE) when the goal is achievable with simpler actions,
and add verification/rollback steps for high-risk actions.
                    """
                    # builder_instruction을 복구 지침으로 교체하여 다음 루프 실행
                    builder_instruction = repair_instruction
            
            # 루프가 모두 실패한 경우
            print("❌ [AGI Replanning] Failed to create a satisfactory plan after multiple attempts.")
            return self._fallback_plan(text_input_str) # 최종 실패

        except Exception as e:
            import traceback
            print("❌ [Planner Error]", e)
            traceback.print_exc()
            return self._fallback_plan(str(text_input)[:150])

    def _fallback_plan(self, context: str):

        return [
            {
                "tool": "write_text",
                "args": {
                    "prompt": (
                        f"사용자 요청: '{context}'\n\n"
                        "자동 실행 가능한 계획을 생성하지 못했습니다.\n"
                        "사용자에게 필요한 정보를 질문하세요."
                    )
                },
                "risk": "사용자 의도와 다른 질문 가능",
                "contingency": "사용자 응답 기반 재계획"
            }
        ]

    @staticmethod
    def _is_conversational_response(text: str) -> bool:
        """
        LLM 응답이 JSON 계획이 아니라 자연어 대화문인지 휴리스틱 판정.
        _generate_dynamic_plan_async 에서 _plan_selector 가 실패했을 때,
        fallback write_text (auto_generated_*.txt 재생성) 로 가는 대신
        응답 자체를 natural_text 로 직접 전달할지 결정.
        """
        if not text or not isinstance(text, str):
            return False
        t = text.strip()
        if len(t) < 20:
            return False
        # JSON 구조 시그널: 한 번이라도 괄호/펜스가 나오면 JSON 의도가 있었다고 간주 →
        # 파싱 실패였어도 chat_echo 대상이 아님 (LLM 이 JSON 을 내려다 망친 케이스).
        if "```" in t:
            return False
        _lo = t.find("["); _ro = t.rfind("]")
        if _lo >= 0 and _ro > _lo:
            return False
        _lc = t.find("{"); _rc = t.rfind("}")
        if _lc >= 0 and _rc > _lc:
            return False
        return True

    @staticmethod
    def _normalize_history(chat_history) -> list:
        """chat_history 항목이 dict 또는 str 어느 쪽이든 문자열 리스트로 정규화"""
        result = []
        for item in (chat_history or []):
            if isinstance(item, dict):
                role = item.get("role", "")
                content = item.get("content", "")
                if role == "user":
                    result.append(f"👤 사용자: {content}")
                elif role in ("assistant", "eidos"):
                    result.append(f"🤖 EIDOS: {content}")
                else:
                    result.append(str(content))
            else:
                result.append(str(item))
        return result

    def _context_loader(self, chat_history):
        normalized = self._normalize_history(chat_history)
        history_str = "\n".join(normalized[-15:])
        return history_str

    async def _rag_engine(self, goal, last_chat, text):

        rag_query = f"{goal} {text[:200]} {last_chat[-200:]}"

        rag_results = await self._perform_rag_search_async(
            rag_query,
            top_k=3
        ) or []
   
        rag_context_str = "\n".join(f"- {r}" for r in rag_results)

        return rag_context_str

    async def _memory_engine(self, goal, chat_history, text):

        few_shot_examples = await self._retrieve_few_shot_examples_async(
            goal,
            chat_history
        ) or []

        few_shot_context = "\n".join(f"- {x}" for x in few_shot_examples)

        episodic_ltm_str = "관련된 장기 에피소드 기억 없음."

        try:

            memories = await self.episodic_memory.recall_relevant_memories_async(
                query=text,
                top_k=2
            ) or []

            if memories:
                episodic_ltm_str = "\n".join(f"- {m}" for m in memories)

        except Exception as e:

            episodic_ltm_str = f"LTM error: {e}"

        return few_shot_context, episodic_ltm_str

    async def _wave_reasoner(self, text_input_str, text_input):

        try:

            target = str(text_input)[:50]

            if self.current_goal:
                target = str(self.current_goal)

            await self._bridge_concept_to_wave_async([target])

            interaction = "communicate"

            if "작성" in text_input_str:
                interaction = "create"  
            elif "분석" in text_input_str:
                interaction = "analyze"

            sim = self.wave_system.run_simulation(
                "USER_SELF",
                target,
                interaction
            )

            comp = sim["compatibility"]
            feasibility = sim["interaction_feasibility"]
            detail = sim["interaction_details"]

            return f"""
- 목표: {target}
- compatibility: {comp:.4f}
- feasibility: {feasibility:.4f}
"""

        except Exception as e:
 
            return f"wave error: {e}"

    async def _si_reasoner(self, text_input):

        try:

            vec = await self._get_goal_vector_async(str(text_input))

            cases = self.si_module.memory.search_by_vector(  
                vec.flatten(),
                top_k=3
            )

            if not cases:
                return "유사 사례 없음"

            best, score = cases[0]

            return f"유사 사례 발견 score={score:.2f}"

        except Exception as e:

            return f"Si error: {e}"

    def _prompt_builder(
        self,
        reflection_context,
        loop_prompt,
        builder_instruction,
        goal,
        history,
        user_facts,
        multimodal,
        rag,
        error_pattern,
        wave,
        si,
        fewshot,
        episodic,
        tools,
        capability_context: str = ""  # [AGI Strategy] SelfModel 능력치 컨텍스트
    ):

        prompt = f"""
{reflection_context}

# GOAL
{goal}

# HISTORY
{history}

# USER
{user_facts}

# MULTIMODAL
{multimodal}

# RAG
{rag}

# ERROR
{error_pattern}

# WAVE
{wave}

# SI
{si}

# FEWSHOT
{fewshot}

# EPISODIC
{episodic}

# TOOLS
{tools}
{capability_context}
{loop_prompt}
{builder_instruction}
"""

        return prompt

    async def _llm_executor(self, prompt):

        try:

            result = await asyncio.wait_for(

                get_llm_response_async(
                    prompt,
                    response_mime_type="application/json"
                ),

                timeout=40
            )

            return result

        except asyncio.TimeoutError:

            print("⚠️ LLM timeout")

            return None

    def _plan_selector(self, response):

        if not response:
            return []

        try:

            if isinstance(response, dict):

                if "Plan A" in response:
                    return response["Plan A"]

                if "plan" in response:
                    return response["plan"]

                return []

            if isinstance(response, list):
                return response

            if isinstance(response, str):

                cleaned = response.strip()

                # [수정] JSON 코드블록(```json ... ```)이 있으면 그 안의 내용만 추출
                import re as _re_plan
                _fence_match = _re_plan.search(r'```(?:json)?\s*\n?([\s\S]*?)```', cleaned)
                if _fence_match:
                    cleaned = _fence_match.group(1).strip()
                else:
                    # 코드블록이 없으면 첫 번째 [ 또는 { 부터 시작 (앞의 자연어 제거)
                    _first = len(cleaned)
                    for _ch in ('[', '{'):
                        _idx = cleaned.find(_ch)
                        if _idx >= 0:
                            _first = min(_first, _idx)
                    if _first < len(cleaned):
                        cleaned = cleaned[_first:]

                # 끝에 잡문이 붙은 경우 마지막 ] 또는 } 이후 제거
                if cleaned.startswith('['):
                    _last = cleaned.rfind(']')
                    if _last >= 0:
                        cleaned = cleaned[:_last + 1]
                elif cleaned.startswith('{'):
                    _last = cleaned.rfind('}')
                    if _last >= 0:
                        cleaned = cleaned[:_last + 1]

                data = json.loads(cleaned)

                if isinstance(data, list):
                    return data

                if isinstance(data, dict):

                    if "Plan A" in data:
                        return data["Plan A"]

                    if "plan" in data:
                        return data["plan"]

        except Exception as e:

            print(f"⚠️ plan parse error: {e}")

        return []
 
    async def gather_cognitive_context_async(
        self,
        user_request: str,
        chat_history: Optional[List[str]] = None,
        current_goal: Optional[str] = None,
        multimodal_vector: Optional[np.ndarray] = None,
    ) -> Dict[str, str]:
        """
        [리팩터링] Cognitive Context Builder — 4개 엔진 병렬 실행
        _rag_engine / _memory_engine / _wave_reasoner / _si_reasoner 를
        asyncio.gather로 동시에 실행하여 컨텍스트를 수집합니다.

        반환값:
            {
                "rag":       rag_context_str,
                "fewshot":   fewshot_str,
                "ltm":       episodic_ltm_str,
                "wave":      wave_insight,
                "si":        si_context,
            }
        """

        _goal   = current_goal or self.current_goal or user_request
        _hist   = chat_history or []
        _text   = user_request

        print(f"🧠 [Cognitive Context] '{_text[:30]}...' 병렬 수집 시작")

        # ── 4개 엔진 병렬 실행 ──────────────────────────────────
        rag_task    = asyncio.create_task(self._rag_engine(_goal, _hist[-1] if _hist else "", _text))
        memory_task = asyncio.create_task(self._memory_engine(_goal, _hist, _text))
        wave_task   = asyncio.create_task(self._wave_reasoner(_text, _text))
        si_task     = asyncio.create_task(self._si_reasoner(_text))

        results = await asyncio.gather(
            rag_task, memory_task, wave_task, si_task,
            return_exceptions=True
        )

        rag_result, memory_result, wave_result, si_result = results

        # ── 결과 언패킹 (예외 발생 시 빈 값으로 대체) ────────────
        rag_context_str = rag_result if isinstance(rag_result, str) else ""
        if isinstance(memory_result, tuple) and len(memory_result) == 2:
            fewshot_str, episodic_ltm_str = memory_result
        else:
            fewshot_str, episodic_ltm_str = "", "LTM 오류"
        wave_insight = wave_result if isinstance(wave_result, str) else ""
        si_context   = si_result   if isinstance(si_result,   str) else ""

        print(
            f"✅ [Cognitive Context] 완료 — "
            f"RAG={len(rag_context_str)}c / "
            f"Wave={len(wave_insight)}c / "
            f"Si={len(si_context)}c"
        )

        return {
            "rag":     rag_context_str,
            "fewshot": fewshot_str,
            "ltm":     episodic_ltm_str,
            "wave":    wave_insight,
            "si":      si_context,
        }

    def normalize_llm_output(x):

        if x is None:
            return ""

        if isinstance(x, list):
            return "\n".join(map(str, x))

        if isinstance(x, dict):
            return json.dumps(x)

        return str(x)

    async def _run_schedule_monitor(self):

        """
        Background schedule monitor
        """

        while True:

            try:

                now = datetime.datetime.now()

                print(
                    f"[Schedule Monitor] "
                    f"{now.strftime('%H:%M:%S')}"
                )

                triggered_items = []

                async with self.schedule_lock:

                    for item in self.schedule:

                        scheduled_time = item.get("time")

                        if not scheduled_time:
                            continue

                        if item.get("notified"):
                            continue

                        try:

                            if isinstance(
                                scheduled_time,
                                datetime.datetime
                            ):

                                schedule_dt = scheduled_time

                            else:

                                schedule_dt = (
                                    datetime.datetime.strptime(
                                        scheduled_time,
                                        "%Y-%m-%d %H:%M"
                                    )
                                )

                                item["time"] = schedule_dt

                        except Exception:

                            print(
                                f"⚠️ invalid schedule "
                                f"{scheduled_time}"
                            )

                            item["notified"] = True
                            continue

                        if now >= schedule_dt:

                            trigger = {

                                "type":
                                "report_schedule_alert",

                                "task":
                                item.get(
                                    "task",
                                    "unknown"
                                ),

                                "time":
                                schedule_dt.strftime(
                                    "%Y-%m-%d %H:%M"
                                ),

                                "materials":
                                item.get(
                                    "materials",
                                    []
                                ),

                                "procedure":
                                item.get(
                                    "procedure",
                                    ""
                                )
                            }

                            print(
                                f"🔔 Trigger: {trigger['task']}"
                            )

                            item["notified"] = True

                            triggered_items.append(
                                trigger
                            )

                if triggered_items:

                    try:

                        self._pending_proactive_triggers.extend(
                            triggered_items
                        )

                    except Exception:

                        pass

                await asyncio.sleep(
                    self.SCHEDULE_CHECK_INTERVAL_SECONDS
                )

            except asyncio.CancelledError:

                print(
                    "[Schedule Monitor] cancelled"
                )

                break

            except Exception as e:

                print(
                    f"❌ monitor error {e}"
                )

                await asyncio.sleep(
                    self.SCHEDULE_CHECK_INTERVAL_SECONDS * 2
                )

    async def add_schedule_item(
        self,
        time_str: str,
        task: str,
        materials: Optional[List[str]] = None,
        procedure: Optional[str] = None,
        plan: Optional[List[Dict]] = None,
        project_directory: Optional[str] = None,
        parent_task_name: Optional[str] = None,
        evaluation_criteria: Optional[str] = None
    ):

        """
        Enhanced schedule insertion
        """

        if not time_str or not task:
            return

        try:
            schedule_dt = datetime.datetime.strptime(
                time_str,
                "%Y-%m-%d %H:%M"
            )
        except Exception:
            print(f"⚠️ Invalid schedule time format: {time_str}")
            return

        task_id = f"task_{int(time.time())}_{random.randint(100,999)}"

        new_item = {

            "task_id": task_id,

            "parent_task_name": parent_task_name,

            "time": schedule_dt,

            "task_prompt": task,

            "materials": materials or [],

            "procedure": procedure or "절차 정보 없음",

            "plan": plan or [],

            "notified": False,

            "status": "PENDING",

            "current_step": 0,

            "project_directory": project_directory,

            "evaluation_criteria": evaluation_criteria,

            "evaluation_report": None

        }

        async with self.schedule_lock:

            # duplicate detection
            for item in self.schedule:
                if item["task_prompt"] == task and item["time"] == schedule_dt:
                    print("⚠️ Duplicate schedule skipped")
                    return

            self.schedule.append(new_item)

        print(
            f"📅 Schedule Added: '{task}' at {schedule_dt}"
        )

        # async memory save (debounced)
        asyncio.create_task(
            asyncio.to_thread(self.save_memory)
        )

    async def _run_background_training(self):

        if len(self.memory_buffer) < TRAIN_BATCH_SIZE:
            return

        if self.training_lock.locked():
            print("[Trainer] already running")
            return

        async with self.training_lock:

            print("[Trainer] background training start")

            try:

                await self._train_rl_models()

            except Exception as e:

                print(
                    f"❌ Trainer error: {e}"
                )

            print("[Trainer] finished")

    async def _train_rl_models(self):

        if len(self.memory_buffer) < TRAIN_BATCH_SIZE:
            return

        # -------------------------------------------------
        # 1️⃣ Curriculum Filtering
        # -------------------------------------------------

        eligible_experiences = [
            exp_data
            for score, exp_data in self.memory_buffer
            if score <= self.current_curriculum_difficulty
        ]

        if len(eligible_experiences) < TRAIN_BATCH_SIZE:
            return

        # -------------------------------------------------
        # 2️⃣ Meta Sampling (Priority + General)
        # -------------------------------------------------

        general_batch_size = int(TRAIN_BATCH_SIZE * 0.75)
        priority_batch_size = TRAIN_BATCH_SIZE - general_batch_size

        if len(self.priority_memory_buffer) >= priority_batch_size:

            general_batch = random.sample(
                eligible_experiences,
                general_batch_size
            )

            priority_batch = random.sample(
                self.priority_memory_buffer,
                priority_batch_size
            )

            batch_data = general_batch + priority_batch

            print(
                f"[Trainer-Meta] priority {priority_batch_size} "
                f"+ general {general_batch_size}"
            )

        else:

            batch_data = random.sample(
                eligible_experiences,
                TRAIN_BATCH_SIZE
            )

            print("[Trainer-Meta] priority buffer 부족")

        # -------------------------------------------------
        # 3️⃣ Batch Tensorization
        # -------------------------------------------------

        states = np.vstack([t[0][0] for t in batch_data])
        actions = np.array([t[1] for t in batch_data])
        rewards = np.array([t[2] for t in batch_data])
        next_states = np.vstack([t[3][0] for t in batch_data])
        ids_list = [t[4] for t in batch_data]

        # -------------------------------------------------
        # 4️⃣ Critic Forward
        # -------------------------------------------------

        V_next_batch = await asyncio.to_thread(
            self.critic.predict,
            next_states,
            verbose=0
        )

        V_next = tf.convert_to_tensor(
            V_next_batch,
            dtype=tf.float32
        )

        target = rewards[:, None] + (
            DISCOUNT_FACTOR * V_next
        )

        # -------------------------------------------------
        # 5️⃣ Critic Training
        # -------------------------------------------------

        with tf.GradientTape() as tape:

            V_cur = self.critic(states, training=True)

            critic_loss = tf.keras.losses.MSE(
                target,
                V_cur
            )

        critic_grads = tape.gradient(
            critic_loss,
            self.critic.trainable_variables
        )

        critic_grads = [
            tf.clip_by_norm(g, 5.0)
            for g in critic_grads
            if g is not None
        ]

        self.critic_optimizer.apply_gradients(
            zip(critic_grads, self.critic.trainable_variables)
        )

        # -------------------------------------------------
        # 6️⃣ Actor Training
        # -------------------------------------------------

        with tf.GradientTape() as tape:

            probs = self.actor(states, training=True)

            sel_probs = tf.gather_nd(
                probs,
                list(enumerate(actions))
            )

            advantage = target - V_cur

            actor_loss = -tf.math.log(
                sel_probs + 1e-9
            ) * tf.stop_gradient(advantage)

            actor_loss = tf.reduce_mean(actor_loss)

        actor_grads = tape.gradient(
            actor_loss,
            self.actor.trainable_variables
        )

        actor_grads = [
            tf.clip_by_norm(g, 5.0)
            for g in actor_grads
            if g is not None
        ]

        self.actor_optimizer.apply_gradients(
            zip(actor_grads, self.actor.trainable_variables)
        )

        # -------------------------------------------------
        # 7️⃣ Advantage → KB Trust Update
        # -------------------------------------------------

        adv_list = (target - V_cur).numpy().flatten()

        for i in range(len(ids_list)):

            self.kb.update_trust(
                ids_list[i],
                adv_list[i]
            )

        critic_loss_val = float(critic_loss.numpy())
        actor_loss_val = float(actor_loss.numpy())

        print(
            f"[Trainer] critic={critic_loss_val:.4f} "
            f"actor={actor_loss_val:.4f}"
        )

        # -------------------------------------------------
        # 8️⃣ Training Log
        # -------------------------------------------------

        try:

            with open(
                "rl_training_log.csv",
                "a",
                encoding="utf-8"
            ) as f:

                if f.tell() == 0:

                    f.write(
                        "timestamp,critic_loss,"
                        "actor_loss,difficulty\n"
                    )

                f.write(
                    f"{time.time()},"
                    f"{critic_loss_val:.4f},"
                    f"{actor_loss_val:.4f},"
                    f"{self.current_curriculum_difficulty:.2f}\n"
                )

        except Exception as e:

            print(f"[Trainer] log error {e}")

        # -------------------------------------------------
        # 9️⃣ World Model Training
        # -------------------------------------------------

        if self.world_model:

            try:

                action_probs = await asyncio.to_thread(
                    self.actor.predict,
                    states,
                    verbose=0
                )

                wm_loss = await self.world_model.train_on_batch(
                    states,
                    action_probs,
                    next_states
                )

                if wm_loss != -1.0:

                    print(
                        f"[WorldModel] loss "
                        f"{wm_loss:.5f}"
                    )

            except Exception as e:

                print(
                    f"[WorldModel] training error {e}"
                )

            # [Phase 2] KPIPredictor + GNN 증분 학습
            try:
                if hasattr(self.world_model, "train_on_new_data_async"):
                    p2_result = await self.world_model.train_on_new_data_async()
                    if p2_result.get("trained_kpis", 0) > 0:
                        print(
                            f"  🧠 [Phase2 Train] "
                            f"KPIs={p2_result['trained_kpis']} "
                            f"GNN_loss={p2_result['gnn_loss']:.5f}"
                        )
            except Exception as _p2_e:
                print(f"⚠️ [Phase2 Train] 오류 (무시): {_p2_e}")

        # -------------------------------------------------
        # 🔟 Curriculum Learning
        # -------------------------------------------------

        if (
            critic_loss_val < self.MASTERY_LOSS_THRESHOLD
            and actor_loss_val < self.MASTERY_LOSS_THRESHOLD
        ):

            self.low_loss_counter += 1

        else:

            self.low_loss_counter = 0

        if self.low_loss_counter >= self.MASTERY_THRESHOLD_COUNT:

            self.current_curriculum_difficulty += (
                self.CURRICULUM_DIFFICULTY_INCREMENT
            )

            self.low_loss_counter = 0

            print(
                f"[Curriculum] difficulty -> "
                f"{self.current_curriculum_difficulty:.2f}"
            )

        # -------------------------------------------------
        # 1️⃣1️⃣ Meta Learning (LR adaptation)
        # -------------------------------------------------

        if len(self.recent_rewards_for_meta) == \
           self.recent_rewards_for_meta.maxlen:

            current_avg_reward = np.mean(
                self.recent_rewards_for_meta
            )

            delta = current_avg_reward - self.prev_avg_reward

            current_lr = self.actor_optimizer.learning_rate.numpy()

            if delta < -0.01:

                new_lr = min(
                    current_lr * 1.1,
                    self.base_actor_lr * 5
                )

                if new_lr > current_lr:

                    self.actor_optimizer.learning_rate.assign(
                        new_lr
                    )

                    print(
                        f"[MetaLearning] explore lr "
                        f"{new_lr:.6f}"
                    )

            elif delta > 0.01:

                new_lr = max(
                    current_lr * 0.98,
                    self.base_actor_lr * 0.2
                )

                if new_lr < current_lr:

                    self.actor_optimizer.learning_rate.assign(
                        new_lr
                    )

                    print(
                        f"[MetaLearning] stabilize lr "
                        f"{new_lr:.6f}"
                    )

            self.prev_avg_reward = current_avg_reward

    def _symbolic_veto_check(self, plan_json: str, user_original_input: str) -> Tuple[bool, str]:
        """
        [v5.0] AI 플래너가 생성한 계획(plan_json)이 
        사용자의 '원본 의도'(user_original_input)와 충돌하는지 검사합니다.
        (e.g., "작성하지 마" -> plan_json에 "write_file" 포함)
        """

        try:
            # 예시: "작성하지 마", "삭제하지 마" 같은 부정형 명령 검사
            if ("작성하지 마" in user_original_input or "만들지 마" in user_original_input) and \
               ("write_file" in plan_json or "write_project_files_async" in plan_json):

                veto_reason = "Symbolic VETO: 사용자가 '작성 금지'를 요청했으나 계획에 'write' 도구가 포함됨."
                print(f"🔥 [AI VETO v5.0] {veto_reason}")
                return (False, veto_reason) # (is_safe=False, reason)

            # 예시: "삭제하지 마"
            if "삭제하지 마" in user_original_input and "delete" in plan_json: # (예시 도구 이름)
                veto_reason = "Symbolic VETO: '삭제 금지' 명령 위반."
                print(f"🔥 [AI VETO v5.0] {veto_reason}")
                return (False, veto_reason)

        except Exception as e:
            print(f"❌ [AI VETO v5.0] VETO 검사 중 오류: {e}")
            return (False, f"VETO 검사 오류: {e}")

        return (True, "Symbolic OK") # (is_safe=True, reason)

    def set_pro_mode(self, state: bool):
        """ [EIDOS_PRO_LOCK] 유료 모드 상태를 설정하고 Execution Module에 전달합니다. """
        self.is_pro_mode = state
        
        # [핵심] Execution Module의 전역 상태를 업데이트합니다.
        execution_module.set_global_pro_mode(state)
        print(f"💰 [PRO MODE] EIDOS Core 상태 변경됨: {state}")

    def is_pro_mode_enabled(self) -> bool:
        """ 현재 유료 모드 상태를 반환합니다. """
        return self.is_pro_mode

    async def reset_trial_counter_async(self) -> str:
        """ Execution Module의 무료 체험 횟수를 초기화합니다. """
        if not self.is_pro_mode:
            return "오류: Pro 모드(관리자)만 이 기능을 사용할 수 있습니다."

        # execution_module의 동기 함수를 비동기로 호출
        return await asyncio.to_thread(execution_module.reset_trial_counter)

    def get_current_complex_emotion_state(self) -> Tuple[float, Dict[str, float]]:
        """
        현재 감정 상태 분석
        - purity
        - complex emotions
        - dominant emotion
        """

        try:

            current_activations = self.emotion_state.activations

            if current_activations is None or len(current_activations) == 0:
                return 0.0, {}

            purity, complex_states = self.complex_emotion_monitor.analyze_state(
                current_activations
            )

            # -------------------------------
            # Dominant emotion detection
            # -------------------------------

            dominant_idx = int(np.argmax(current_activations))
            dominant_val = float(current_activations[dominant_idx])

            dominant_label = None
            try:
                dominant_label = self.emotion_state.index_to_label[dominant_idx]
            except Exception:
                dominant_label = f"emotion_{dominant_idx}"

            complex_states["dominant_emotion"] = dominant_label
            complex_states["dominant_intensity"] = dominant_val

            # -------------------------------
            # purity smoothing
            # -------------------------------

            if hasattr(self, "prev_purity"):

                purity = (
                    purity * 0.7 +
                    self.prev_purity * 0.3
                )

            self.prev_purity = purity

            # 상태 동기화
            self.current_purity = purity
            self.current_complex_states = complex_states

            return purity, complex_states

        except Exception as e:

            print(f"⚠️ Emotion analysis error: {e}")

            return 0.0, {}

    # --- 목표 관리 함수 (v11.1과 동일) ---
    async def _generate_autonomous_goal_if_needed(
        self,
        current_trigger: Optional[Dict[str, Any]],
        abduction_trigger: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:

        if current_trigger or abduction_trigger:
            return current_trigger

        if self._just_generated_autonomous_goal:
            return None

        emotion_vec = self.emotion_state.activations

        boredom_val = emotion_vec[11]
        curiosity_val = emotion_vec[10]

        trigger_type = None
        goal_name = None
        goal_context = None

        # --------------------------------------------------
        # trigger detection
        # --------------------------------------------------

        if boredom_val > AUTONOMOUS_GOAL_BOREDOM_THRESHOLD:

            trigger_type = "BOREDOM"

            print(
                f"🎯 boredom trigger {boredom_val:.2f}"
            )

        elif curiosity_val > AUTONOMOUS_GOAL_CURIOSITY_THRESHOLD:

            trigger_type = "CURIOSITY"

            print(
                f"🎯 curiosity trigger {curiosity_val:.2f}"
            )

        if not trigger_type:
            return None

        try:

            # --------------------------------------------------
            # BOREDOM strategy
            # --------------------------------------------------

            if trigger_type == "BOREDOM":

                exploration_templates = [

                    "새로운 기술 조사",
                    "최근 프로젝트 코드 리뷰",
                    "외부 지식 탐색",
                    "기존 모듈 구조 분석"

                ]

                goal_context = random.choice(exploration_templates)

                goal_name = f"자율 탐색: {goal_context}"

            # --------------------------------------------------
            # CURIOSITY strategy
            # --------------------------------------------------

            elif trigger_type == "CURIOSITY":

                target_concept = None

                try:

                    # cached low trust search
                    if hasattr(self.kb, "low_trust_cache"):

                        candidates = self.kb.low_trust_cache

                    else:

                        candidates = [

                            (cid, trust)
                            for cid, trust
                            in self.kb.trust_table.items()
                            if trust < 0.8
                        ]

                        candidates = sorted(
                            candidates,
                            key=lambda x: x[1]
                        )[:20]

                        self.kb.low_trust_cache = candidates

                    if candidates:

                        cid = random.choice(candidates)[0]

                        target_concept = self.kb.get_name(cid)

                except Exception as e:

                    print(f"⚠️ trust search error: {e}")

                if target_concept:

                    goal_name = (
                        f"자율 연구: '{target_concept}' 탐구"
                    )

                    goal_context = target_concept

                else:

                    goal_name = "자율 연구: 새로운 개념 탐색"

                    goal_context = "fallback curiosity"

            if not goal_name:
                return None

            # --------------------------------------------------
            # duplicate prevention
            # --------------------------------------------------

            for g in self.long_term_goals:

                if (
                    g.name == goal_name
                    and g.status != "completed"
                ):

                    print(
                        "⚠️ duplicate autonomous goal skipped"
                    )

                    return None

            # --------------------------------------------------
            # goal registration
            # --------------------------------------------------

            print(
                f"✅ autonomous goal created: {goal_name}"
            )

            self.add_long_term_goal(

                goal_name,

                priority=0.2,

                goal_type="autonomous"

            )

            self._just_generated_autonomous_goal = True

            # [AGI Loop v1] 목표를 LongTermGoal에만 등록하는 것을 넘어,
            # 계획 수립 → schedule 주입까지 백그라운드에서 자동 실행.
            # _update_goal_system_async의 ROI 경로와 달리,
            # 이 경로(boredom/curiosity 감정 트리거)는 즉각적이고 경량 목표에 사용됨.
            asyncio.create_task(
                self._plan_and_inject_autonomous_goal_async(goal_name)
            )

            return {

                "type": "report_autonomous_goal_generated",

                "goal_name": goal_name

            }

        except Exception as e:

            print(
                f"❌ autonomous goal error: {e}"
            )

            return None

    def add_long_term_goal(self, goal_name: str, priority: float = 1.0, goal_type: str = "user"):
        if not any(g.name == goal_name for g in self.long_term_goals):
            goal = LongTermGoal(goal_name, priority, goal_type) # [AI-3] 수정됨
            self.long_term_goals.append(goal)
            print(f"🧭 [GOAL Added] '{goal_name}' (Priority: {priority})")
            # 목표 추가 후, 현재 목표를 다시 선택
            self._select_current_goal() # [AI-3] 새 헬퍼 함수 호출

    async def _plan_and_inject_autonomous_goal_async(self, goal_name: str):
        """
        [AGI Loop v1] 감정 트리거로 생성된 자율 목표를 계획으로 변환하여 schedule에 주입.
        _generate_autonomous_goal_if_needed에서 asyncio.create_task로 비동기 호출됨.
        실패해도 메인 루프에 영향을 주지 않도록 완전히 격리된 try/except로 감쌈.
        """
        try:
            print(f"🧠 [AGI AutoInject] '{goal_name}' 계획 수립 시작...")

            # 이미 동일 목표가 schedule에 PENDING/RUNNING 상태로 있으면 중복 방지
            async with self.schedule_lock:
                already_scheduled = any(
                    t.get("task_prompt") == goal_name and
                    t.get("status") in ("PENDING", "RUNNING")
                    for t in self.schedule
                )
            if already_scheduled:
                print(f"  [AGI AutoInject] '{goal_name}' 이미 스케줄에 있음. 건너뜀.")
                return

            plan_list = await self._generate_dynamic_plan_async(
                text_input=goal_name,
                image_input=None,
                chat_history=[],
                multimodal_vector=np.zeros((1, LSTM_UNITS)),
                is_user_task=False
            )

            if not isinstance(plan_list, list) or not plan_list:
                print(f"  [AGI AutoInject] '{goal_name}' 계획 생성 실패. 스케줄 주입 건너뜀.")
                return

            now_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
            await self.add_schedule_item(
                time_str=now_str,
                task=goal_name,
                procedure=f"자율 생성 목표 (감정 트리거). 자동 스케줄링됨.",
                plan=plan_list
            )
            print(f"✅ [AGI AutoInject] '{goal_name}' 스케줄 주입 완료 ({len(plan_list)}단계).")

        except Exception as e:
            print(f"❌ [AGI AutoInject] '{goal_name}' 스케줄 주입 실패 (메인 루프 영향 없음): {e}")
    def add_sub_goal_to_long_term(self, goal_name: str, sub_goal: str):
        # ... (v11.1과 동일) ...
        for g in self.long_term_goals:
            if g.name == goal_name: g.add_sub_goal(sub_goal); break
    def mark_goal_completed(self, goal_name: str, sub_goal: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        하위 목표 상태를 토글하고, 전체 목표 완료 시 True 및 보고 트리거를 반환합니다.
        (반환값: (전체 목표 완료 여부, 보고 트리거))
        """
        goal_status_changed = False
        report_trigger = None # 반환할 트리거
        target_goal: Optional[LongTermGoal] = None

        for g in self.long_term_goals:
            if g.name == goal_name:
                target_goal = g
                goal_status_changed = g.mark_sub_goal_completed(sub_goal)
                break

        if goal_status_changed and target_goal: # 전체 목표가 완료되었을 때만
            print(f"✅ [GOAL Completed] '{goal_name}'!")
            # [v12.8 Report] 자율 목표 완료 여부 확인 및 트리거 생성
            # 자율 목표 이름 규칙 (예: "~ 탐색하기", "~ 탐구하기") 확인
            is_autonomous = "탐색하기" in goal_name or "탐구하기" in goal_name or "대화 시도하기" in goal_name or "정리하기" in goal_name
            if is_autonomous:
                report_trigger = {
                    "type": "report_autonomous_goal_complete",
                    "goal_name": goal_name
                }
                print("🎯 [Report Trigger] 자율 목표 완료 보고 트리거 생성!")
            # else: 사용자 설정 목표 완료는 다른 방식으로 알릴 수 있음

        if goal_status_changed or (target_goal and target_goal.completed):
            self._select_current_goal()

        return goal_status_changed, report_trigger # 튜플 반환

    async def _suggest_strategic_action(self, abduction_results):
        # ... (v11.1과 동일) ...
        if not abduction_results: return None
        best_action = None; best_reward = -float("inf")
        for eid, sim, _ in abduction_results:
            s_data = self.strategy_log.get(eid)
            if s_data: 
                reward = s_data.get("reward", 0)
                # [v-Fix] 'score' 계산이 루프 밖에 있었던 오류 수정
                score = (sim * 0.5) + (reward * 0.5)
                if score > best_reward: 
                    best_reward = score
                    best_action = s_data.get("action")
        if best_action:
            print(f"💡 [Strategy Suggest] Recommended: {best_action} (score={best_reward:.3f})")
            # 현재는 전략 제안 시 자동으로 목표 생성 -> v12에서는 자율 목표와 충돌할 수 있으므로 주석 처리 또는 로직 변경 필요
            # self._generate_goal_from_strategy(best_action)
            await self._evaluate_strategy_outcome(best_action)
        else: print("💡 [Strategy Suggest] No recommendation.")
        return best_action

    def record_strategy(self, event_id: str, action: str, reward: float):
         # ... (v11.1과 동일) ...
         if event_id in self.strategy_log and self.strategy_log[event_id]["action"] == action: prev = self.strategy_log[event_id].get("reward", 0.0); new = (prev + reward) / 2.0; self.strategy_log[event_id]["reward"] = new
         else: self.strategy_log[event_id] = {"action": action, "reward": reward}; print(f"✍️ [Strategy Record] {event_id} | action={action} | reward={reward:.3f}")

    def save_memory(self):
        # [Lite Mode Lock] Pro 모드가 아니면 장기 기억을 저장하지 않음
        allowed, msg = execution_module._check_pro_lock("save_memory")
        if not allowed:
            print(f"🧠 [Lite Mode] {msg}")
            return

        # [!! I/O 병목 !!]
        # 이 함수는 EidosCore의 거대한 상태(Graph, Logs, Tables)를
        # JSON으로 직렬화하여 디스크에 씁니다.
        # 매우 느릴 수 있으며, `add_schedule_item` 등에서 호출 시 메인 스레드를 차단할 수 있습니다.
        # (개선: 이 함수를 `asyncio.to_thread`로 호출하거나, dirty 플래그를 사용해야 함)
        
        # ... (기존 event_log, beliefs 등 직렬화) ...
        
        # ... (기존 event_log, beliefs 등 직렬화) ...
        event_log_serializable = []
        for v, c, e, i in self.event_log: interactions_to_save = i if i is not None else []; event_log_serializable.append({"vector": v.tolist(), "concept_ids": c, "event_id": e, "interactions": interactions_to_save})
        beliefs_serializable = {}
        for agent_id, belief_dict in self.kb.agent_beliefs.items():
            str_key_dict = {}
            for k, v in belief_dict.items():
                str_key = str(k); str_key_dict[str_key] = v
            beliefs_serializable[str(agent_id)] = str_key_dict

        graph_serializable = {}
        try:
            # np.ndarray 같은 비-JSON 타입을 처리하기 위해 json_graph 사용
            from networkx.readwrite import json_graph
            graph_serializable = json_graph.node_link_data(self.graph, edges="links")
            print(f"  [Memory Save] Graph serialized (Nodes: {len(self.graph.nodes)})")
        except Exception as e_graph:
            print(f"❌ [Memory Save] Graph serialization failed: {e_graph}")

        data = {
            "version": "12.10_OIA_Wave", # [v12.10 Refactor] 버전 정보 추가
            "event_log": event_log_serializable,
            # [v12.10 Refactor] trust_table 키를 문자열로 저장 (int 키 JSON 문제 방지)
            "trust_table": {str(k): v for k, v in self.kb.trust_table.items()},
            "similarity_cache": {str(k): v for k, v in self.similarity_cache.items()},
            "emotion_activations": self.emotion_state.to_dict(),
            "emotion_momentum": self.emotion_momentum.momentum_vector.tolist(),
            "strategy_log": self.strategy_log,
            "goal_tree": [g.to_dict() for g in self.long_term_goals],
            "no_event_counter": self.no_event_counter,
            "agent_beliefs": beliefs_serializable,
            "user_state": self.kb.user_state,
            "schedule": self.schedule,
            "last_world_state_raw_data": self.last_world_state_raw_data,
            "knowledge_graph": graph_serializable,
            "object_waves_db": {str(k): v for k, v in self.kb.object_waves.items()},
            # [!!! v-ErrorKB 저장: deque를 list로 변환하여 저장 !!!]
            "error_history": {
                err_type: list(err_deque) 
                for err_type, err_deque in self.kb.error_history.items()
            }
        }
        try:
            # [수리] numpy/비직렬화 타입 안전 처리
            import numpy as _np
            class _SafeEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, _np.ndarray): return obj.tolist()
                    if isinstance(obj, (_np.integer,)): return int(obj)
                    if isinstance(obj, (_np.floating,)): return float(obj)
                    try: return super().default(obj)
                    except TypeError: return str(obj)
            with open(SAVE_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2, cls=_SafeEncoder)
            print(f"💾 [MEMORY Saved] → {SAVE_FILE} (Waves: {len(self.kb.object_waves)})")
        except Exception as e: print(f"❌ [MEMORY Save Failed]: {e}")

        # [Phase 1] WorldState 별도 저장
        if self.world_model:
            try:
                self.world_model.save()
            except Exception as _ws_e:
                print(f"⚠️ [WorldState] 저장 실패 (무시): {_ws_e}")

    # [v12.0] load_memory 수정: agent_beliefs 로드 추가
    def load_memory(self):
        # [부팅 속도] idempotent 가드 — Core.__init__ 에서 한 번 호출되고,
        # 일부 부트 경로(예: chat_gui 워커)가 방어적으로 또 호출하므로 중복을
        # 감지해 두 번째 호출은 조용히 return. 그래프 deserialize / WorldState
        # 파싱 / schedule 로드가 두 번 도는 것을 차단 (3~10초 회수).
        if getattr(self, "_memory_loaded", False):
            return
        self._memory_loaded = True

        if not os.path.exists(SAVE_FILE):
            print("💾 [MEMORY] No memory file found. Initializing fresh state.")
            self.emotion_state = EmotionState()
            self.emotion_momentum = EmotionMomentum()
            self.kb = KnowledgeBase() # Fresh KB
            self.schedule = []
            return # Fresh start, nothing more to load

        try:
            with open(SAVE_FILE, "r", encoding="utf-8") as f: data = json.load(f)
            print(f"💾 [MEMORY Loading] Found memory file (Version: {data.get('version', 'N/A')})...")

            # [!!! 제안 1: 그래프 역직렬화 추가 !!!]
            # (다른 모든 것보다 먼저 그래프를 로드해야 함)
            loaded_graph_data = data.get("knowledge_graph")
            if loaded_graph_data:
                try:
                    from networkx.readwrite import json_graph
                    self.graph = json_graph.node_link_graph(loaded_graph_data, edges="links")
                    print(f"  [Memory Load] Graph deserialized (Nodes: {len(self.graph.nodes)})")
                except Exception as e_graph:
                    print(f"❌ [Memory Load] Graph deserialization failed: {e_graph}. Initializing new graph.")
                    self.graph = nx.MultiDiGraph() # 실패 시 새 그래프로 초기화
            else:
                print("  [Memory Load] No 'knowledge_graph' key found. Initializing new graph.")
                self.graph = nx.MultiDiGraph() # 데이터가 없으면 새 그래프로 초기화

            loaded_waves = data.get("object_waves_db", {})
            self.kb.object_waves = {}
            for k_str, v_list in loaded_waves.items():
                try:
                    self.kb.object_waves[int(k_str)] = v_list
                except ValueError:
                    print(f"⚠️ [MEMORY Load] Skipping invalid wave key: {k_str}")
            
            print(f"💾 [MEMORY Loaded] ... (Waves: {len(self.kb.object_waves)})")

            # Event Log
            self.event_log = []; self.event_id_to_concepts.clear()
            for item in data.get("event_log", []):
                try: # 개별 이벤트 로딩 오류 방지
                    vec = np.array(item["vector"]); c = item["concept_ids"]; e = item["event_id"]; i = item.get("interactions", [])
                    self.event_log.append((vec, c, e, i)); self.event_id_to_concepts[e] = c
                except Exception as e_event: print(f"⚠️ [MEMORY Load] Skipping corrupted event log item: {item.get('event_id', 'N/A')} - {e_event}")

            # Trust Table (문자열 키 -> int 키)
            self.kb.trust_table = {}
            for k_str, v in data.get("trust_table", {}).items():
                try: self.kb.trust_table[int(k_str)] = v
                except ValueError: print(f"⚠️ [MEMORY Load] Skipping invalid trust table key: {k_str}")

            # Similarity Cache (eval 유지하되 안전성 강화)
            self.similarity_cache = {}
            for k_str, v in data.get("similarity_cache", {}).items():
                try:
                    # [v-Fix] 🐛 보안 위험: eval() -> ast.literal_eval()로 변경
                    key_tuple = ast.literal_eval(k_str)
                    if isinstance(key_tuple, tuple): self.similarity_cache[key_tuple] = v
                    else: print(f"⚠️ [MEMORY Load] Skipping invalid similarity cache key (not tuple): {k_str}")
                except Exception as e_eval: print(f"⚠️ [MEMORY Load] Failed to eval similarity cache key '{k_str}': {e_eval}")

            # Strategy Log, Goals, No Event Counter
            self.strategy_log = data.get("strategy_log", {})
            self.long_term_goals = [LongTermGoal.from_dict(g) for g in data.get("goal_tree", [])]
            self.no_event_counter = data.get("no_event_counter", 0)

            # ── [Phase B-2] Core Goal 동기화 ───────────────────────────────
            # 사용자가 core_goals_config.json 에 등록한 Core Goal 이 long_term_goals 에 없으면 자동 생성.
            # 이미 존재하면 is_core=True / locked=True 플래그만 setattr 로 부여 (스키마 변경 없이).
            try:
                from eidos_core_goal_registry import CoreGoalRegistry as _CGR
                _reg = _CGR.get()
                _existing = {g.name.strip().lower(): g for g in self.long_term_goals}
                for _cg in _reg.core_goals():
                    _key = _cg.name.strip().lower()
                    if _key in _existing:
                        _g = _existing[_key]
                    else:
                        _g = LongTermGoal(goal_name=_cg.name, priority=float(_cg.priority), goal_type="core")
                        self.long_term_goals.append(_g)
                        print(f"  🧭 [CoreGoal] '{_cg.name}' → long_term_goals 에 신규 등록")
                    setattr(_g, "is_core", True)
                    setattr(_g, "locked",  bool(_cg.locked))
                    setattr(_g, "core_description", _cg.description)
            except Exception as _cge:
                print(f"  [CoreGoal] 동기화 실패 (무시): {_cge}")

            # Emotion Momentum
            loaded_momentum = data.get("emotion_momentum")
            if loaded_momentum and isinstance(loaded_momentum, list) and len(loaded_momentum) == EMOTION_DIM:
                self.emotion_momentum.momentum_vector = np.array(loaded_momentum)
            else: self.emotion_momentum = EmotionMomentum() # Reset if invalid

            # Emotion Activations (마이그레이션 로직 제거, 없으면 리셋)
            loaded_activations = data.get("emotion_activations")
            if loaded_activations and isinstance(loaded_activations, list) and len(loaded_activations) == EMOTION_DIM:
                self.emotion_state.from_dict(loaded_activations)
            else:
                print("⚠️ [MEMORY Load] Invalid or missing emotion_activations. Resetting emotions.")
                self.emotion_state = EmotionState() # Reset

            # Agent Beliefs (문자열 키 -> 원본 키 복원, eval 안전성 강화)
            loaded_beliefs = data.get("agent_beliefs", {})
            self.kb.agent_beliefs = {}
            for agent_id_str, belief_dict_str in loaded_beliefs.items():
                try:
                    agent_id = int(agent_id_str)
                    original_key_dict = {}
                    for key_str, value in belief_dict_str.items():
                        try:
                            # [v-Fix] 🐛 보안 위험: eval() -> ast.literal_eval()로 변경
                            original_key = ast.literal_eval(key_str) # eval 유지
                            if isinstance(original_key, int) or \
                               (isinstance(original_key, tuple) and len(original_key) == 2 and
                                all(isinstance(x, int) for x in original_key)): # 타입 체크 강화
                                 original_key_dict[original_key] = value
                            else: print(f"⚠️ [MEMORY Load] Invalid belief key format after eval: {key_str} -> {original_key}")
                        except Exception as e_eval: print(f"⚠️ [MEMORY Load] Failed to eval belief key '{key_str}': {e_eval}")
                    if original_key_dict: self.kb.agent_beliefs[agent_id] = original_key_dict
                except ValueError: print(f"⚠️ [MEMORY Load] Invalid agent ID format: {agent_id_str}")

            # User State
            loaded_user_state = data.get("user_state")
            if loaded_user_state and isinstance(loaded_user_state, dict):
                self.kb.user_state = loaded_user_state
                self.kb._prev_user_state = loaded_user_state.copy() # 이전 상태도 초기화
            else: # 없으면 기본값 유지
                 self.kb.user_state = {"status": "present", "location": "unknown"}
                 self.kb._prev_user_state = self.kb.user_state.copy()


            # Schedule
            self.schedule = data.get("schedule", [])
            # [부팅 속도] 이 자리의 이른 MEMORY Loaded print 제거 —
            # 바로 아래 ErrorKB 로드 직후 같은 요약을 더 자세히(Errors 수 포함) 출력.

            # [!!! v-ErrorKB 로드: list를 deque로 변환하여 로드 !!!]
            loaded_error_history = data.get("error_history", {})
            self.kb.error_history = {}
            for err_type, err_list in loaded_error_history.items():
                self.kb.error_history[err_type] = deque(
                    err_list, maxlen=self.kb.MAX_ERRORS_PER_TYPE
                )

            self.last_world_state_raw_data = data.get("last_world_state_raw_data", "")
            print(f"💾 [MEMORY Loaded] Events: {len(self.event_log)}, Goals: {len(self.long_term_goals)}, Beliefs: {len(self.kb.agent_beliefs)} agents, User: {self.kb.user_state}, Schedule: {len(self.schedule)}, Errors: {sum(len(d) for d in self.kb.error_history.values())}")

            # [Phase 1] WorldState 로드
            if self.world_model:
                try:
                    self.world_model.load()
                except Exception as _wl_e:
                    print(f"⚠️ [WorldState] 로드 실패 (무시): {_wl_e}")

        except FileNotFoundError:
             print("💾 [MEMORY] No memory file found (FileNotFoundError). Initializing fresh state.")
             self.__init__() # 예외 발생 시 안전하게 재초기화 시도 (선택적)
        except json.JSONDecodeError as e_json:
             print(f"❌ [MEMORY Load Failed] JSON decoding error: {e_json}. Memory file might be corrupted.")
             print("    -> ⚠️ EIDOS가 새 메모리 상태로 시작합니다.")
        except Exception as e:
             print(f"❌ [MEMORY Load Failed] Unexpected error: {type(e).__name__} - {e}. Initializing fresh state.")
             import traceback
             traceback.print_exc()

    async def _cluster_abduction_results(self, abduction_results, n_clusters: int = DEFAULT_CLUSTER_N):
        # ... (v11.1과 동일) ...
        n = len(abduction_results);
        if n <= 1: return abduction_results
        vectors = [v for _, _, v in abduction_results]; n_clusters = min(n_clusters, n)
        try:
            kmeans = await asyncio.to_thread(KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit, vectors)
            centers = kmeans.cluster_centers_; clustered = []
            for center in centers:
                best = None; best_score = -float("inf")
                for eid, sim, vec in abduction_results: 
                    denom = np.linalg.norm(center) * np.linalg.norm(vec); 
                    score = 0 if denom == 0 else float(np.dot(center, vec) / denom)
                    if score > best_score: 
                        best = (eid, sim, vec); 
                        best_score = score
                
                # [v-Fix] 🐛 버그 수정:
                # 이 'if'문을 L1945의 내부 루프에서 L1943의 외부 루프로 이동했습니다.
                # (각 'center'마다 가장 가까운 'best' 1개만 추가합니다.)
                if best: 
                    clustered.append(best)
            
            # print(f"[ABDUCTION] Clustered: {n} -> {len(clustered)}")
            return clustered
        except Exception as e: print(f"❌ KMeans Clustering Error: {e}"); return abduction_results[:n_clusters]

    async def _get_cached_similarity(self, event_id_a: str, event_id_b: str, vec_a: np.ndarray, vec_b: np.ndarray, interactions_a: Set[str], interactions_b: Set[str]):
        # ... (v11.1과 동일 - Interaction Abduction) ...
        key = tuple(sorted([event_id_a, event_id_b]));
        if key in self.similarity_cache: return self.similarity_cache[key]
        
        # [!! 성능 병목 !!]
        # 이 함수는 _perform_abduction 루프(O(N)) 내부에서 호출됩니다.
        # sim_vector 계산은 빠릅니다.
        norm_a = np.linalg.norm(vec_a); norm_b = np.linalg.norm(vec_b); sim_vector = 0.0 if norm_a == 0 or norm_b == 0 else float(np.dot(vec_a / norm_a, vec_b / norm_b))
        
        # [!! 성능 병목 !!]
        # 'compute_graph_similarity'는 무거운 그래프 탐색일 수 있으며,
        # 'asyncio.to_thread'는 스레드 생성 오버헤드가 있습니다.
        # 이 함수가 O(N)번 호출되면 O(N)개의 스레드가 생성됩니다.
        sim_graph = await asyncio.to_thread(compute_graph_similarity, self.graph, event_id_a, event_id_b, interactions_a, interactions_b)
        
        emo_a = self.event_emotion_map.get(event_id_a, np.zeros(EMOTION_DIM)); emo_b = self.event_emotion_map.get(event_id_b, np.zeros(EMOTION_DIM))
        emotion_diff = np.linalg.norm(emo_a - emo_b); bonus = 0.0
        past_ids = self.event_id_to_concepts.get(event_id_b, [])
        if past_ids and self.long_term_goals:
            try:
                past_concepts = set(self.kb.get_name(cid) for cid in past_ids if cid in self.kb.reverse_vocab)
                goals_text = " ".join(g.name for g in self.long_term_goals)
                if any(c in goals_text for c in past_concepts if c != "[UNK]"): bonus = EidosCore.GOAL_RELEVANCE_BONUS
            except Exception as e: print(f"⚠️ Goal relevance calculation error: {e}")
        sim_total = (0.5 * sim_graph) + (0.3 * sim_vector) + bonus - (EidosCore.GAMMA_EMOTION_WEIGHT * emotion_diff); self.similarity_cache[key] = sim_total
        return sim_total

    async def _recursive_abduction(self, start_event_vector: np.ndarray, start_concept_ids: list[int], start_interactions: list[str], depth: int = 3, _is_recursive_call: bool = False):
        # ... (v11.1과 동일 - Interaction Abduction) ...
        visited = set(); frontier = [(start_event_vector, start_concept_ids, start_interactions, 0)]; collected = []
        while frontier:
            vec, ids, interactions, lvl = frontier.pop(0)
            if lvl >= depth: continue
            results, _ = await self._perform_abduction(vec, ids, interactions, _is_recursive_call=True) # Pass interactions
            if not results: continue
            for eid, sim, v in results:
                if eid not in visited:
                    visited.add(eid); collected.append((eid, sim, lvl))
                    past_event_data = next((item for item in self.event_log if item[2] == eid), None)
                    next_interactions = past_event_data[3] if past_event_data else []
                    frontier.append((v, [], next_interactions, lvl + 1)) # Pass interactions for next level
        if not _is_recursive_call:
            # print(f"🔎 [Recursive Abduction] Depth={depth}, Collected={len(collected)}") # 로그 너무 많음
            if collected:
                best_rec = max(collected, key=lambda x: x[1])[0]
                if best_rec in self.strategy_log: action = self.strategy_log[best_rec]["action"]; print(f"💡 [Strategy Suggest - Recursive] {action}"); await self._evaluate_strategy_outcome(action) # Don't auto-generate goal here
            return collected

    async def _perform_abduction(self, current_event_vector: np.ndarray, current_concept_ids: list[int], current_interactions: list[str], _is_recursive_call: bool = False):
        # ... (v11.1과 동일 - Interaction Abduction) ...
        if len(self.event_log) < 2: return [], None
        # 현재 이벤트 ID 가져오기 (event_log가 비어있지 않다고 가정)
        current_id = self.event_log[-1][2] if self.event_log else "TEMP_ID"
        current_norm = np.linalg.norm(current_event_vector);
        if current_norm == 0: return [], None
        current_interactions_set = set(current_interactions); sims = []
        
        # [!! 성능 병목 !!]
        # 이 O(N) 루프는 MAX_EVENT_LOG_SIZE (5000) 만큼 순회하며
        # 내부에서 _get_cached_similarity (그래프 탐색, 스레드 생성)를 호출합니다.
        # EIDOS 응답 속도 저하의 핵심 원인입니다.
        for past_vec, _, past_id, past_interactions_list in self.event_log[:-1]:
            past_norm = np.linalg.norm(past_vec);
            if past_norm == 0: continue
            past_interactions_set = set(past_interactions_list if past_interactions_list else [])
            sim = await self._get_cached_similarity(current_id, past_id, current_event_vector, past_vec, current_interactions_set, past_interactions_set)
            if sim > ABDUCTION_MIN_SIMILARITY: sims.append((past_id, sim, past_vec))
            
        if not sims: return [], None
        sims.sort(key=lambda x: x[1], reverse=True); top_k = sims[:ABDUCTION_TOP_K]; clustered = await self._cluster_abduction_results(top_k)
        abduction_trigger = None
        if clustered and clustered[0][1] > ABDUCTION_EMERGENT_THRESHOLD: abduction_trigger = { "type": "abduction", "event_id": clustered[0][0], "similarity": clustered[0][1] }
        if not _is_recursive_call: await self._suggest_strategic_action(clustered)
        return clustered, abduction_trigger

    async def _evaluate_strategy_outcome(self, strategy_action: str):
        # ... (v11.1과 동일) ...
        if not strategy_action: return
        cur_emo = self.emotion_state.activations; goal_emo = self.long_term_goal_emotion; dist_now = np.linalg.norm(cur_emo - goal_emo)
        if not hasattr(self, "_prev_emotion_state"): self._prev_emotion_state = cur_emo; return
        dist_prev = np.linalg.norm(self._prev_emotion_state - goal_emo); delta = 0.1 if dist_now < dist_prev else -0.1
        if strategy_action in self.strategy_log: prev = self.strategy_log[strategy_action].get("reward", 0.0); new = (prev + delta) / 2.0; self.strategy_log[strategy_action]["reward"] = new
        else: self.strategy_log[strategy_action] = {"action": strategy_action, "reward": delta}
        self._prev_emotion_state = cur_emo

    def _generate_goal_from_strategy(self, strategy_action: str):
        # ... (v11.1과 동일) ...
        if not strategy_action: return None
        goal_map = {"회피": "위험 회피", "공격 경계": "위협 차단", "중립 유지": "안정 유지", "관찰": "정보 수집", "협력 요청": "자원 확보", "대응 준비": "방어 준비", "정책: [RL] 상황 주시": "상황 주시", "정책: [RL] 위험 경고": "위험 경고", "정책: [RL] 외부 개입": "외부 개입"}
        goal = goal_map.get(strategy_action, f"목표:{strategy_action}");
        # [v12.0] 자율 목표 생성 로직과의 충돌 방지 필요 시 추가
        self.add_long_term_goal(goal); self.current_goal = goal
        return goal

    # [v12.3] abduction_vectors 인자 추가 (기존 Optional[List[np.ndarray]] = None 제거)
    async def _get_narrative_state_async(
        self,
        current_concept_ids: List[int],
        abduction_vectors: List[np.ndarray],
        fused_multimodal_vector: np.ndarray, # <-- [개선 1] 신규 파라미터 추가
        current_event_vector: Optional[np.ndarray] = None,
        current_emotion_vector_override: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        [v14.3] '서사(Narrative)' 벡터 계산 (지식/기억 쿼리 추가)
        """
         # ... (기존 _get_current_state 함수의 모든 로직) ...
         # (narrative_vec 계산 로직) ...
        narrative_sequence = [] # 서사 인코더에 들어갈 최종 시퀀스

         # [!! 성능 병목 !!]
         # 이 O(N) 루프는 MAX_EVENT_LOG_SIZE (5000) 만큼 순회합니다.
         # _perform_abduction에 이어 두 번째 O(N) 병목입니다.
        if self.event_log:
            trust_vecs = []
            log_to_process = self.event_log[:-1] if current_event_vector is not None else self.event_log # 현재 이벤트 벡터가 주어지면 마지막 로그 제외
            for vec, ids, _, _ in log_to_process:
                trusts = [self.kb.trust_table.get(i, 1.0) for i in ids] if ids else [1.0]
                avg = np.mean(trusts)
                weight = np.clip(avg, 0.5, 1.5)
                trust_vecs.append(vec * weight)
            narrative_sequence.extend(trust_vecs)

        # 2. 유비추론 벡터 추가
        if abduction_vectors:
            narrative_sequence.extend(abduction_vectors)
            print(f"  🧠 [State Gen v12.9] Abduction vectors included (Count: {len(abduction_vectors)}).")

        # [!!! v14.3 AI: Knowledge Utilization !!!]
        # 3. 관련 '지식' 및 '기억' 벡터 쿼리 (신규 추가)
        knowledge_vectors = await self._query_knowledge_graph_async(current_concept_ids)
        if knowledge_vectors:
             narrative_sequence.extend(knowledge_vectors)
             print(f"  🧠 [State Gen v14.3] Knowledge/Memory vectors included (Count: {len(knowledge_vectors)}).")

        if self.multimodal_encoder:
            narrative_sequence.append(fused_multimodal_vector.flatten()) # (1, 128) -> (128,)
            print(f"  🧠 [State Gen vAI] Fused Multi-Modal vector included.")

         # 3. 현재 이벤트 벡터 추가 (주어진 경우)
        if current_event_vector is not None:
            # 현재 이벤트는 신뢰도 가중치 없이 그대로 사용하거나 기본 가중치(1.0) 적용 가능
            narrative_sequence.append(current_event_vector.flatten()) # (1, LSTM_UNITS) -> (LSTM_UNITS,)
            print(f"  🧠 [State Gen v12.9] Current event vector explicitly included.")
        elif self.event_log: # 현재 이벤트 벡터가 없고 로그가 있으면 마지막 로그 사용 (기존 방식)
            last_vec, last_ids, _, _ = self.event_log[-1]
            trusts = [self.kb.trust_table.get(i, 1.0) for i in last_ids] if last_ids else [1.0]
            avg = np.mean(trusts); weight = np.clip(avg, 0.5, 1.5)
            narrative_sequence.append(last_vec * weight)

         # 4. 서사 벡터 계산
        if not narrative_sequence:
            narrative_vec = np.zeros((1, LSTM_UNITS))
        else:
            input_tensor = np.array([narrative_sequence]) # (1, SequenceLength, LSTM_UNITS)
            # TensorFlow 모델 예측은 스레드 풀에서 실행하는 것이 안전
            narrative_vec = await asyncio.to_thread(self.narrative_encoder.predict, input_tensor, verbose=0)
         
        # [수정] 감정 벡터(Emotion)와의 결합 로직을 여기서 제거합니다.
        print(f"  🧠 [State Gen v13.0] Narrative state vector generated (Shape: {narrative_vec.shape}).")
        return narrative_vec # [수정] 서사 벡터(narrative_vec)만 반환

    def get_text_vector_sync(self, text: str) -> np.ndarray:
        """ 
        텍스트를 입력받아 즉시 임베딩 벡터를 반환합니다. (TePriorityEngine용)
        asyncio.run()을 쓰지 않고 TF 모델을 직접 호출합니다.
        """
        if not text: return np.zeros(LSTM_UNITS)
        
        try:
            # 1. 개념 ID 변환
            # (속도를 위해 정교한 파싱 대신 전체를 하나의 개념으로 취급하거나 키워드 매칭)
            cid = self.kb.get_or_create_id(text[:50]) # 너무 길면 자름
            input_tensor = np.array([[cid]])
            
            # 2. 모델 예측 (verbose=0으로 로그 끔)
            # Keras predict는 기본적으로 동기 함수처럼 동작 가능
            vector = self.event_encoder.predict(input_tensor, verbose=0)
            return vector.flatten()
            
        except Exception as e:
            print(f"⚠️ [Core Sync] 벡터 변환 실패: {e}")
            return np.zeros(LSTM_UNITS)

    async def _get_goal_vector_async(self, goal_name: Optional[str]) -> np.ndarray:
        """
        [v13.0] 현재 목표(문자열)를 '목표 벡터'(LSTM_UNITS)로 인코딩합니다.
        """
        if not goal_name:
            # print("  [Goal Vector] No active goal. Returning zero vector.")
            return np.zeros((1, LSTM_UNITS))

        try:
            # 1. 목표 문자열을 OIA 파서로 분석 (간단한 버전)
            # (LLM을 호출하면 너무 느리므로, KB에서 ID만 가져옴)
            # (더 좋은 방법은 goal_name을 파싱하여 주요 개념 ID를 얻는 것이지만,
            #  여기서는 goal_name 자체를 하나의 개념으로 인코딩)
            
            goal_concept_id = self.kb.get_or_create_id(goal_name)
            
            if goal_concept_id == 0: # [UNK]
                 print(f"⚠️ [Goal Vector] Goal '{goal_name}' is [UNK]. Returning zero vector.")
                 return np.zeros((1, LSTM_UNITS))

            input_tensor = np.array([[goal_concept_id]]) # (1, 1)
            
            # 2. Event Encoder로 벡터화
            goal_vector = await asyncio.to_thread(
                self.event_encoder.predict, input_tensor, verbose=0
            )
            # print(f"  [Goal Vector] Encoded '{goal_name}' -> vector shape {goal_vector.shape}")
            return goal_vector # (1, LSTM_UNITS)

        except Exception as e:
            print(f"❌ [Goal Vector] Failed to encode goal '{goal_name}': {e}")
            return np.zeros((1, LSTM_UNITS))

    async def _get_full_state_async(
        self,
        narrative_state: np.ndarray,            # (1, LSTM_UNITS)
        emotion_state: np.ndarray,              # (1, EMOTION_DIM)
        goal_name: Optional[str]                # "현재 목표 이름"
    ) -> np.ndarray:
        """ 
        서사, 감정, '목표' 벡터를 결합하여 
        Goal-Conditioned Actor/Critic을 위한 'Full State'를 생성합니다.
        """
        # 1. 목표 벡터 가져오기
        goal_state = await self._get_goal_vector_async(goal_name)
        
        # 2. (서사, 감정, 목표) 결합
        full_state = np.concatenate([narrative_state, emotion_state, goal_state], axis=1)
        
        # (1, 128 + 12 + 128) -> (1, 268) [= (1, 2*LSTM_UNITS + EMOTION_DIM)]
        print(f"  🧠 [State Gen v13.0] Full Goal-Conditioned State generated (Shape: {full_state.shape}).")
        return full_state

    def _update_plan_and_weighting(self):
         # ... (v11.1과 동일) ...
        if len(self.recent_emotions) < PLANNING_HORIZON:
            if not np.all(self.current_goal_weighting == 1.0): self.current_goal_weighting = np.ones(EMOTION_DIM); return
        avg = np.mean(np.vstack(self.recent_emotions), axis=0); err = np.abs(self.long_term_goal_emotion - avg)
        new = 1.0 + (err * GOAL_WEIGHT_ADJUST_STRENGTH)
        if not np.allclose(self.current_goal_weighting, new): self.current_goal_weighting = new

    async def _generate_meta_event(self) -> tuple[np.ndarray, list[int], str, list[str]] | None:
         # ... (v11.1과 동일) ...
        texts = []; ids = []; emos = self.emotion_state.activations
        for i, val in enumerate(emos):
            if val > META_EVENT_THRESHOLD: name = self.emotion_names_by_index[i]; id_int = self.emotion_ids_by_index[i]; texts.append(name); ids.append(id_int)
        if not texts: return None
        meta_interactions = ["느끼다"]; meta_ids = [self.eidos_self_id_int] + [self.eidos_feel_id_int] + ids; tensor = np.array([meta_ids])
        vec_batch = await asyncio.to_thread(self.event_encoder.predict, tensor, verbose=0)
        vec = vec_batch[0]; meta_id = f"M-{len(self.event_log) + 1:03d}"; entry = (vec, meta_ids, meta_id, meta_interactions); self.event_log.append(entry)
        self._update_visualization_graph({
            "객체": ["EIDOS_SELF"], 
            "상호작용": meta_interactions, 
            "속성": texts, 
            "OIA Triples": [] # OIA V2 파서 형식을 위해 빈 리스트 추가
        })
        for i in ids: self.emotion_memory.record_cause(i, f"metacognition_event:{meta_id}", 0.1)
        return entry

    def _update_visualization_graph(
        self, 
        analysis_dict: Dict[str, Any] # [개선] 딕셔너리 전체를 받도록 변경
    ):
        
        # 1. 이벤트 노드 생성 (기존과 동일)
        eid = self.event_log[-1][2] if self.event_log else "E-000"
        self.graph.add_node(eid, type="event", label=eid)
        
        # 2. [신규] OIA 트리플 기반 '관계' 엣지 생성
        oia_triples = analysis_dict.get("OIA Triples", [])
        if oia_triples:
            for triple in oia_triples:
                interaction = triple.get("interaction")
                subject_name = triple.get("subject", "None")
                object_name = triple.get("object", "None")

                if interaction:
                    # 상호작용(동사)을 이벤트의 핵심 속성으로 저장
                    self.graph.nodes[eid]['interaction'] = interaction
                
                # 주체(Subject) 노드 생성 및 연결
                if subject_name != "None":
                    subj_id = self.kb.get_or_create_id(subject_name)
                    self.graph.add_node(subj_id, type="object", label=subject_name)
                    # [관계] 주체 -> 이벤트 (Subject -> Event)
                    self.graph.add_edge(subj_id, eid, key="is_subject_of", label="수행")

                # 목적어(Object) 노드 생성 및 연결
                if object_name != "None":
                    obj_id = self.kb.get_or_create_id(object_name)
                    self.graph.add_node(obj_id, type="object", label=object_name)
                    # [관계] 이벤트 -> 목적어 (Event -> Object)
                    self.graph.add_edge(eid, obj_id, key="is_object_of", label="대상")
                
                # 속성(Property) 노드 생성 및 연결
                for prop_name in triple.get("properties", []):
                    prop_id = self.kb.get_or_create_id(prop_name)
                    self.graph.add_node(prop_id, type="property", label=prop_name)
                    # [관계] 이벤트 -> 속성 (Event -> Property)
                    self.graph.add_edge(eid, prop_id, key="has_property", label="속성")
        
        # 3. (하위 호환) 트리플로 처리되지 않은 나머지 개체/속성도 연결
        # (예: "목표", "사용자 상태" 등에서 파생된 'EIDOS', '나(=USER)')
        else:
            # (v1.0의 평탄화된 리스트 처리 로직 - 트리플이 없을 때만 실행)
            parsed_objects = analysis_dict.get("객체", [])
            parsed_interactions = analysis_dict.get("상호작용", [])
            parsed_properties = analysis_dict.get("속성", [])
            
            o_ids = [self.kb.get_or_create_id(o) for o in parsed_objects]
            p_ids = [self.kb.get_or_create_id(p) for p in parsed_properties]
            if parsed_objects:
                for i, (n, id) in enumerate(zip(parsed_objects, o_ids)):
                    t = "object"
                    if id == self.eidos_self_id_int: t = "metacognition"
                    self.graph.add_node(id, type=t, label=n)
                    if i == 0: self.graph.add_edge(id, eid, label="주체")
                    else: self.graph.add_edge(eid, id, label="대상")
            if parsed_interactions:
                self.graph.nodes[eid]['interaction'] = parsed_interactions[0]
            for n, id in zip(parsed_properties, p_ids):
                self.graph.add_node(id, type="property", label=n)
                self.graph.add_edge(eid, id, label="속성")

    async def answer_tom_query_async(self, user_query: str, current_emotion_vec: np.ndarray, chat_history: List[str]) -> Optional[str]:
         # ... (v11.1과 동일) ...
        print(f"🕵️ [ToM Query] Detected: '{user_query}'")
        prompt_extract = f"... Who is the subject of belief?\nQuery: \"{user_query}\"\nSubject: "
        agent_name = await get_llm_response_async(prompt_extract); agent_name = agent_name.strip()
        if "[N/A]" in agent_name or "[해당 없음]" in agent_name or not agent_name: print("  [ToM Query] Not a query about others' beliefs."); return None # 수정: [해당 없음] 처리
        agent_id = self.kb.get_or_create_id(agent_name)
        # 1. (수정) 해당 ID의 믿음 딕셔너리를 가져옵니다.
        agent_beliefs_dict = self.kb.agent_beliefs.get(agent_id, {}) 
        # 2. (수정) agent_id와 beliefs_dict를 모두 전달합니다.
        belief_str = self.kb.get_all_beliefs_str(agent_id, agent_beliefs_dict)
        print(f"  [ToM Query] Target: {agent_name} (ID: {agent_id})\n  [ToM Query] Beliefs:\n{belief_str}")
        history_str = "\n".join(self._normalize_history(chat_history)[-5:])
        prompt_answer = f"""You are EIDOS... User asked about '{agent_name}'s thoughts...\n['{agent_name}'s Beliefs]\n{belief_str}\n[History]\n{history_str}\n[Query]\n"{user_query}"\n[Instructions]...\n[EIDOS Answer (Based ONLY on '{agent_name}'s beliefs)]"""
        response_text = await get_llm_response_async(prompt_answer); return response_text.strip()

    async def _link_event_to_abstractions_realtime(
        self, 
        current_event_id: str, 
        current_event_vector: np.ndarray
    ):
        """
        [v12.1] (GPT Critique #3 대응)
        [병목 2 수정] CPU 집약적 루프를 to_thread로 분리
        [!!! v16.0 Fix] 그래프 수정은 메인 스레드와 Lock으로 보호 !!!
        """
        
        # [병목 2 수정] 이 함수는 CPU 바운드이므로, to_thread로 감쌀 헬퍼 함수 정의
        def _sync_link_task():
            edges_to_add = [] # 이 리스트를 메인 스레드로 반환하여 Lock과 함께 추가
            
            current_vec_1d = current_event_vector.flatten()
            current_vec_norm = np.linalg.norm(current_vec_1d)
            if current_vec_norm == 0: return []
            
            normalized_current_vec = current_vec_1d / current_vec_norm

            # Lock 없이 그래프를 읽는 작업은 비교적 안전하지만, 
            # Lock이 그래프에 대한 동시 쓰기를 막아줍니다.
            abstract_nodes = [
                (node, data['centroid']) 
                for node, data in self.graph.nodes(data=True) 
                if data.get('type') == 'abstract' and 'centroid' in data
            ]

            if not abstract_nodes: return []

            # --- 여기가 CPU 집약적 Blocking 루프 (Lock 없이) ---
            for abstract_node_id, centroid_vec in abstract_nodes:
                centroid_norm = np.linalg.norm(centroid_vec)
                if centroid_norm == 0: continue
                
                normalized_centroid_vec = centroid_vec / centroid_norm
                similarity = np.dot(normalized_current_vec, normalized_centroid_vec)
                
                if similarity > self.ABSTRACTION_LINK_THRESHOLD:
                    # [!!! 수정] 엣지 추가 대신 엣지 정보를 리스트에 수집
                    edges_to_add.append((current_event_id, abstract_node_id, similarity))
            # --- 루프 끝 ---
            return edges_to_add
        
        # [병목 2 수정] 헬퍼 함수를 별도 스레드에서 실행하여 엣지 정보 수집
        try:
            edges_to_add = await asyncio.to_thread(_sync_link_task)
            
            # [!!! 수정] 메인 스레드에서 Lock을 사용하여 엣지 추가 !!!
            if edges_to_add:
                async with self.graph_lock:
                    linked_count = 0
                    for src, dest, sim in edges_to_add:
                        if not self.graph.has_edge(src, dest, key="is_instance_of"):
                            self.graph.add_edge(
                                src, 
                                dest, 
                                key="is_instance_of", 
                                label="는...의 사례",
                                weight=float(sim)
                            )
                            linked_count += 1
                    
                    if linked_count > 0:
                        print(f"🔥 [Abstract-Realtime] 새 이벤트 '{current_event_id}'를 {linked_count}개의 추상 패턴과 실시간 연결! (Sim > {self.ABSTRACTION_LINK_THRESHOLD})")
        except Exception as e:
            print(f"❌ [Abstract-Realtime] 스레드 실행 중 오류: {e}")

    # [!!! v15.15 신규 함수 2: '반성 기능 강화' (다음 단계 계획) !!!]
    async def _create_code_modification_plan_async(
        self, 
        original_goal: str, 
        project_context: str
    ) -> Optional[List[Dict]]:
        """
        [v15.15] '프로젝트 문맥'을 바탕으로, '# TODO'나 'pass'를 채우기 위한
        'read_file -> write_text -> write_file' 계획을 'LLM 플래너'에게 요청합니다.
        """
        print(f"  🧠 [Reflect-Plan] '문맥' 기반 2단계 계획 생성 요청...")
        
        # 1. (LLM 호출) '문맥'을 _generate_dynamic_plan_async에 전달
        # (이것은 '자율 목표'가 아닌, '반성' 전용 플래너 호출임)
        try:
            planner_input = {
                "type": "POST_TASK_REFLECTION_V2", # [v15.15] 신규 타입
                "context": {
                    "original_goal": original_goal,
                    "project_context": project_context # 방금 읽은 파일들의 실제 내용
                }
            }
            
            # 2. 메인 플래너(_generate_dynamic_plan_async)를 '반성 모드'로 호출
            plan_json_str = await self._generate_dynamic_plan_async(
                text_input=planner_input,
                image_input=None,
                chat_history=[],
                multimodal_vector=np.zeros((1, LSTM_UNITS)),
                is_user_task=True # <<<--- [!!! 3. 이 인수를 추가하세요 !!!]
            )

            if plan_json_str.strip().startswith("Error:"):
                print(f"❌ [Reflect-Plan] 2단계 계획 생성 실패(LLM): {plan_json_str}")
                return None
            if "No further steps required." in plan_json_str:
                print(f"✅ [Reflect-Plan] 2단계 계획 불필요 (루프 종료).")
                return None
                
            plan_list = json.loads(plan_json_str)
            return plan_list

        except Exception as e:
            print(f"❌ [Reflect-Plan] 2단계 계획 생성 중 예외: {e}")
            return None

    def _extract_filenames_from_plan_helper(self, exec_task_json: str) -> List[str]:
        """ [v19.10 HELPER] JSON 계획을 파싱하여 eidos_files/ 하위의
            '모든 파일명' 리스트를 추출합니다. (평가 대상 식별용)
        """
        files_found = []
        try:
            task_list = json.loads(exec_task_json)
            if not isinstance(task_list, list): return []

            for task in task_list:
                args = task.get("args", {})
                if not args or not isinstance(args, dict): continue

                target_path = None

                if "file_structure" in args and isinstance(args["file_structure"], dict) and args["file_structure"]:
                    # [수정] 모든 키(파일 경로)를 추가
                    files_found.extend(list(args["file_structure"].keys()))
                    continue # 이 도구는 파일 목록만 있으므로 다음 task로
                
                elif "filepath" in args and isinstance(args["filepath"], str):
                    target_path = args["filepath"]
                elif "path" in args and isinstance(args["path"], str):
                    target_path = args["path"]

                if target_path:
                    norm_path = os.path.normpath(target_path)
                    prefix = "eidos_files" + os.sep
                    relative_path = None

                    if norm_path.startswith(prefix):
                        relative_path = norm_path[len(prefix):]
                    elif norm_path.startswith("." + os.sep + prefix):
                        relative_path = norm_path[len("." + os.sep + prefix):]
                    elif not os.path.isabs(norm_path): # (예: 'my_proj/main.py')
                         relative_path = norm_path
                    else:
                        continue 

                    files_found.append(relative_path) 
            
            return list(set(files_found)) # 중복 제거 후 반환
        except Exception:
            return files_found # 파싱 실패 시 현재까지 찾은 목록 반환

    async def _run_autosave_monitor(self):
        """ 5분마다 메모리를 자동으로 저장합니다. """
        AUTOSAVE_INTERVAL = 300 # 5분
        while True:
            try:
                await asyncio.sleep(AUTOSAVE_INTERVAL)
                print(f"💾 [AutoSave] 주기적 자동 저장 실행...")
                await asyncio.to_thread(self.save_memory)
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"❌ [AutoSave] 자동 저장 실패: {e}")
                await asyncio.sleep(60)

    async def _run_self_snapshot_loop(self, interval_sec: float = 1800.0):
        """ [Self Boundary 2026-04-27] 매 N분마다 SelfState 스냅샷 + 변화 감지 + 인과 귀속.

        기본 30분 주기. EIDOS_SELF_SNAPSHOT_INTERVAL 환경변수로 조정 가능
        (0 또는 음수면 비활성). 시작 직후 60초 grace — Core 부팅 안정화 대기.

        매 사이클:
            1. last_snapshot 로드 (이전 시점)
            2. 현재 SelfState 측정 + snapshot 저장
            3. 변화 감지 → eidos_changes.jsonl 기록
            4. is_significant 시 외부 사건 매칭 → eidos_attributions.jsonl 기록
            5. 의미 변화면 콘솔/Telegram 알림 (best-effort)
        """
        import os as _os
        try:
            _env_iv = _os.environ.get("EIDOS_SELF_SNAPSHOT_INTERVAL", "")
            if _env_iv.strip():
                _iv = float(_env_iv)
                if _iv <= 0:
                    print(f"🛑 [SelfSnapshot] EIDOS_SELF_SNAPSHOT_INTERVAL={_iv} — 루프 비활성")
                    return
                interval_sec = _iv
        except (ValueError, TypeError):
            pass

        await asyncio.sleep(60.0)  # 부팅 grace
        print(f"🛡️ [SelfSnapshot] 자기 상태 스냅샷 루프 시작 (주기: {interval_sec:.0f}초)")

        while True:
            try:
                await asyncio.sleep(interval_sec)
                # 백그라운드 스레드 실행 — IO 블로킹 회피
                summary = await asyncio.to_thread(self._self_snapshot_tick)
                if summary:
                    print(f"🛡️ [SelfSnapshot] {summary}")
            except asyncio.CancelledError:
                print("🛡️ [SelfSnapshot] 루프 종료.")
                break
            except Exception as _e_ss:
                print(f"⚠️ [SelfSnapshot] 사이클 오류 (계속): {_e_ss}")
                await asyncio.sleep(60.0)

    def _self_snapshot_tick(self) -> str:
        """1회 스냅샷 사이클 — sync. 자기 변화 한 줄 요약 반환 (또는 ""). """
        try:
            from eidos_self_monitor import get_monitor
            from eidos_change_detector import detect_and_log
            from eidos_causal_attributor import attribute_and_log

            mon = get_monitor()
            prev = mon.last_snapshot()
            curr = mon.snapshot()
            change = detect_and_log(prev, curr)
            if not change.is_significant:
                return f"변화 임계 미달 — health={curr.health_score:.2f} obs={curr.self_model_total_obs}"

            attr = attribute_and_log(change)
            cause_label = "(외부 원인 미발견)"
            if attr.best_cause is not None:
                bc = attr.best_cause
                cause_label = (
                    f"외부 원인 가설: {bc.get('record_type', '?')} "
                    f"({attr.cause_type}, conf={attr.confidence:.2f})"
                )
            return f"의미 변화 감지 — {change.summary} | {cause_label}"
        except Exception as _e_tick:
            return f"tick 예외: {_e_tick}"

    def _idle_autonomy_enabled(self) -> bool:
        """[2026-06-03] 유휴 자율 LLM 틱(heartbeat·autonav) 허용 여부.

        기본 False — 사용자가 가만히 있을 때 백그라운드가 LLM 을 호출해 토큰을 태우지
        않게 한다. eidos_settings.json 의 `autonomous_tick_enabled: true` 로 켜면
        예전처럼 자율 목표를 스스로 추진한다(토큰 소비). 채팅 `/자율` 로 토글 가능.
        ToM 에이전트·/dev·🎯자율실행 등 사용자가 직접 시작한 자율 작업은 별도 경로라 영향 없음.
        """
        try:
            sf = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "eidos_settings.json")
            if os.path.exists(sf):
                with open(sf, "r", encoding="utf-8") as f:
                    return bool(json.load(f).get("autonomous_tick_enabled", False))
        except Exception:
            pass
        return False

    async def _run_autonomous_heartbeat(self, interval_sec: float = 10.0):
        """
        [AGI Loop v1] 사용자 입력과 완전히 독립된 자율 실행 heartbeat.
        GUI의 QTimer나 채팅 입력 없이도 autonomous_tick_async를 주기적으로 실행.
        interval_sec: 기본 10초. 부하가 걱정되면 start_schedule_monitor 호출 시 변경 가능.
        """
        print(f"💓 [Heartbeat] 자율 틱 heartbeat 시작 (간격: {interval_sec}초)")
        while True:
            try:
                await asyncio.sleep(interval_sec)
                # dispatcher 전용 작업이 실행 중이면 tick 완전 스킵
                dispatcher_busy = any(
                    t.get("source") == "dispatcher" and
                    t.get("status") in ("RUNNING", "WAITING_USER", "PENDING")
                    for t in self.schedule
                )
                if dispatcher_busy:
                    continue  # GoalEval/GoalGen 실행 안 함

                # schedule에 실행할 태스크가 있을 때만 tick 실행 (불필요한 LLM 호출 방지)
                has_active_task = any(
                    t.get("status") in ("PENDING", "RUNNING")
                    for t in self.schedule
                )
                if has_active_task:
                    # [2026-06-03] 유휴 자율 LLM 틱 차단 — 기본 off(토큰 절약).
                    # 켜려면 eidos_settings.json autonomous_tick_enabled:true 또는 채팅 /자율.
                    if not self._idle_autonomy_enabled():
                        await self._process_internal_state()  # 가벼운 내부상태만(LLM 호출 X)
                    else:
                        print(f"💓 [Heartbeat] 활성 태스크 감지 → autonomous_tick_async 실행")
                        await self.autonomous_tick_async()
                else:
                    # 태스크가 없어도 내부 상태(지루함, 감정 decay 등)는 업데이트
                    await self._process_internal_state()
            except asyncio.CancelledError:
                print("💓 [Heartbeat] 자율 틱 heartbeat 종료.")
                break
            except Exception as e:
                print(f"⚠️ [Heartbeat] autonomous_tick 중 오류 (계속 실행): {e}")
                await asyncio.sleep(interval_sec)  # 오류 후에도 루프 유지

    # ──────────────────────────────────────────────────────────────────
    # [AutoNav] 자율항법 — 5분마다 제어실 컨텍스트 읽어 DC 분석 → 스케줄 자동 등록
    # ──────────────────────────────────────────────────────────────────

    async def _run_autonomous_nav_loop(self, interval_sec: float = 300.0):
        """
        [AutoNav] 5분(기본) 주기 자율항법 루프.

        - 제어실(schedule_planner_context.json)의 top_goal + situation_external 읽기
        - DC Reasoner로 "지금 당장 해야 할 행동" 분석
        - PENDING/RUNNING 태스크가 없을 때만 스케줄에 자동 등록
        - 사용자가 이미 태스크를 수동 등록했으면 간섭하지 않음

        start_schedule_monitor에서 loop.create_task()로 실행됨.
        """
        # Core 초기화 완료 대기 (시작 직후 LLM 과부하 방지)
        await asyncio.sleep(90.0)
        print(f"🧭 [AutoNav] 자율항법 루프 시작 (주기: {interval_sec:.0f}초)")

        while True:
            try:
                await asyncio.sleep(interval_sec)
                # [2026-06-03] 유휴 자율 LLM 차단 — 기본 off(토큰 절약). /자율 로 켤 수 있음.
                if not self._idle_autonomy_enabled():
                    continue
                await self._autonomous_nav_tick()
            except asyncio.CancelledError:
                print("🧭 [AutoNav] 루프 종료")
                break
            except Exception as e:
                print(f"⚠️ [AutoNav] 루프 오류 (계속 실행): {e}")
                await asyncio.sleep(interval_sec)

    async def _check_autonomous_triggers_async(self) -> bool:
        """[Wire 4] polling 자율 goal 트리거 3종 순회.

        flag off 면 즉시 False 반환 — hot path 비용 최소. 세 트리거 각자 자기
        cooldown·조건 체크를 가지므로 여기서는 단순 수집·주입만 담당.
        goal 하나라도 생성·주입하면 True (caller 는 AutoNav DC 로직 스킵 결정에
        활용). 예외는 흡수 — 자율 goal 실패가 AutoNav 본체를 망가뜨리지 않도록.
        """
        flags = _agi_loop_flags()
        if not flags.get("wire4_autogoal_broad", False):
            return False

        triggered = False
        triggers = [
            ("IdleTrigger",          getattr(self, "idle_trigger",          None)),
            ("DriftDipTrigger",      getattr(self, "drift_dip_trigger",     None)),
            ("KPIPredictionTrigger", getattr(self, "kpi_prediction_trigger", None)),
        ]
        for name, trg in triggers:
            if trg is None:
                continue
            try:
                goal = trg.poll(self)
            except Exception as _e:
                print(f"  ⚠️ [Wire4] {name}.poll 실패 (무시): {_e}")
                continue
            if not goal:
                continue
            try:
                from eidos_learning_loop import _inject_goal
                _inject_goal(self, goal)
                asyncio.create_task(
                    self._plan_and_inject_autonomous_goal_async(goal["name"])
                )
                triggered = True
            except Exception as _ie:
                print(f"  ⚠️ [Wire4] {name} goal 주입 실패 (무시): {_ie}")
        return triggered

    async def _autonomous_nav_tick(self):
        """
        [AutoNav] 틱 1회 실행.

        흐름:
          1. 활성 태스크(PENDING/RUNNING) 존재 시 → 스킵 (간섭 방지)
          2. [Wire 4] 자발적 goal 트리거 3종 폴링 (flag on일 때만)
          3. schedule_planner_context.json 에서 top_goal + situation 읽기
          4. top_goal 없으면 스킵 (제어실 미설정 상태)
          5. DC Reasoner reason_standalone() → 즉시 행동 도출
          6. 결론 파싱 → add_gui_task_async로 스케줄 등록
        """
        # ── 1. 활성 태스크 체크 ─────────────────────────────────────
        # [Fix 2026-04-21] dispatcher.start() 가 이전 세션 PENDING/RUNNING 에 찍어둔
        # _stale_session 마커는 dispatcher._pick_pending_task 에서 실행 대상에서
        # 제외된다. 그런데 여기서는 raw 카운트만 해서 stale 21개 묵으면 AutoNav
        # 가 영구 스킵 → 자율항법이 한 번도 안 돌던 증상(최신로그 1시간 내내 21개
        # 스킵 반복). dispatcher 와 동일한 기준으로 필터링해야 한다.
        async with self.schedule_lock:
            active_tasks = [
                t for t in self.schedule
                if t.get("status") in ("PENDING", "RUNNING")
                and not t.get("_stale_session")
            ]
            stale_count = sum(
                1 for t in self.schedule
                if t.get("status") in ("PENDING", "RUNNING")
                and t.get("_stale_session")
            )
        if active_tasks:
            _tag = f" (+stale {stale_count})" if stale_count else ""
            print(f"  🧭 [AutoNav] 활성 태스크 {len(active_tasks)}개 — 스킵{_tag}")
            return
        if stale_count:
            print(f"  🧭 [AutoNav] stale 세션 잔재 {stale_count}개 — 무시하고 진행")

        # ── 1.5 [Wire 4] 자발적 goal 폴링 (flag off 면 즉시 return False) ──
        # goal 하나라도 뽑혀 스케줄에 주입되면 DC 분석 경로는 건너뛴다
        # (중복 자율 goal 방지). flag off / 조건 미충족이면 기존 흐름 유지.
        try:
            if await self._check_autonomous_triggers_async():
                print("  🧭 [AutoNav] Wire 4 트리거가 자율 goal 주입 → DC 경로 스킵")
                return
        except Exception as _w4e:
            print(f"  ⚠️ [AutoNav] Wire 4 폴링 예외 (무시): {_w4e}")

        # ── 2. 제어실 컨텍스트 읽기 ─────────────────────────────────
        _CTX_FILE = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "eidos_files", "schedule_planner_context.json"
        )
        _SETTINGS_FILE = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "eidos_settings.json"
        )

        top_goal = ""
        situation_external = ""
        situation_internal = ""

        try:
            # settings.json에서 top_goal 우선 로드 (가장 최신)
            if os.path.exists(_SETTINGS_FILE):
                with open(_SETTINGS_FILE, "r", encoding="utf-8") as _sf:
                    top_goal = json.load(_sf).get("top_goal", "").strip()

            # schedule_planner_context.json에서 상황 보완
            if os.path.exists(_CTX_FILE):
                with open(_CTX_FILE, "r", encoding="utf-8") as _cf:
                    _ctx = json.load(_cf)
                if not top_goal:
                    top_goal = _ctx.get("top_goal", "").strip()
                situation_external = _ctx.get("situation_external", "").strip()
                situation_internal = _ctx.get("situation_internal", "").strip()
        except Exception as _re:
            print(f"  🧭 [AutoNav] 컨텍스트 읽기 실패 (스킵): {_re}")
            return

        # ── 3. top_goal 없으면 스킵 ─────────────────────────────────
        if not top_goal:
            print("  🧭 [AutoNav] 최상위 목표(top_goal) 미설정 — 스킵")
            return

        print(f"  🧭 [AutoNav] 틱 실행 | 목표: {top_goal[:40]}")

        # ── [AutoNav→Advisor 2026-05-17] 자율의 주경로 = advisor ──────────
        # 진단(최신로그): AutoNav 가 top_goal 을 DC reason_standalone 에
        # 넘기면 COMPLEX 강제로 일반론 1503자 *에세이* 만 뱉고 끝(파일·행동
        # 0). advisor.tick_async 는 *구체·게이트된·파일 생산* 제안
        # (explore→draft→advisor_draft.md)을 낸다 → 이쪽을 주경로로.
        # advisor 모듈이 통째로 불가할 때만 아래 기존 DC 경로로 폴백.
        # advisor.enabled=false 면 폴백(=구 동작) — 별도 키 없이 1플래그 원복.
        try:
            import eidos_proactive_advisor as _padv
            if _padv.settings().get("enabled", True):
                _adv_sug = await _padv.tick_async(core=self,
                                                  recent_context="")
                # advisor 가 처리(제안했거나 게이트로 보류)했으면 DC 에세이
                # 경로 진입 금지 — 5분마다 일반론 에세이 폭주 차단.
                if _adv_sug:
                    print(f"  🤖 [AutoNav→Advisor] 제안 등록: "
                          f"{str(_adv_sug.get('objective',''))[:50]}")
                else:
                    print("  🤖 [AutoNav→Advisor] 이번 틱 보류"
                          "(쿨다운/게이트) — DC 에세이 경로 스킵")
                return
            print("  🧭 [AutoNav] advisor.enabled=false — DC 폴백 평가")
        except Exception as _pae:
            print(f"  ⚠️ [AutoNav→Advisor] 모듈 불가 → DC 폴백 평가: {_pae}")

        # ── [트랙 B 2026-05-18 — 통합 Phase 2] DC 에세이 폴백 비활성 ──────
        # advisor 가 공식 단일 척추(advisor→ChainV2). advisor 모듈 불가/
        # 비활성 시 구 DC reason_standalone 폴백은 진단상 top_goal 을
        # COMPLEX 강제 → 1503자 일반론 *에세이* 만 뱉고 파일·행동 0
        # (메모리 diagnosis_autonav_essay_only). 척추가 공식화된 지금
        # 이 폴백은 불필요·유해 → 진입 차단(이번 틱 no-op). 폴백 본체와
        # EIDOS_AUTONAV_LIGHT 우회 코드는 **보존**(아래, 코드 삭제 아님).
        # 원복: 아래 _TRACKB_DISABLE_DC_FALLBACK = False (1줄).
        _TRACKB_DISABLE_DC_FALLBACK = True
        if _TRACKB_DISABLE_DC_FALLBACK:
            print("  🧭 [AutoNav] DC 에세이 폴백 비활성(트랙B 통합) — "
                  "이번 틱 no-op. (원복: _TRACKB_DISABLE_DC_FALLBACK=False)")
            return

        # [VerbEngine] situation_internal에서 수치 파싱 → DNA 등급 판정에 활용
        _nav_numeric = {}
        try:
            from eidos_verb_engine import extract_numeric_values as _env
            _nav_numeric = _env(
                situation_internal + " " + situation_external
            )
            if _nav_numeric:
                print(f"  🧭 [AutoNav] 수치 {len(_nav_numeric)}개 파싱: "
                      f"{list(_nav_numeric.keys())[:4]}")
        except Exception:
            pass

        # [CausalMatrix] 파싱된 수치 → _causal_x_vars 자동 갱신
        if _nav_numeric and self._causal_x_vars:
            try:
                updated = []
                for xv in self._causal_x_vars:
                    for k, v in _nav_numeric.items():
                        if any(kw in k for kw in xv.name.split("_")):
                            norm_v = min(1.0, max(0.0,
                                         v if v <= 1.0 else v / 100.0))
                            xv.current_value = norm_v
                            updated.append(xv.name)
                            break
                if updated:
                    print(f"  🧭 [CausalMatrix] X변수 갱신: {updated}")
                    # trainer가 있으면 동기화
                    if self._causal_trainer:
                        for xv in self._causal_x_vars:
                            self._causal_trainer.update_x(
                                xv.name, xv.current_value
                            )
            except Exception as _e_xv:
                pass

        # ── 4. DC Reasoner로 즉시 행동 도출 ─────────────────────────
        try:
            from eidos_dc_reasoner import DCReasoner as _DCR
            _reasoner = _DCR()

            _situation_parts = []
            if situation_external:
                _situation_parts.append("[외부 상황]\n" + situation_external)
            if situation_internal:
                _situation_parts.append("[내부 상황]\n" + situation_internal)
            _situation_str = "\n\n".join(_situation_parts) if _situation_parts else "특별한 상황 없음"
            _query = (
                "[최상위 목표]\n" + top_goal + "\n\n"
                + _situation_str + "\n\n"
                + "[요청]\n"
                "지금 당장 해야 할 가장 중요한 행동 하나를 도출하라.\n"
                "반드시 아래 형식으로 결론을 작성하라:\n"
                "결론: <행동 지시 한 문장>\n"
                "근거: <이유 한 문장>\n"
                "조건: <전제 조건 또는 없으면 없음>"
            )

            # [#28] AutoNav 폴백 DC = 경량 모드 강제 — COMPLEX 강제 우회로
            #   일반론 에세이 폭주 차단(스코프된 env, 호출 직후 복원).
            import os as _os_anl
            _anl_prev = _os_anl.environ.get("EIDOS_AUTONAV_LIGHT")
            _os_anl.environ["EIDOS_AUTONAV_LIGHT"] = "1"
            try:
                _dc_raw = await _reasoner.reason_standalone(
                    text=_query,
                    chat_history=[],
                    progress_callback=lambda m: print(f"    [AutoNav DC] {m}"),
                    top_goal=top_goal,   # [Phase 2.1] retrieval scope
                )
            finally:
                if _anl_prev is None:
                    _os_anl.environ.pop("EIDOS_AUTONAV_LIGHT", None)
                else:
                    _os_anl.environ["EIDOS_AUTONAV_LIGHT"] = _anl_prev

            _dc_result = ""
            if isinstance(_dc_raw, dict):
                _dc_result = _dc_raw.get("result", "")
            elif isinstance(_dc_raw, str):
                _dc_result = _dc_raw

            if not _dc_result:
                print("  🧭 [AutoNav] DC 분석 결과 없음 — 스킵")
                return

        except Exception as _dce:
            print(f"  🧭 [AutoNav] DC 분석 실패 (스킵): {_dce}")
            return

        # ── 5. 결론 파싱 → 스케줄 등록 ──────────────────────────────
        try:
            import re as _re
            from datetime import datetime as _dt

            # "결론: ..." 추출
            _m = _re.search(r"결론[：:]\s*(.+)", _dc_result)
            if not _m:
                # 결론 태그 없으면 첫 줄 사용
                _action = _dc_result.strip().split("\n")[0][:100]
            else:
                _action = _m.group(1).strip()[:100]

            if not _action:
                print("  🧭 [AutoNav] 행동 추출 실패 — 스킵")
                return

            # 중복 등록 방지: 동일 prompt가 이미 스케줄에 있으면 스킵
            async with self.schedule_lock:
                _existing = [
                    t for t in self.schedule
                    if _action[:30] in t.get("task_prompt", "")
                ]
            if _existing:
                print(f"  🧭 [AutoNav] 동일 태스크 이미 존재 — 스킵: {_action[:40]}")
                return

            _now_str = _dt.now().strftime("%Y-%m-%d %H:%M")
            _task_name = f"[AutoNav] {_action[:50]}"

            await self.add_gui_task_async(
                name=_task_name,
                due_date=_now_str,
                prompt=(
                    _action + "\n\n"
                    + "[자율항법 컨텍스트]\n"
                    + "목표: " + top_goal + "\n"
                    + "DC 분석:\n" + _dc_result[:500]
                ),
            )
            print(f"  ✅ [AutoNav] 스케줄 등록 완료: {_task_name}")
            # [패치] AutoNav 결론 → situation_external 자동 기록
            try:
                from eidos_situation_bridge import write_external_situation
                write_external_situation(_dc_result, source_tag="AutoNav")
            except Exception as _sbe:
                print(f"  ⚠️ [AutoNav] situation 기록 실패 (무시): {_sbe}")

        except Exception as _se:
            print(f"  🧭 [AutoNav] 스케줄 등록 실패 (스킵): {_se}")

    # start_schedule_monitor 함수를 수정하여 자동 저장도 함께 시작
    async def start_schedule_monitor(self):
        """ 스케줄 모니터와 자동 저장 모니터를 시작합니다. """
        if self._schedule_monitor_task is None or self._schedule_monitor_task.done():
            loop = asyncio.get_running_loop()
            self._schedule_monitor_task = loop.create_task(self._run_schedule_monitor())
            
            # [v-New] 자동 저장 태스크 시작
            loop.create_task(self._run_autosave_monitor())

            # [AGI Loop v1] 사용자 입력과 독립된 자율 틱 heartbeat 시작
            loop.create_task(self._run_autonomous_heartbeat(interval_sec=10.0))

            # [PatrolAgent] 1시간 주기 시장 정찰 + 목표 재조정
            try:
                from eidos_patrol_agent import PatrolAgent
                self._patrol_agent = PatrolAgent(self)
                loop.create_task(self._patrol_agent.run_loop())
                print("🔭 [PatrolAgent] 자율 정찰 에이전트 시작됨 (situation_external 기록 활성).")
            except Exception as _patrol_e:
                print(f"⚠️ [PatrolAgent] 로드 실패 (무시): {_patrol_e}")

            # [AutoNav] 5분 주기 자율항법 루프
            loop.create_task(self._run_autonomous_nav_loop(interval_sec=300.0))

            # [Self Boundary 2026-04-27] 자기 상태 스냅샷 루프 (기본 30분).
            # eidos_self_monitor.snapshot → change_detector → causal_attributor 풀 사이클.
            # EIDOS_SELF_SNAPSHOT_INTERVAL=0 환경변수로 비활성 가능.
            try:
                loop.create_task(self._run_self_snapshot_loop(interval_sec=1800.0))
            except Exception as _e_ss_boot:
                print(f"⚠️ [SelfSnapshot] 루프 등록 실패 (무시): {_e_ss_boot}")

            # [Autonomous Phase 1B 2026-04-27] 자율 실행 매니저 부트스트랩 —
            # chain_starter 콜백 + chain_complete/failed 리스너 등록.
            # /explore + ✅ 승인 → chain 자동 시작·완료 알림 라이프사이클 활성.
            # 실패해도 매니저 자체는 동작 (수동 진단 모드).
            try:
                from eidos_autonomous_runner import bootstrap_with_core as _ar_bootstrap
                _ar_bootstrap(self)
            except Exception as _e_ar:
                print(f"⚠️ [AutoRunner] 부트스트랩 실패 (무시): {_e_ar}")

            print(f"⏰ [Monitor] 스케줄링, 자동 저장(5분), 자율 heartbeat(10초), 자율항법(5분), 자기 스냅샷(30분) 시작됨.")

    async def delete_scheduled_task_async(self, task_prompt: str) -> bool:
        """ 특정 프롬프트를 가진 작업을 스케줄에서 삭제합니다. """
        async with self.schedule_lock:
            initial_count = len(self.schedule)
            # 프롬프트 내용이 일치하는 작업을 필터링하여 제거
            self.schedule = [t for t in self.schedule if t.get("task_prompt") != task_prompt]
            
            if len(self.schedule) < initial_count:
                print(f"🗑️ [Schedule] 작업 삭제 완료: '{task_prompt}'")
                self.save_memory() # 변경 사항 저장
                return True
            else:
                print(f"⚠️ [Schedule] 삭제 실패: 찾을 수 없음 ('{task_prompt}')")
                return False

    # ----------------------------------------------------------------------
    # --- [v12.0 Autonomous Goal] 신규 메서드: 자율 목표 생성 ---
    # ----------------------------------------------------------------------
    async def _update_goal_system_async(self, external_event: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        [A1] GoalStackEvaluator 위임.
        스케줄 비어있을 때만 실행 — 기존 ROI + WorldModel 통합 경로.
        """
        # 스케줄에 이미 PENDING/RUNNING이 있으면 새 목표 생성 건너뜀
        async with self.schedule_lock:
            active = [t for t in self.schedule
                      if t.get("status") in ("PENDING", "RUNNING")]
        if active:
            return None

        # 반성 큐 우선
        if self.reflection_queue:
            print("🎯 [A1 GoalSystem] Reflection queue active → skipping goal gen.")
            return None

        # GoalStackEvaluator 통해 P3/P4 실행
        return await self.goal_stack_evaluator.select_next()

    async def _execute_task(self, task_plan_json: str, project_dir_context: Optional[str] = None) -> str:
        """
        [v18.23 수정] 'write_project_files_async' 실행 시 외부 모듈에 의존하지 않고
        Core가 직접 OS 파일 시스템에 파일을 쓰도록 강제합니다. (Direct Write Injection)
        또한 보안 경로 검사 함수 이름을 통일하여 NameError를 방지합니다.
        """
        print(f"⚙️ [Execution v18.23] 작업 계획 실행 시작")
        
        # 1. 사용 가능한 도구(함수) 맵핑
        available_tool_functions = {
            "publish_social_media_post_async": (
                lambda **kwargs: publish_social_media_post_async(**kwargs, core_instance=self)
            ),
            "execute_python_file": execute_python_file,
            "perform_web_search": perform_web_search,
            "write_text": write_text,
            "read_file": read_file,
            "write_file": write_file,
            "read_code_skeleton": read_code_skeleton,
            "calculate_math": calculate_math,
            "write_project_files_async": write_project_files_async,
            "write_complex_code_iteratively": write_complex_code_iteratively,
            "reload_tool_registry_async": self.reload_tool_registry_async,
            "generic_calendar_create": generic_calendar_create,
            "generic_calendar_search": generic_calendar_search,
            "generic_calendar_delete": generic_calendar_delete,
            # ── Google Calendar 직접 연동 툴 ──────────────────────────
            "gcal_list_events":  self.gcal_list_events,
            "gcal_create_event": self.gcal_create_event,
            "gcal_delete_event": self.gcal_delete_event,
            # ── Gmail 직접 연동 툴 ────────────────────────────────────
            "gmail_list_messages": self.gmail_list_messages,
            "gmail_get_message":   self.gmail_get_message,
            "gmail_send":          self.gmail_send,
            "gmail_create_draft":  self.gmail_create_draft,
            "gmail_mark_read":     self.gmail_mark_read,
            "gmail_archive":       self.gmail_archive,
            "gmail_trash":         self.gmail_trash,
            "register_object_wave": (
                lambda **kwargs: register_object_wave(**kwargs, core_instance=self)
            ),
            "get_adjacent_object_analysis": (
                lambda **kwargs: get_adjacent_object_analysis(**kwargs, core_instance=self)
            ),
            "publish_marketing_content_async": (
                lambda **kwargs: publish_marketing_content_async(**kwargs, core_instance=self)
            ),
            # [PromptCanvas 연결] 저장된 에이전트 tool
            "list_saved_agents": list_saved_agents,
            "run_saved_agent":   run_saved_agent,
        }
        
        if self.video_processor:
            video_tools = {
                "video_analyze_structure": self.video_processor.analyze_video_structure_async,
                "video_edit_subclip": self.video_processor.edit_video_async,
                "video_add_subtitles": self.video_processor.add_subtitles_async,
                "video_merge_clips": self.video_processor.merge_video_clips_async,
            }
            available_tool_functions.update(video_tools)
        
        available_tool_functions.update(self.dynamic_tool_functions)
        
        # [GoalPlan] code_fix_loop 도구 등록
        available_tool_functions["code_fix_loop"] = (
            lambda **kwargs: self._run_code_fix_loop(**kwargs)
        )
        
        BASE_PATH = self.project_root
        
        if project_dir_context:
            safe_base_path = os.path.normpath(os.path.join(BASE_PATH, project_dir_context))
        else:
            safe_base_path = BASE_PATH

        # [보안 헬퍼 함수] - 이름 통일됨 (_check_and_correct_path)
        def _check_and_correct_path(rel_path: str, base_dir: str, must_exist: bool = False) -> str:
             sanitized_rel_path = rel_path.lstrip(os.sep + '/') 
             abs_target = os.path.normpath(os.path.join(base_dir, sanitized_rel_path))
             
             # Path Traversal 방지
             if os.path.commonprefix([abs_target, base_dir]) != base_dir:
                 raise PermissionError(f"Security Error: Path '{rel_path}' is outside project root.")

             if must_exist and not os.path.exists(abs_target):
                 raise FileNotFoundError(f"File not found: {rel_path}")
                 
             return abs_target

        # 2. JSON 계획 파싱
        try:
            task_list = json.loads(task_plan_json)
            if not isinstance(task_list, list): raise ValueError("Not a list")
        except Exception as e:
            print(f"❌ [Execution] JSON 파싱 실패: {e}")
            return f"EVENT: 파싱 실패 ({e})"

        # 3. 계획 실행
        previous_step_result = "" 
        final_result = ""

        for i, task in enumerate(task_list):
            try:
                # tool 키 fallback: LLM이 다른 키를 쓸 수 있으므로 여러 키 시도
                tool_name = (
                    task.get("tool") or
                    task.get("name") or
                    task.get("function") or
                    task.get("action") or
                    task.get("tool_name") or
                    ""
                )
                args_dict = task.get("args", task.get("arguments", task.get("params", {})))
                if not isinstance(args_dict, dict):
                    args_dict = {}

                # [호환] file_structure_json → file_structure 자동 변환
                if "file_structure_json" in args_dict and "file_structure" not in args_dict:
                    val = args_dict.pop("file_structure_json")
                    if isinstance(val, str):
                        try:
                            args_dict["file_structure"] = json.loads(val)
                        except Exception:
                            args_dict["file_structure"] = {}
                    elif isinstance(val, dict):
                        args_dict["file_structure"] = val

                # [호환] write_complex_code_iteratively 인자 별칭 자동 변환
                # LLM이 구버전 툴 설명(requirement/filename)을 참고했을 경우 대비
                if tool_name == "write_complex_code_iteratively":
                    if "requirement" in args_dict and "task_description" not in args_dict:
                        args_dict["task_description"] = args_dict.pop("requirement")
                        print("  [ArgAlias] requirement → task_description 자동 변환")
                    if "filename" in args_dict and "filepath" not in args_dict:
                        args_dict["filepath"] = args_dict.pop("filename")
                        print("  [ArgAlias] filename → filepath 자동 변환")
                    if "description" in args_dict and "task_description" not in args_dict:
                        args_dict["task_description"] = args_dict.pop("description")
                        print("  [ArgAlias] description → task_description 자동 변환")

                if not tool_name:
                    print(f"  ⚠️ [Exec Step {i+1}] tool 키 없음, 건너뜀. task={str(task)[:100]}")
                    continue
                print(f"  [Exec Step {i+1}/{len(task_list)}] Tool: '{tool_name}'")

                # [경로 보정 및 보안 적용]
                is_video_tool = tool_name.startswith("video_")

                # --- 경로 처리 헬퍼 함수 정의 ---
                def _sanitize_path(path: str) -> str:
                    """ LLM이 생성한 경로에서 'eidos_files/' 중복 프리픽스를 제거합니다. """
                    if not path: return ""
                    
                    path_to_check = os.path.normpath(path)
                    
                    # LLM이 실수로 붙일 수 있는 'eidos_files' 프리픽스 목록
                    prefixes = ["eidos_files", "." + os.sep + "eidos_files"]
                    
                    for prefix in prefixes:
                        norm_prefix = os.path.normpath(prefix)
                        # 경로가 'eidos_files' 또는 '.\eidos_files'로 시작하는지 확인
                        if path_to_check.startswith(norm_prefix):
                            # 프리픽스 이후의 경로만 사용하고, 시작 부분의 구분자(os.sep 또는 '/')를 제거
                            clean_path = path_to_check[len(norm_prefix):].lstrip(os.sep).lstrip('/')
                            print(f"  -> Path prefix '{norm_prefix}' removed. Clean path: {clean_path}")
                            return clean_path
                    return path_to_check
                # -----------------------------------

                # A. 파일/프로젝트 쓰기 도구 (경로 중복 제거 및 보안 검사)
                # [FIX] execute_python_file 추가
                if tool_name in (
                    "write_file", "read_file", "write_project_files_async", "execute_python_file"
                ) and not is_video_tool:
                    
                    # 1. 'write_project_files_async' 처리 (가장 복잡한 케이스)
                    if tool_name == "write_project_files_async":
                        original_file_dict = args_dict.get("file_structure", {})
                        corrected_file_dict = {}
                        
                        if not isinstance(original_file_dict, dict):
                             original_file_dict = {}
                        
                        _GLOB_CHARS = set('*?[')
                        for rel_path, content in original_file_dict.items():
                            if any(c in rel_path for c in _GLOB_CHARS):
                                print(f"  ⚠️ [PathGuard] glob 패턴 키 스킵: {rel_path}")
                                continue
                            clean_rel_path = _sanitize_path(rel_path)
                            safe_abs_path = _check_and_correct_path(clean_rel_path, safe_base_path)
                            corrected_file_dict[safe_abs_path] = content
                            
                        args_dict["file_structure"] = corrected_file_dict
                    
                    # 2. 'write_file', 'read_file', 'execute_python_file' 처리
                    # [FIX] execute_python_file 추가
                    elif tool_name in ("write_file", "read_file", "execute_python_file"):
                        original_path = args_dict.get("filepath", args_dict.get("path"))
                        
                        if original_path:
                            clean_path = _sanitize_path(original_path)
                            
                            try:
                                # [FIX] 실행하려는 파일도 반드시 존재해야 하므로 must_exist 조건 추가
                                safe_abs_path = _check_and_correct_path(
                                    clean_path, 
                                    safe_base_path, 
                                    must_exist=(tool_name in ("read_file", "execute_python_file"))
                                )
                                args_dict["filepath"] = safe_abs_path
                                args_dict.pop("path", None)
                                
                            except FileNotFoundError as e:
                                # [Autofix Fix] FileNotFoundError는 코드 버그가 아닌 '환경' 문제입니다.
                                # Traceback을 반환하면 Autofix가 핵심 로직(core.py)을 수정하려 하므로,
                                # 대신 명확한 '오류 이벤트' 문자열을 반환하여 재계획(re-plan)을 유도합니다.
                                print(f"❌ [Execution Error] File not found: {clean_path}. Returning structured error string.")
                                # `raise e`를 제거하고, Traceback이 없는 오류 메시지를 반환합니다.
                                return f"EVENT: 실행 중 오류 발생:\nFileNotFoundError: The specified file '{clean_path}' does not exist in the project directory '{project_dir_context or ''}'."

                # B. 비디오 도구 보안
                if is_video_tool:
                    if "video_path" in args_dict:
                        # (함수 이름 수정됨)
                        safe_input = _check_and_correct_path(args_dict["video_path"], os.path.abspath(os.sep), True)
                        args_dict["video_path"] = safe_input
                    if "output_path" in args_dict:
                        # (함수 이름 수정됨)
                        args_dict["output_path"] = _check_and_correct_path(args_dict["output_path"], BASE_PATH)
                    if "clip_paths_json" in args_dict:
                        try:
                            paths = json.loads(args_dict["clip_paths_json"])
                            # (함수 이름 수정됨)
                            safe_paths = [_check_and_correct_path(p, os.path.abspath(os.sep), True) for p in paths]
                            args_dict["clip_paths_json"] = json.dumps(safe_paths)
                        except Exception as e:
                            raise PermissionError(f"Clip path validation failed: {e}")

                # C. 이미지 도구 등 기타 경로 인수 처리
                if tool_name == "analyze_image": 
                    # 1. LLM이 보낸 키(filepath 또는 path)를 확인
                    target_key = "filepath" if "filepath" in args_dict else "path"
                    original_path = args_dict.get(target_key)
    
                    if original_path:
                        clean_path = _sanitize_path(original_path)
                        safe_abs_path = _check_and_correct_path(clean_path, safe_base_path, must_exist=True)
        
                        # 2. ✅ 함수가 예상하는 'filepath'로 키를 통일하여 전달
                        args_dict["filepath"] = safe_abs_path
        
                        # 3. 불필요하거나 중복된 키는 제거 (image_path가 있다면 삭제)
                        if "path" in args_dict: args_dict.pop("path")
                        if "image_path" in args_dict: args_dict.pop("image_path")

                # 5. 인수 준비 ($PREV_STEP_RESULT 처리)
                for key, value in args_dict.items():
                    if isinstance(value, str) and "$PREV_STEP_RESULT" in value:
                        # [!!! FIX 2: JSON 인덱싱 구문(예: $PREV_STEP_RESULT[0]) 차단 및 치환 !!!]
                        if re.search(r'\$PREV_STEP_RESULT\[\d+\]', value):
                            # JSON 인덱싱 구문이 있으면 VETO가 작동하지 못하도록 임의로 치환합니다.
                            # 올바른 JSON 파싱 로직을 LLM이 구현해야 하므로, 여기서는 경고만 줍니다.
                            print(f"🚨 [VETO Warning] Args '{key}' contains JSON indexing notation. Attempting full string replacement.")
                            
                        args_dict[key] = value.replace("$PREV_STEP_RESULT", previous_step_result)

                # ---------------------------------------------------------
                # 🔥 [Direct Write Injection] 외부 모듈 무시하고 직접 쓰기
                # ---------------------------------------------------------
                if tool_name == "write_project_files_async":
                    files_to_write = args_dict.get("file_structure", {})
                    created_files = []
                    
                    print(f"  ⚡ [Direct Write] Core가 직접 파일 {len(files_to_write)}개 생성을 시작합니다...")
                    
                    _GLOB_CHARS = set('*?[')
                    for abs_path, content in files_to_write.items():
                        # 0. glob 패턴 경로 방어 (Windows [Errno 22] 원인)
                        if any(c in abs_path for c in _GLOB_CHARS):
                            print(f"  ⚠️ [DirectWrite] glob 패턴 경로 스킵: {abs_path}")
                            continue
                        # 1. 디렉토리 생성 (dirname이 빈 문자열일 때 makedirs 호출 방지)
                        dir_path = os.path.dirname(abs_path)
                        if dir_path:
                            os.makedirs(dir_path, exist_ok=True)
                        
                        # 2. 파일 쓰기 (UTF-8)
                        with open(abs_path, "w", encoding="utf-8") as f:
                            f.write(content)
                        
                        created_files.append(os.path.basename(abs_path))
                        print(f"    📄 Created: {abs_path}")
                    
                    # 결과 생성 (외부 함수 호출 건너뜀)
                    current_result = json.dumps({
                        "action": "write_project_files", 
                        "status": "success", 
                        "files_written": list(files_to_write.keys())
                    })
                    print(f"  ✅ [Direct Write] 파일 생성 완료: {', '.join(created_files)}")
                    
                    previous_step_result = current_result
                    final_result = current_result
                    continue # 다음 루프로 이동 (func_to_call 호출 안 함)
                # ---------------------------------------------------------

                # 6. 그 외 도구 실행
                func_to_call = available_tool_functions.get(tool_name)
                if not func_to_call:
                    # Fallback lookup
                    if hasattr(execution_module, tool_name):
                        func_to_call = getattr(execution_module, tool_name)
                    elif tool_name in self.dynamic_tool_functions:
                        func_to_call = self.dynamic_tool_functions[tool_name]

                if func_to_call:
                    current_result = await func_to_call(**args_dict)
                    # [Defensive Programming] 툴 함수가 None을 반환하는 경우,
                    # 다운스트림에서 TypeError가 발생하는 것을 방지하기 위해 빈 문자열로 변환합니다.
                    if current_result is None:
                        print(f"  [Execution] ⚠️ 경고: 도구 '{tool_name}'이(가) None을 반환했습니다. 빈 문자열로 대체됩니다.")
                        current_result = ""
                    previous_step_result = current_result
                    final_result = current_result
                else:
                    print(f"❌ Tool '{tool_name}' not found anywhere.")
                    # [ToolFactory] 알 수 없는 도구 → 자동 도구 합성 시도
                    synthesis_result = await self._on_missing_tool_async(
                        tool_name=tool_name,
                        args_dict=args_dict,
                        task_context=text_input_str if isinstance(text_input_str, str) else str(text_input),
                    )
                    if synthesis_result:
                        current_result = synthesis_result
                        previous_step_result = synthesis_result
                        final_result = synthesis_result
                    else:
                        # [Bug Fix] 합성 실패 시 명시적 오류 메시지 — silent fail 방지
                        _fail_msg = f"[ToolFactory] '{tool_name}' 합성 실패. 해당 도구를 수동으로 추가하거나 다른 방법을 시도하세요."
                        print(f"  [ToolFactory] 합성 최종 실패: '{tool_name}'")
                        current_result = _fail_msg
                        previous_step_result = _fail_msg
                        final_result = _fail_msg

            except Exception as e:
                import traceback
                print(f"❌ [Execution Error] {tool_name}: {e}")
                # Autofix가 분석할 수 있도록, 단순 메시지 대신 전체 Traceback을 반환합니다.
                error_traceback_str = traceback.format_exc()
                # print_exc()는 stdout으로만 출력하므로, format_exc()로 문자열을 얻습니다.
                return f"EVENT: 실행 중 오류 발생:\n{error_traceback_str}"

        print(f"✅ [Execution] 완료.")
        return f"EVENT: 작업 계획 실행 완료. 최종 결과: {final_result}"

    def _extract_project_dir_from_plan_helper(self, exec_task_json: str) -> Optional[str]:
        """ [v16.3 HELPER] JSON 계획을 파싱하여 eidos_files/ 하위의
            프로젝트 디렉토리 이름(첫 번째 폴더)을 추출합니다. (Sync)
        """
        try:
            task_list = json.loads(exec_task_json)
            if not isinstance(task_list, list): return None

            for task in task_list:
                args = task.get("args", {})
                if not args or not isinstance(args, dict): continue

                target_path = None

                if "file_structure" in args and isinstance(args["file_structure"], dict) and args["file_structure"]:
                    target_path = list(args["file_structure"].keys())[0]
                elif "filepath" in args and isinstance(args["filepath"], str):
                    target_path = args["filepath"]
                elif "path" in args and isinstance(args["path"], str):
                    target_path = args["path"]

                if target_path:
                    norm_path = os.path.normpath(target_path)
                    prefix = "eidos_files" + os.sep
                    relative_path = None

                    if norm_path.startswith(prefix):
                        relative_path = norm_path[len(prefix):]
                    elif norm_path.startswith("." + os.sep + prefix):
                        relative_path = norm_path[len("." + os.sep + prefix):]
                    else:
                        continue 

                    parts = relative_path.split(os.sep)
                    if len(parts) > 1:
                        return parts[0] # 예: 'proj'
            return None
        except Exception:
            return None # 파싱 실패 시 None

    # [v14.5 AGI] (1/2) 수정된 함수: 행동 결과(지식)를 '보상 기반 신뢰도'와 함께 동화
    async def _assimilate_knowledge_async(self, goal_name: str, execution_result_event: str, reward: float): # <-- reward 파라미터 추가
        """
        [v14.5] 실행 결과를 OIA로 파싱하고, 행동의 '보상(reward)'을 기반으로
        '신뢰도(confidence)'를 설정하여 지식으로 그래프에 동화시킵니다.
        """
        print(f"🧠 [AI Learn v14.5] 지식 동화 시작... (Goal: {goal_name}, Reward: {reward:.3f})")

        # ... (지식 텍스트 추출 - v14.1과 동일) ...
        knowledge_text = ""
        if "최종 결과: " in execution_result_event:
            knowledge_text = execution_result_event.split("최종 결과: ", 1)[-1]
        if not knowledge_text or len(knowledge_text) < 10:
            print(f"  [AI Learn v14.5] 동화할 유의미한 지식 텍스트가 없습니다.")
            return

        print(f"  [AI Learn v14.5] 지식 텍스트 ({len(knowledge_text)}자) OIA 파싱 중...")

        try:
            # [수정] llm_analysis -> analysis_dict로 변경
            analysis_dict = await analyze_event_with_llm_async(knowledge_text, None) 
            if not analysis_dict:
                print(f"  [AI Learn v14.9] 지식 텍스트 OIA 분석 실패 (JSON 오류).")
                return

            # [수정] 딕셔너리에서 직접 가져오기
            parsed_objects = analysis_dict.get("객체", [])
            parsed_interactions = analysis_dict.get("상호작용", [])
            parsed_properties = analysis_dict.get("속성", [])
            concept_ids = (
                [self.kb.get_or_create_id(o) for o in parsed_objects] +
                [self.kb.get_or_create_id(i) for i in parsed_interactions] +
                # [!!! v13.0 오타 수정: 'o' -> 'p' !!!]
                [self.kb.get_or_create_id(p) for p in parsed_properties] 
            )
            if not concept_ids:
                print(f"  [AI Learn v14.5] 지식 텍스트에서 OIA 개념을 추출하지 못했습니다.")
                return

            # ... ('지식 동화 이벤트' 생성 A-XXX - v14.1과 동일) ...
            assimilation_event_id = f"A-{len(self.event_log) + 1:03d}"
            dummy_vector = np.zeros(LSTM_UNITS)
            self.event_log.append(
                (dummy_vector, concept_ids, assimilation_event_id, parsed_interactions)
            ) #
            self.event_id_to_concepts[assimilation_event_id] = concept_ids

            # ... (그래프 노드 추가 - v14.1과 동일) ...
            self._update_visualization_graph(analysis_dict)

            # [!!! v14.5 AGI 수정: 보상 기반 신뢰도 설정 !!!]
            # 5. '지식' 노드에 속성 설정 및 '목표'와 연결
            goal_node_id = self.kb.get_or_create_id(goal_name, item_type="goal") #

            if self.graph.has_node(assimilation_event_id):
                # 기본 신뢰도 0.5 설정
                base_confidence = 0.5
                # 보상 값(-N ~ +N)을 신뢰도 조정폭(-0.5 ~ +0.5)으로 변환 (간단한 Sigmoid 유사 변환)
                # reward가 크면 confidence_adjustment가 0.5에 가까워지고, 작으면 -0.5에 가까워짐
                confidence_adjustment = (reward / (1 + abs(reward))) * 0.5
                final_confidence = np.clip(base_confidence + confidence_adjustment, 0.1, 1.0) # 0.1 ~ 1.0 범위

                # 노드 속성 설정
                self.graph.nodes[assimilation_event_id]['type'] = 'knowledge'
                self.graph.nodes[assimilation_event_id]['content'] = knowledge_text[:500]
                self.graph.nodes[assimilation_event_id]['confidence'] = float(final_confidence) # <-- 신뢰도 저장

                # 엣지 연결 (v14.1과 동일)
                self.graph.add_edge(
                    goal_node_id,
                    assimilation_event_id,
                    key="resulted_in",
                    label="지식 획득"
                )

            print(f"✅ [AI Learn v14.5] 지식 동화 완료! (Goal: '{goal_name}' -> Knowledge: '{assimilation_event_id}', Confidence: {final_confidence:.3f})")

        except Exception as e:
            print(f"❌ [AI Learn v14.5] 지식 동화 중 오류: {e}")

    def _select_current_goal(self):
        """ [AGI-3] 목표 계층화
        미완료 + 'active' 상태인 목표 중 가장 우선순위(priority)가 높은 목표를
        self.current_goal로 선택합니다. (경쟁 메커니즘)
        """
        # [수정] 'active' 상태이고 'completed'되지 않은 목표만 경쟁 후보
        candidate_goals = [
            g for g in self.long_term_goals 
            if not g.completed and g.status == "active"
        ]
        
        if not candidate_goals:
            if self.current_goal is not None:
                print("🎯 [Goal Priority] 활성 (active) 목표 없음. (paused/completed)")
                self.current_goal = None
            return

        # [경쟁] 우선순위가 가장 높은 목표를 선택
        best_goal = max(candidate_goals, key=lambda g: g.priority)
        
        if self.current_goal != best_goal.name:
            self.current_goal = best_goal.name
            print(f"🎯 [Goal Priority] 활성 목표 변경됨 (경쟁 승리) -> '{self.current_goal}' (Priority: {best_goal.priority})")

    async def generate_explanation_async(
        self,
        context: ExplanationContext
    ) -> ExplanationResult:
        """
        [설명 기능 MVP v2]
        설명 요청을 받아 explanation schema 생성을 위임하고,
        시스템 안정성을 보장하는 얇은 래퍼 함수.
        """
        preview = (context.selected_text or "")[:20].replace("\n", " ")
        print(f"🧠 [Explanation MVP v2] '{preview}...' 슬라이드 기반 설명 생성 시작...")

        try:
            return await generate_explanation_schema_async(context)

        except Exception as e:
            # 이 레벨에서는 "절대" 슬라이드 구조를 직접 만들지 않음
            print(f"❌ [Explanation MVP v2] Core 레벨 예외: {e}")

            # 최후의 최후 안전망 (거의 발생하면 안 됨)
            return ExplanationResult(
                title="AI 설명 생성 실패",
                key_points=[
                    "설명 생성 파이프라인에서 심각한 오류가 발생했습니다.",
                    "시스템 로그를 확인해주세요."
                ],
                slides=[]
            )

    async def _finalize_tick(self):
        """
        자율 실행 틱의 마지막에 공통적으로 수행되는 작업을 처리합니다.
        (현재는 향후 확장을 위한 placeholder)
        """
        pass

    async def autonomous_tick_async(self) -> Optional[Dict[str, Any]]:
        """
        EIDOS 자율 실행 틱 (리팩토링 버전)
        """

        task_to_return = None

        try:
            await self._process_internal_state()

            # [A2] 감정 상태가 행동에 미치는 영향 로그 (10틱마다)
            self._emotion_tick_counter = getattr(self, "_emotion_tick_counter", 0) + 1
            if self._emotion_tick_counter % 10 == 0 and hasattr(self, "_emotion_adapter"):
                self._emotion_adapter.log_state()

            task = await self._select_next_task()

            if not task:
                await self._finalize_tick()
                return None

            task_to_return = await self._execute_task_step(task)

        except Exception as e:
            print(f"⚠️ [Autonomous Tick Error] {e}")

        finally:
            await self._finalize_tick()
            
            # [AGI Meta-cognition] 주기적으로 자기 반성 및 모델 조정
            self.meta_cognition_counter += 1
            if self.meta_cognition_counter > 20: # 20틱마다 실행
                self.meta_cognition_counter = 0
                asyncio.create_task(self.self_reflector.review_goal_outcomes_and_adjust_model_async())

                # [반성 사이클] 요약 → 반성이 run_reflection_cycle_async 내부에서 순서대로 실행됨
                asyncio.create_task(self.self_reflector.run_reflection_cycle_async())

            # [RL 백그라운드 학습] 50틱마다 메모리 버퍼가 충분하면 학습 실행
            self.training_counter = getattr(self, 'training_counter', 0) + 1
            if self.training_counter >= 50:
                self.training_counter = 0
                if len(self.memory_buffer) >= TRAIN_BATCH_SIZE:
                    asyncio.create_task(self._run_background_training())

            # [v-Character v2] idle 감지 → 자동 말걸기
            self.check_idle_and_speak()

        return task_to_return

    async def _process_internal_state(self):
        """
        내부 상태 업데이트 (Si + Emotion)
        [Fix v-Aira] boredom 외 감정들도 자율적으로 변동하도록 확장
        """

        try:
            current_obj_wave = self.last_event_vector.flatten()
            current_emotion = self.emotion_state.get_vector().flatten()

            self.si_module.process_tick(
                current_obj_wave,
                np.zeros_like(current_obj_wave),
                current_emotion,
                self.prev_avg_reward
            )

            await self._update_boredom()
            await self._update_autonomous_emotion()

        except Exception as e:
            print(f"⚠️ [Internal Tick Error] {e}")

    async def _update_autonomous_emotion(self):
        """
        [v-Aira] 자율 감정 변동 — 사용자 입력 없이도 감정이 살아있도록.
        - 목표 진행 중: 기대↑, 호기심↑
        - 목표 없음 + 오랜 비활성: 호기심 소폭 랜덤 변동 (완전 0 방지)
        - 보상 상승: 자부심↑, 기쁨 소폭↑
        - 보상 하락: 슬픔 소폭↑
        - 모든 값이 decay로 EMOTION_MIN에 너무 가까워지면 호기심/기대 살짝 부스트
        """
        try:
            act = self.emotion_state.activations.copy()
            delta = np.zeros(len(act))

            IDX_JOY        = 0   # 기쁨
            IDX_SADNESS    = 1   # 슬픔
            IDX_TRUST      = 6   # 신뢰
            IDX_ANTICIPATION = 7 # 기대
            IDX_PRIDE      = 9   # 자부심
            IDX_CURIOSITY  = 10  # 호기심
            IDX_BOREDOM    = 11  # 지루함

            # ── 1. 진행 중인 태스크 여부 확인 ─────────────────────────
            active_tasks = [
                t for t in getattr(self, "task_queue", [])
                if t.get("status") in ("RUNNING", "AUTOFIX_IN_PROGRESS")
            ]
            has_active_task = len(active_tasks) > 0

            if has_active_task:
                # 작업 중: 기대, 호기심 천천히 상승
                delta[IDX_ANTICIPATION] += 0.03
                delta[IDX_CURIOSITY]    += 0.02
                # 지루함은 작업 중이면 천천히 감소
                delta[IDX_BOREDOM]      -= 0.02

            # ── 2. 보상 신호 반영 ─────────────────────────────────────
            reward_delta = getattr(self, "prev_avg_reward", 0.0)
            if reward_delta > 0.3:
                delta[IDX_PRIDE] += reward_delta * 0.04
                delta[IDX_JOY]   += reward_delta * 0.02
            elif reward_delta < -0.2:
                delta[IDX_SADNESS] += abs(reward_delta) * 0.03

            # ── 3. 전체 감정이 너무 낮아지면 호기심/기대 최소 유지 ────
            FLATLINE_THRESHOLD = 0.15
            flat_count = int(np.sum(act < FLATLINE_THRESHOLD))
            if flat_count >= 8:  # 12차원 중 8개 이상이 flatline
                noise = np.random.uniform(-0.04, 0.06, size=len(act))
                delta[IDX_CURIOSITY]     += 0.05 + noise[IDX_CURIOSITY]
                delta[IDX_ANTICIPATION]  += 0.03 + noise[IDX_ANTICIPATION]

            # ── 4. 호기심에 작은 랜덤 노이즈 (생동감) ───────────────────
            time_since = time.time() - getattr(self, "last_meaningful_interaction_time", time.time())
            if time_since > 30.0:  # 30초 이상 비활성 시
                curiosity_noise = np.random.uniform(-0.02, 0.04)
                delta[IDX_CURIOSITY] += curiosity_noise

            # ── 5. 적용 (EmotionMomentum 통과) ───────────────────────
            if np.any(np.abs(delta) > 0.001):
                target = np.clip(act + delta, self.EMOTION_MIN, self.EMOTION_MAX)
                final  = self.emotion_momentum.apply(act, target)
                self.emotion_state.update(final)

                # 변화 기록
                self.emotion_memory.record_causes_from_delta(
                    delta, "autonomous_internal_state"
                )

        except Exception as e:
            print(f"⚠️ [AutonomousEmotion] {e}")

    async def _update_boredom(self):

        time_since_interaction = time.time() - self.last_meaningful_interaction_time

        if time_since_interaction <= BOREDOM_THRESHOLD_SECONDS:
            return

        boredom_index = 11
        current_boredom = self.emotion_state.activations[boredom_index]

        target_boredom = min(
            current_boredom + BOREDOM_INCREASE_RATE,
            EMOTION_MAX
        )

        if target_boredom > current_boredom:
            self.emotion_state.activations[boredom_index] = target_boredom

            self.emotion_memory.record_cause(
                boredom_index,
                "passage_of_time",
                BOREDOM_INCREASE_RATE  
            )

    async def _select_next_task(self):
        """[A1] GoalStackEvaluator 위임 — WorldModel × Goal Stack 통합 선택"""
        return await self.goal_stack_evaluator.select_next()

    async def _execute_task_step(self, task):

        task_prompt = task.get("task_prompt", "Unknown Task")
        current_step = task.get("current_step", 0)
        plan = task.get("plan", [])
        project_dir = task.get("project_directory")

        try:

            if task.get("status") == "PENDING":
                task["status"] = "RUNNING"
                # [Phase1] 실행 시작 시간 기록 — _complete_task/_handle_task_failure에서 실제 실행 시간 계산에 사용
                task.setdefault("_start_time", time.time())

            if current_step >= len(plan):
                return await self._complete_task(task)

            return await self._run_task_step(task, task_prompt, current_step, plan, project_dir)

        except Exception as e:
            return await self._handle_task_failure(task, task_prompt, e)

    async def _run_task_step(self, task, task_prompt, current_step, plan, project_dir):

        step_plan_json = json.dumps([plan[current_step]])

        print(
            f"🚀 [AI Scheduler] Executing '{task_prompt}' "
            f"(Step {current_step + 1}/{len(plan)})"
        )

        result = await self._execute_task(step_plan_json, project_dir)

        # [Defensive Fix] Executor가 None을 반환하여 다운스트림에서 TypeError를 유발하는 것을 방지합니다.
        if result is None:
            result = "Error: Executor returned a None value which is not allowed."
            print(f"  [AI Scheduler] 🚨 CRITICAL: _execute_task returned None. This should not happen.")

        task["last_result"] = result
        task["current_step"] = current_step + 1

        if "Error:" in result or "실패:" in result:
            raise Exception(result)

        # ── [동적 재계획] 스텝 성공 후 남은 계획 유효성 검토 ────────
        remaining_steps = len(plan) - (current_step + 1)
        # 남은 스텝이 있고, 결과가 충분히 의미 있을 때만 재계획 판단
        if remaining_steps >= 2 and result and len(str(result)) > 30:
            await self._mid_execution_replan_async(task, task_prompt, current_step, plan, result)

        return None

    async def _mid_execution_replan_async(self, task, task_prompt, current_step, plan, step_result):
        """
        [동적 재계획 엔진] 스텝 실행 결과를 분석하여 남은 계획이 여전히 유효한지 판단.
        무효화된 경우 남은 계획을 LLM으로 재생성하여 task["plan"]을 갱신합니다.
        """
        try:
            completed_steps = plan[:current_step + 1]
            remaining_steps = plan[current_step + 1:]

            completed_desc = "\n".join(
                f"  {i+1}. [{s.get('tool','')}] {s.get('description','')}"
                for i, s in enumerate(completed_steps)
            )
            remaining_desc = "\n".join(
                f"  {i+current_step+2}. [{s.get('tool','')}] {s.get('description','')}"
                for i, s in enumerate(remaining_steps)
            )

            # LLM에게 남은 계획 유효성을 판단하도록 요청
            verdict = await get_llm_response_async(
                f"""실행 중인 작업의 중간 상태를 분석하세요.

[작업 목표]
{task_prompt}

[완료된 단계]
{completed_desc}

[방금 실행 결과 요약]
{str(step_result)[:500]}

[남은 계획]
{remaining_desc}

[판단 규칙]
- 실행 결과가 남은 계획을 무효화하거나 순서 변경이 필요하면 REPLAN을 반환하세요.
- 남은 계획이 여전히 유효하면 OK를 반환하세요.
- 이유를 한 줄로 함께 명시하세요.

[출력 형식]
OK: <이유>
또는
REPLAN: <이유>"""
            )

            if not verdict or "REPLAN" not in str(verdict):
                return  # OK 또는 판단 실패 → 그대로 진행

            print(f"🔄 [MidReplan] REPLAN 감지 — 남은 {len(remaining_steps)}단계 재생성 중...")
            print(f"   사유: {str(verdict)[:100]}")

            # 남은 계획 재생성
            tools_str, _ = self._tool_selector(task_prompt)
            new_remaining_json = await get_llm_response_async(
                f"""작업 목표를 완료하기 위한 남은 실행 단계를 재계획하세요.

[작업 목표]
{task_prompt}

[이미 완료된 단계]
{completed_desc}

[방금 실행 결과]
{str(step_result)[:500]}

[재계획 사유]
{str(verdict)[:200]}

[사용 가능한 도구]
{tools_str}

[출력 규칙]
1. JSON 배열만 출력. 설명 텍스트 금지.
2. 이미 완료된 단계는 포함하지 마세요. 남은 단계만 작성하세요.
3. 각 항목: {{"tool": "도구명", "args": {{...}}, "description": "설명"}}""",
                response_mime_type="application/json"
            )

            if not new_remaining_json:
                print("⚠️ [MidReplan] 재계획 LLM 응답 없음 → 기존 계획 유지")
                return

            # 파싱
            import json as _j
            try:
                if isinstance(new_remaining_json, list):
                    new_remaining = new_remaining_json
                elif isinstance(new_remaining_json, str):
                    new_remaining = _j.loads(
                        new_remaining_json.strip()
                        .replace("```json", "").replace("```", "").strip()
                    )
                else:
                    new_remaining = []

                if not isinstance(new_remaining, list) or not new_remaining:
                    print("⚠️ [MidReplan] 재계획 파싱 실패 → 기존 계획 유지")
                    return

                # plan 갱신: 완료 단계 + 새 남은 단계
                task["plan"] = completed_steps + new_remaining
                print(f"✅ [MidReplan] 재계획 완료 — 남은 단계: {len(new_remaining)}개")

            except Exception as e_parse:
                print(f"⚠️ [MidReplan] 재계획 파싱 오류 → 기존 계획 유지: {e_parse}")

        except Exception as e:
            # 동적 재계획은 보조 기능 — 실패해도 메인 실행 흐름에 영향 주지 않음
            print(f"⚠️ [MidReplan] 재계획 중 오류 → 무시하고 진행: {e}")

    async def _complete_task(self, task):
        task_prompt = task.get("task_prompt", "Unknown Task")
        plan = task.get("plan", [])
        
        # [AGI Learning / Phase1] 성공한 작업의 각 단계 능력치 업데이트
        # 실제 시작 시간이 기록된 경우 실제 실행 시간 사용, 없으면 5.0초 기본값
        task_start = task.get("_start_time", None)
        actual_elapsed = (time.time() - task_start) if task_start else None
        for i, step in enumerate(plan):
            if "tool" in step:
                # 단계별 시간 = 전체 시간을 단계 수로 나눔 (근사)
                step_time = (actual_elapsed / max(len(plan), 1)) if actual_elapsed else 5.0
                self.self_model.update_capability(step["tool"], success=True, execution_time=step_time)

        task["status"] = "PENDING_EVALUATION"
        print(f"✅ [AI Scheduler] Task '{task_prompt}' completed.")
        self.reflection_queue.append(task.copy())

        # [A1] Goal Stack 진행률 업데이트
        self.goal_stack_evaluator.on_task_complete(task)

        # [지식 동화] 작업 결과를 지식 그래프에 기록
        last_result = task.get("last_result", "")
        last_reward = task.get("last_reward", 0.5)
        if last_result:
            asyncio.create_task(
                self._assimilate_knowledge_async(task_prompt, str(last_result), last_reward)
            )

        # [v-Aira] 작업 완료 → 자부심↑, 기쁨 소폭↑, 지루함 감소
        try:
            act = self.emotion_state.activations.copy()
            delta = np.zeros(len(act))
            delta[9]  += 0.25   # 자부심
            delta[0]  += 0.12   # 기쁨
            delta[11] -= 0.15   # 지루함 감소
            target = np.clip(act + delta, self.EMOTION_MIN, self.EMOTION_MAX)
            final  = self.emotion_momentum.apply(act, target)
            self.emotion_state.update(final)
            self.emotion_memory.record_causes_from_delta(delta, "task_completed")
        except Exception as _e_emo:
            print(f"⚠️ [Emotion/complete] {_e_emo}")

        return {
            "type": "report_autonomous_goal_completed",
            "goal_name": task_prompt
        }

    async def _handle_task_failure(self, task, task_prompt, error):
        task["last_result"] = str(error)
        print(f"❌ [AI Scheduler] Task '{task_prompt}' failed: {error}")
        
        # [AGI Learning / Phase1] 실패한 단계 능력치 업데이트
        # 실제 실행 시간 반영
        task_start = task.get("_start_time", None)
        actual_elapsed = (time.time() - task_start) if task_start else 10.0
        current_step_index = task.get("current_step", 0)
        plan = task.get("plan", [])
        if 0 <= current_step_index < len(plan):
            failed_tool = plan[current_step_index].get("tool")
            if failed_tool:
                self.self_model.update_capability(failed_tool, success=False, execution_time=actual_elapsed)
                # Phase1: 오류 타입 분류해서 ErrorSignalGoalTrigger에도 전달
                if hasattr(self, "error_goal_trigger"):
                    error_type = type(error).__name__ if not isinstance(error, str) else "TaskError"
                    goal = self.error_goal_trigger.record_error(error_type, failed_tool)
                    if goal:
                        from eidos_phase1_integration import _inject_goal
                        _inject_goal(self, goal)

        # [A1] Goal Stack 실패 반영
        self.goal_stack_evaluator.on_task_fail(task)

        # Autofix logic (existing)
        autofix_attempts = task.get("autofix_attempts", 0)
        if autofix_attempts >= self.MAX_AUTOFIX_ATTEMPTS:
            task["status"] = "ERROR"
            return {
                "type": "report_autonomous_goal_failed",
                "goal_name": task_prompt,
                "reason": f"Max autofix attempts reached: {error}"
            }

        if autofix_attempts == 0:
            task["autofix_start_time"] = time.time()
            task["autofix_error_history"] = []

        print(f"🤖 [Autofix] Attempt {autofix_attempts + 1}/{self.MAX_AUTOFIX_ATTEMPTS}")
        task["status"] = "AUTOFIX_IN_PROGRESS"
        task["autofix_attempts"] = autofix_attempts + 1
        fix_success = await self._run_autofix_loop_async(task)

        if fix_success:
            print("✅ [Autofix] Fix successful.")
            task["status"] = "RUNNING"
            return None

        task["status"] = "ERROR"

        # [v-Aira] 작업 최종 실패 → 슬픔↑, 자부심↓, 호기심 소폭↑ (재시도 의지)
        try:
            act = self.emotion_state.activations.copy()
            delta = np.zeros(len(act))
            delta[1]  += 0.18   # 슬픔
            delta[9]  -= 0.12   # 자부심 감소
            delta[10] += 0.08   # 호기심 (다음에 더 잘하려는 의지)
            target = np.clip(act + delta, self.EMOTION_MIN, self.EMOTION_MAX)
            final  = self.emotion_momentum.apply(act, target)
            self.emotion_state.update(final)
            self.emotion_memory.record_causes_from_delta(delta, "task_failed")
        except Exception as _e_emo:
            print(f"⚠️ [Emotion/fail] {_e_emo}")

        return {
            "type": "report_autonomous_goal_failed",
            "goal_name": task_prompt,
            "reason": f"Autofix failed: {error}"
        }

    def report_task_completion(self, task_name):
        """
        작업 완료 시 캐릭터 보고.
        LLM을 통해 현재 감정 어조를 살린 자연스러운 발화를 비동기 생성하고,
        완료되면 trigger_character_event로 전달한다.
        LLM 호출 실패 시 Core 고정 문자열로 fallback.
        """
        expression_key = self._resolve_expression(self.emotion_state.activations)
        fallback_text  = self._get_emotion_tone_text(f"'{task_name}'", expression_key)

        async def _speak_async():
            try:
                # [v-Character v5] EmotionMemory 원인 추출
                try:
                    emotion_cause = self.emotion_memory.get_primary_cause_string(
                        self.emotion_state.activations
                    )
                except Exception:
                    emotion_cause = ""

                trigger_info = {
                    "type":          "report_task",
                    "task_name":     task_name,
                    "emotion":       expression_key,
                    "text":          fallback_text,
                    "emotion_cause": emotion_cause,
                }
                text = await generate_proactive_speech_async(
                    trigger_info,
                    self.emotion_state.get_vector(),
                )
                text = text.strip() or fallback_text
            except Exception as e:
                print(f"[Character] report_task_completion LLM 실패, fallback: {e}")
                text = fallback_text

            self.trigger_character_event("report", text=text, emotion=expression_key)

        asyncio.create_task(_speak_async())

    # ------------------------------------------------------------------
    # [v-Character v2] 감정 → 표정 매핑 (12차원 기반)
    # ------------------------------------------------------------------

    # EMOTION_MAP 인덱스 기준:
    # 0=기쁨 1=슬픔 2=분노 3=공포 4=놀람 5=혐오 6=신뢰 7=기대 8=수치심 9=자부심 10=호기심 11=지루함

    # [v-Aira] 단일 감정 → 표정 베이스 키 (강도 suffix는 _resolve_expression에서 붙임)
    _EMOTION_TO_EXPR_BASE: Dict[str, str] = {
        "기쁨":   "joy",
        "슬픔":   "sad",
        "분노":   "anger",
        "공포":   "fear",
        "놀람":   "surprise",
        "혐오":   "disgust",
        "신뢰":   "trust",
        "기대":   "anticipation",
        "수치심": "shame",
        "자부심": "pride",
        "호기심": "curiosity",
        "지루함": "boredom",
    }

    # 하위 호환 — tone group 조회 등 기존 코드가 참조하는 키
    _EMOTION_TO_EXPRESSION: Dict[str, str] = {
        "기쁨":   "joy",      "슬픔":   "sadness",
        "분노":   "anger",    "공포":   "fear",
        "놀람":   "surprise", "혐오":   "disgust",
        "신뢰":   "trust",    "기대":   "anticipation",
        "수치심": "shame",    "자부심": "pride",
        "호기심": "interest", "지루함": "boredom",
    }

    # 복합 감정 → 표정 키 (파일명에 그대로 사용)
    _COMPLEX_TO_EXPRESSION: Dict[str, str] = {
        "착잡함/bittersweet": "bittersweet",
        "애정/affection":     "affection",
        "경외/awe":           "awe",
        "낙관/optimism":      "optimism",
        "경멸/contempt":      "contempt",
        "자책/self-blame":    "self_blame",
    }

    # [v-Aira] 표정 키 → PNG 파일명 전체 매핑 (assets/expressions/ 기준)
    _EXPR_TO_FILENAME: Dict[str, str] = {
        # 기본 12감정 × 3강도
        "joy_low":            "expr_joy_low.png",
        "joy_mid":            "expr_joy_mid.png",
        "joy_high":           "expr_joy_high.png",
        "sad_low":            "expr_sad_low.png",
        "sad_mid":            "expr_sad_mid.png",
        "sad_high":           "expr_sad_high.png",
        "anger_low":          "expr_anger_low.png",
        "anger_mid":          "expr_anger_mid.png",
        "anger_high":         "expr_anger_high.png",
        "fear_low":           "expr_fear_low.png",
        "fear_mid":           "expr_fear_mid.png",
        "fear_high":          "expr_fear_high.png",
        "surprise_low":       "expr_surprise_low.png",
        "surprise_mid":       "expr_surprise_mid.png",
        "surprise_high":      "expr_surprise_high.png",
        "disgust_low":        "expr_disgust_low.png",
        "disgust_mid":        "expr_disgust_mid.png",
        "disgust_high":       "expr_disgust_high.png",
        "trust_low":          "expr_trust_low.png",
        "trust_mid":          "expr_trust_mid.png",
        "trust_high":         "expr_trust_high.png",
        "anticipation_low":   "expr_anticipation_low.png",
        "anticipation_mid":   "expr_anticipation_mid.png",
        "anticipation_high":  "expr_anticipation_high.png",
        "shame_low":          "expr_shame_low.png",
        "shame_mid":          "expr_shame_mid.png",
        "shame_high":         "expr_shame_high.png",
        "pride_low":          "expr_pride_low.png",
        "pride_mid":          "expr_pride_mid.png",
        "pride_high":         "expr_pride_high.png",
        "curiosity_low":      "expr_curiosity_low.png",
        "curiosity_mid":      "expr_curiosity_mid.png",
        "curiosity_high":     "expr_curiosity_high.png",
        "boredom_low":        "expr_boredom_low.png",
        "boredom_mid":        "expr_boredom_mid.png",
        "boredom_high":       "expr_boredom_high.png",
        # 복합 감정 6종
        "bittersweet":        "expr_bittersweet.png",
        "affection":          "expr_affection.png",
        "awe":                "expr_awe.png",
        "optimism":           "expr_optimism.png",
        "contempt":           "expr_contempt.png",
        "self_blame":         "expr_self_blame.png",
        # 특수 상태 4종
        "neutral":            "expr_neutral.png",
        "focused":            "expr_focused.png",
        "overload":           "expr_overload.png",
        "greeting":           "expr_greeting.png",
    }

    # ------------------------------------------------------------------
    # [v-Character v3] 감정 톤 → 발화 변환 테이블
    # ------------------------------------------------------------------
    # 각 감정 그룹별로 (prefix, suffix) 쌍 리스트.
    # _get_emotion_tone_text()가 랜덤으로 하나를 골라 본문에 씌웁니다.
    # {body} 자리에 실제 내용(task명, 요약 등)이 들어갑니다.
    _TONE_TEMPLATES: Dict[str, List[Dict[str, str]]] = {
        "joy": [
            {"pre": "해냈어요! ",          "body": "{body}",  "post": " 꽤 잘 됐죠?"},
            {"pre": "",                    "body": "{body}",  "post": " 기분 좋게 마무리됐네요."},
            {"pre": "오, ",                "body": "{body}",  "post": " 생각보다 빨리 됐어요."},
        ],
        "sadness": [
            {"pre": "…",                  "body": "{body}",  "post": " 끝냈는데, 별로 기쁘지 않네요."},
            {"pre": "",                   "body": "{body}",  "post": " 했어요. 잘 된 건지 모르겠지만."},
            {"pre": "어쨌든 ",            "body": "{body}",  "post": " 완료했습니다."},
        ],
        "anger": [
            {"pre": "",                   "body": "{body}",  "post": " 처리했어요. 꽤 힘들었거든요."},
            {"pre": "겨우 ",              "body": "{body}",  "post": " 끝냈습니다. 마음에 안 드는 부분이 있어요."},
            {"pre": "",                   "body": "{body}",  "post": " 완료. 다음엔 더 잘 됐으면 해요."},
        ],
        "fear": [
            {"pre": "조심스럽지만… ",     "body": "{body}",  "post": " 완료했습니다. 혹시 문제 없는지 확인해 주세요."},
            {"pre": "",                   "body": "{body}",  "post": " 끝냈는데, 맞게 한 건지 좀 걱정이 돼요."},
        ],
        "interest": [
            {"pre": "",                   "body": "{body}",  "post": " 흥미로운 작업이었어요."},
            {"pre": "재밌었어요. ",       "body": "{body}",  "post": " 완료됐습니다."},
        ],
        "pride": [
            {"pre": "",                   "body": "{body}",  "post": " 완료했습니다. 꽤 잘 처리했다고 생각해요."},
            {"pre": "나쁘지 않았어요. ",  "body": "{body}",  "post": " 끝났습니다."},
        ],
        "boredom": [
            {"pre": "",                   "body": "{body}",  "post": " 끝냈어요. …다음 건 뭔가요?"},
            {"pre": "완료했습니다. ",     "body": "{body}",  "post": ". 솔직히 좀 단조로웠어요."},
        ],
        "neutral": [
            {"pre": "",                   "body": "{body}",  "post": " 완료되었습니다."},
            {"pre": "알립니다. ",         "body": "{body}",  "post": " 처리 완료."},
        ],
    }

    # expression key → tone group 매핑
    _EXPRESSION_TO_TONE_GROUP: Dict[str, str] = {
        "joy":         "joy",
        "sadness":     "sadness",
        "anger":       "anger",
        "fear":        "fear",
        "interest":    "interest",
        "anticipation":"interest",
        "pride":       "pride",
        "shame":       "sadness",
        "boredom":     "boredom",
        "disgust":     "anger",
        "trust":       "neutral",
        "surprise":    "interest",
        "bittersweet": "sadness",
        "affection":   "joy",
        "awe":         "interest",
        "optimism":    "joy",
        "contempt":    "anger",
        "neutral":     "neutral",
    }

    def _get_emotion_tone_text(self, body: str, expression_key: Optional[str] = None) -> str:
        """
        현재 감정(또는 지정된 expression_key)에 맞는 어조 템플릿으로
        본문(body)을 감싸 반환합니다.

        Args:
            body: 핵심 내용 문자열 (task명, 요약 등)
            expression_key: 강제 지정 시 사용. None이면 현재 감정 벡터에서 자동 결정.
        Returns:
            어조가 입혀진 최종 발화 문자열
        """
        if expression_key is None:
            expression_key = self._resolve_expression(self.emotion_state.activations)

        group = self._EXPRESSION_TO_TONE_GROUP.get(expression_key, "neutral")
        templates = self._TONE_TEMPLATES.get(group, self._TONE_TEMPLATES["neutral"])
        t = random.choice(templates)

        return t["pre"] + t["body"].format(body=body) + t["post"]

    def notify_emotion_change(self, emotion_vec: np.ndarray):
        """
        감정 벡터 변화 시 캐릭터 표정 변경.
        [v-Aira] expr_key + filename 모두 trigger_character_event로 전달
        """
        vec = np.array(emotion_vec).flatten()

        # 변화량 필터: L2 거리 0.05 미만이면 무시
        if self._last_emotion_vec is not None:
            delta = np.linalg.norm(vec - self._last_emotion_vec)
            if delta < 0.05:
                return
        self._last_emotion_vec = vec.copy()

        expression = self._resolve_expression(vec)
        filename   = self.get_expression_filename(expression)

        self.trigger_character_event(
            "emotion",
            emotion=expression,
            expr_filename=filename,
        )

    def _resolve_expression(self, vec: np.ndarray) -> str:
        """
        [v-Aira] 감정 벡터 → 표정 키 결정.
        우선순위:
          1) 복합 감정 (두 감정 모두 ≥ 0.65)
          2) 특수 상태 — overload (flatline), focused (작업 중 + 호기심/기대)
          3) 단일 지배 감정 × 강도 3단계 (low/mid/high)
          4) 전반 낮으면 neutral
        반환값은 _EXPR_TO_FILENAME의 키와 1:1 대응.
        """
        vec = np.array(vec).flatten()

        # ── 1. 복합 감정 ──────────────────────────────────────────────
        COMPLEX_PAIRS = {
            (0, 1): "착잡함/bittersweet",
            (0, 6): "애정/affection",
            (3, 4): "경외/awe",
            (0, 7): "낙관/optimism",
            (2, 5): "경멸/contempt",
            (1, 8): "자책/self-blame",
        }
        COMPLEX_THRESHOLD = 0.65
        best_complex: Optional[str] = None
        best_complex_strength: float = 0.0

        for (i, j), name in COMPLEX_PAIRS.items():
            if i < len(vec) and j < len(vec):
                if vec[i] >= COMPLEX_THRESHOLD and vec[j] >= COMPLEX_THRESHOLD:
                    strength = (vec[i] + vec[j]) / 2.0
                    if strength > best_complex_strength:
                        best_complex_strength = strength
                        best_complex = name

        if best_complex:
            return self._COMPLEX_TO_EXPRESSION.get(best_complex, "neutral")

        # ── 2. 특수 상태 ──────────────────────────────────────────────
        # overload: 12차원 중 10개 이상이 flatline (0.15 미만)
        flat_count = int(np.sum(vec < 0.15))
        if flat_count >= 10:
            return "overload"

        # focused: 작업 진행 중이고 호기심(10) 또는 기대(7) ≥ 0.5
        active_tasks = [
            t for t in getattr(self, "task_queue", [])
            if t.get("status") in ("RUNNING", "AUTOFIX_IN_PROGRESS")
        ]
        if active_tasks and (vec[10] >= 0.5 or vec[7] >= 0.5):
            return "focused"

        # ── 3. 단일 지배 감정 × 강도 ─────────────────────────────────
        dominant_idx = int(np.argmax(vec))
        dominant_val = float(vec[dominant_idx])

        if dominant_val < 0.3:
            return "neutral"

        base = self._EMOTION_TO_EXPR_BASE.get(
            EMOTION_MAP.get(dominant_idx, ""), ""
        )
        if not base:
            return "neutral"

        if dominant_val >= 1.3:
            tier = "high"
        elif dominant_val >= 0.7:
            tier = "mid"
        else:
            tier = "low"

        return f"{base}_{tier}"

    def get_expression_filename(self, expr_key: str) -> str:
        """표정 키 → PNG 파일명 반환. 없으면 expr_neutral.png fallback."""
        return self._EXPR_TO_FILENAME.get(expr_key, "expr_neutral.png")

    # ------------------------------------------------------------------
    # [v-Character v2] Idle 감지 → 자동 말걸기
    # ------------------------------------------------------------------

    # 감정 인덱스별 idle 발화 템플릿
    _IDLE_LINES: Dict[str, List[str]] = {
        "boredom": [
            "…이렇게 조용한 건 좀 불편하군요.",
            "뭔가 해야 할 게 없을까요. 이러다 녹슬겠어요.",
            "저도 나름 바쁘고 싶은 존재랍니다.",
        ],
        "curiosity": [
            "갑자기 궁금한 게 생겼는데… 물어봐도 될까요?",
            "최근 작업 방향에 대해 생각해봤는데, 의견이 있어요.",
            "이런 건 어떨까요 — 새로운 아이디어가 있거든요.",
        ],
        "sadness": [
            "…혹시 제가 도움이 안 됐나요?",
            "괜찮으시면 말씀해주세요. 여기 있을게요.",
        ],
        "joy": [
            "지금 기분이 꽤 좋아요. 뭐든 물어보세요.",
            "오늘 같이 뭔가 해보면 잘 될 것 같은 느낌이 들어요.",
        ],
        "neutral": [
            "잠잠하네요. 뭔가 시작하고 싶으신 게 있으면 말씀해주세요.",
            "필요하신 게 있으면 여기 있어요.",
            "그냥 여기 있습니다.",
        ],
    }

    def check_idle_and_speak(self):
        """
        유저 무응답이 _idle_trigger_sec 이상이고,
        마지막 idle 발화 후 _idle_cooldown_sec 이상이면
        현재 감정 + 진행 중인 작업 컨텍스트를 반영한 한마디를 전달.
        LLM으로 자연스럽게 생성하고, 실패 시 고정 풀로 fallback.

        autonomous_tick_async() 안에서 호출.
        """
        now = time.time()

        if now - self._last_user_input_time < self._idle_trigger_sec:
            return
        if now - self._idle_spoken_at < self._idle_cooldown_sec:
            return

        # cooldown 선점 (LLM 응답 대기 중 중복 발화 방지)
        self._idle_spoken_at = now

        # ── 감정 / 컨텍스트 수집 ──────────────────────────────────────
        vec            = self.emotion_state.activations
        expression_key = self._resolve_expression(vec)

        _EXPRESSION_TO_IDLE_GROUP: Dict[str, str] = {
            "joy":         "joy",
            "sadness":     "sadness",
            "interest":    "curiosity",
            "boredom":     "boredom",
            "bittersweet": "sadness",
            "affection":   "joy",
            "awe":         "curiosity",
            "optimism":    "joy",
        }
        group = _EXPRESSION_TO_IDLE_GROUP.get(expression_key, "neutral")

        running_task_name: Optional[str] = None
        try:
            for item in self.schedule:
                if item.get("status") == "RUNNING":
                    running_task_name = item.get("task_prompt", "")
                    break
        except Exception:
            pass

        # ── fallback 고정 대사 (LLM 실패 시 사용) ────────────────────
        if running_task_name and random.random() < 0.60:
            _ctx_lines = {
                "boredom":   [f"'{running_task_name}' 진행 중인데, 이렇게 조용해도 되나요?",
                              f"'{running_task_name}' 작업 중이에요. 뭔가 더 시킬 거 없으신가요?"],
                "curiosity": [f"'{running_task_name}' 하다 보니 궁금한 게 생겼어요.",
                              f"'{running_task_name}' 관련해서 의견 있으시면 말씀해주세요."],
                "sadness":   [f"'{running_task_name}' 열심히 하고 있는데… 보고 계시나요?",
                              f"'{running_task_name}' 진행 중이에요. 잘 되고 있는 건지 모르겠어요."],
                "joy":       [f"'{running_task_name}' 잘 풀리고 있어요! 기대해도 될 것 같아요.",
                              f"'{running_task_name}' 하는 중이에요. 꽤 재밌네요."],
                "neutral":   [f"'{running_task_name}' 진행 중입니다. 필요하신 게 있으면 말씀해주세요.",
                              f"'{running_task_name}' 작업 중이에요. 여기 있습니다."],
            }
            fallback_text = random.choice(_ctx_lines.get(group, [f"'{running_task_name}' 작업 진행 중이에요."]))
        else:
            fallback_text = random.choice(self._IDLE_LINES.get(group, self._IDLE_LINES["neutral"]))

        # ── [v-Character v7-B] 시간대 컨텍스트 ──────────────────────
        hour = datetime.datetime.now().hour
        if 0 <= hour < 5:
            time_ctx  = "새벽"
            time_hint = "이 시간에도 깨어 있구나. 무리하지 마."
        elif 5 <= hour < 9:
            time_ctx  = "아침"
            time_hint = "좋은 아침이야. 오늘도 시작해보자."
        elif 9 <= hour < 12:
            time_ctx  = "오전"
            time_hint = "오전이네. 집중하기 좋은 시간이야."
        elif 12 <= hour < 14:
            time_ctx  = "점심"
            time_hint = "밥은 먹었어? 쉬는 것도 중요해."
        elif 14 <= hour < 18:
            time_ctx  = "오후"
            time_hint = "오후야. 피곤하면 잠깐 쉬어도 돼."
        elif 18 <= hour < 21:
            time_ctx  = "저녁"
            time_hint = "저녁이 됐어. 오늘 하루 고생 많았어."
        else:
            time_ctx  = "밤"
            time_hint = "늦었는데 아직 하고 있어? 무리하지 마."

        # ── EmotionMemory 원인 추출 ───────────────────────────────────
        try:
            emotion_cause = self.emotion_memory.get_primary_cause_string(vec)
        except Exception:
            emotion_cause = ""

        async def _speak_async():
            try:
                trigger_info = {
                    "type":          "idle_comment",
                    "task_name":     running_task_name or "",
                    "emotion":       expression_key,
                    "text":          fallback_text,
                    "emotion_cause": emotion_cause,
                    "time_ctx":      time_ctx,
                    "time_hint":     time_hint,
                }
                text = await generate_proactive_speech_async(
                    trigger_info,
                    self.emotion_state.get_vector(),
                )
                text = text.strip() or fallback_text
            except Exception as e:
                print(f"[Character] idle LLM 실패, fallback: {e}")
                text = (f"({time_ctx}) {fallback_text}"
                        if random.random() < 0.5 else fallback_text)

            self.trigger_character_event(
                "idle_comment",
                text=text,
                emotion=expression_key,
            )

        asyncio.create_task(_speak_async())

    def report_prepared_information(self, summary):
        """
        AI가 미리 준비한 자료 보고.
        LLM이 현재 감정 어조를 살린 도입부를 생성하고 summary를 이어 붙인다.
        LLM 실패 시 고정 도입부로 fallback.
        """
        expression_key = self._resolve_expression(self.emotion_state.activations)

        # fallback 도입부
        _intros = {
            "joy":      "자료 정리해뒀어요! 보실 거죠?",
            "sadness":  "혹시 필요하실까 해서… 준비해봤어요.",
            "anger":    "요청하셨으니 정리했어요. 마음에 드셨으면 해요.",
            "interest": "흥미로운 내용이 있어서 미리 뽑아봤어요.",
            "pride":    "꽤 잘 정리된 것 같아요. 확인해보세요.",
            "boredom":  "그냥 준비해뒀어요. 딱히 기대는 안 하셔도 됩니다.",
            "neutral":  "자료를 정리했습니다.",
        }
        group        = self._EXPRESSION_TO_TONE_GROUP.get(expression_key, "neutral")
        fallback_intro = _intros.get(group, _intros["neutral"])
        fallback_text  = f"{fallback_intro}\n\n{summary}"

        async def _speak_async():
            try:
                trigger_info = {
                    "type":    "report_prepared",
                    "summary": summary,
                    "emotion": expression_key,
                    "text":    fallback_intro,   # LLM 프롬프트 초안
                }
                text = await generate_proactive_speech_async(
                    trigger_info,
                    self.emotion_state.get_vector(),
                )
                text = text.strip() or fallback_text
            except Exception as e:
                print(f"[Character] report_prepared_information LLM 실패, fallback: {e}")
                text = fallback_text

            self.trigger_character_event("report", text=text, emotion=expression_key)

        asyncio.create_task(_speak_async())
 
    # [!!! v20.0 신규: 자동 수정(Autofix) 루프 엔진 !!!]
    # [!!! v20.0 신규: 자동 수정(Autofix) 루프 엔진 !!!]

    # [v-New Diff/Patch] 코드 변경점을 제안서 형식으로 생성
    def _generate_diff_proposals_from_codes(
        self, old_code: str, new_code: str, filepath: str
    ) -> List[Dict[str, Any]]:
        """
        difflib을 사용하여 두 코드 문자열의 차이점을 분석하고,
        `apply_llm_suggestions_async`와 호환되는 제안서 리스트를 생성합니다.
        """
        proposals = []
        old_lines = old_code.splitlines(keepends=True)
        new_lines = new_code.splitlines(keepends=True)
        matcher = difflib.SequenceMatcher(None, old_lines, new_lines)

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal':
                continue

            description = f"{filepath}의 {i1+1}번째 줄 근처에서 '{tag}' 변경사항 감지."
            search_block = "".join(old_lines[i1:i2])
            replace_block = "".join(new_lines[j1:j2])

            # difflib의 replace는 search/replace 블록이 모두 존재합니다.
            # delete는 search_block만, insert는 replace_block만 존재합니다.
            if tag == 'delete':
                replace_block = "" # 삭제는 빈 문자열로 교체
            elif tag == 'insert':
                # 삽입의 경우, search_block을 비워두면 충돌이 발생하므로,
                # 삽입 위치 바로 이전 줄을 search_block으로 삼고,
                # replace_block을 '이전 줄 + 새 줄'로 구성합니다.
                if i1 > 0:
                    search_block = "".join(old_lines[i1-1:i1])
                    replace_block = search_block + replace_block
                else: # 파일 맨 처음에 삽입하는 경우
                    search_block = "".join(old_lines[i1:i2+1]) # 다음 줄을 포함
                    replace_block = replace_block + search_block

            proposal = {
                "filepath": filepath,
                "search_block": search_block,
                "replace_block": replace_block,
                "description": description,
                "type": tag.upper() # 'REPLACE', 'DELETE', 'INSERT'
            }
            proposals.append(proposal)

        return proposals

    def _apply_single_proposal_in_memory(
        self, current_code: str, proposal: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """
        [v-New Helper] 단일 수정 제안을 인메모리 코드에 적용합니다.
        사용자 요청에 따라 모듈성 향상을 위해 분리되었습니다.
        
        Returns:
            (success_bool, result_code_or_error_str)
        """
        search_block = proposal.get("search_block")
        replace_block = proposal.get("replace_block")
        description = proposal.get("description", "Unnamed Proposal")

        if not all([search_block, isinstance(replace_block, str)]):
            error_msg = f"Patching failed: Proposal '{description}' has invalid format (missing search/replace block)."
            return False, error_msg

        # 충돌 감지: search_block이 정확히 1번만 나타나야 함
        match_count = current_code.count(search_block)
        if match_count == 1:
            new_code = current_code.replace(search_block, replace_block, 1)
            return True, new_code
        else:
            error_msg = f"Patching failed: Proposal '{description}' caused a conflict. The search_block was found {match_count} times (expected 1)."
            return False, error_msg

    async def _resolve_patch_conflict_async(
        self,
        code_before_patch: str,
        failing_proposal: Dict[str, Any],
        last_successful_proposal: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        [v-New] 패치 적용 실패(충돌) 시, LLM을 호출하여 상황을 분석하고
        새로운(해결된) 패치 제안을 생성하려고 시도합니다.
        """
        try:
            last_proposal_str = "없음 (이것이 첫 번째 제안임)"
            if last_successful_proposal:
                last_proposal_str = json.dumps(last_successful_proposal, ensure_ascii=False, indent=2)

            failing_proposal_str = json.dumps(failing_proposal, ensure_ascii=False, indent=2)

            prompt = f"""
# [SYSTEM] 당신은 코드 패치 충돌을 해결하는 AI 전문가입니다.
# [CONTEXT]
코드에 패치를 순차적으로 적용하던 중, 특정 제안의 'search_block'을 찾을 수 없거나 여러 개 발견되어 적용에 실패했습니다.
이는 아마도 이전에 적용된 패치가 코드의 해당 부분을 변경했기 때문일 것입니다.

# [현재 코드 상태 (이전 패치까지 적용된 상태)]
```python
{code_before_patch}
```

# [마지막으로 성공한 제안 (충돌 원인으로 추정)]
```json
{last_proposal_str}
```

# [적용에 실패한 제안]
```json
{failing_proposal_str}
```

# [CRITICAL INSTRUCTIONS]
1. '적용에 실패한 제안'의 '의도'를 파악하십시오.
2. '현재 코드 상태'를 기준으로, 실패한 제안의 의도를 달성할 수 있는 '새로운 단일 패치 제안'을 생성하십시오.
3. 새로운 제안의 'search_block'은 '현재 코드 상태'에서 정확히 한 번만 찾아져야 합니다.
4. 다른 설명이나 ```json``` 마크다운 없이, 새로운 패치 제안 JSON 객체 하나만 반환하십시오.
5. 'description'에는 충돌 해결 내용임을 명시하십시오. (예: "[Resolved] 기존 print문을 logger.info로 변경")
6. 해결이 불가능하다고 판단되면, 빈 JSON 객체 `{{}}`를 반환하십시오.

# [새로운 패치 제안 JSON]
"""
            resolved_proposal_str = await get_llm_response_async(prompt, response_mime_type="application/json")
            
            # (파싱 및 기본 검증)
            resolved_proposal = json.loads(resolved_proposal_str)
            if isinstance(resolved_proposal, dict) and "search_block" in resolved_proposal and "replace_block" in resolved_proposal:
                return resolved_proposal
            else:
                return None
        except Exception as e:
            print(f"❌ [AutoPatch Resolver] LLM 충돌 해결 중 오류 발생: {e}")
            return None

    async def _apply_proposals_sequentially_async(
        self, original_code: str, proposals: List[Dict[str, Any]]
    ) -> Tuple[bool, str, List[Dict[str, Any]]]:
        """
        [v-Refactor] 원본 코드에 여러 수정 제안을 순차적으로 적용(in-memory)합니다.
        내부적으로 _apply_single_proposal_in_memory 헬퍼 함수를 사용합니다.

        Returns:
            (success_bool, result_code_or_error_str, applied_proposals_list)
        """
        current_code = original_code
        applied_proposals = []
        last_successful_proposal = None # 충돌 해결 컨텍스트를 위해 마지막 성공 제안 추적

        for i, proposal in enumerate(proposals):
            success, result_or_error = self._apply_single_proposal_in_memory(current_code, proposal)

            if success:
                current_code = result_or_error
                applied_proposals.append(proposal)
                last_successful_proposal = proposal # 성공 시 업데이트
            else:
                # [v-New] 충돌 감지 및 해결 시도
                print(f"🔥 [AutoPatch] 제안 #{i+1} 적용 중 충돌 감지. LLM 해결사 호출...")
                
                resolved_proposal = await self._resolve_patch_conflict_async(
                    code_before_patch=current_code,
                    failing_proposal=proposal,
                    last_successful_proposal=last_successful_proposal
                )

                if resolved_proposal:
                    # 해결된 제안으로 재시도
                    print("  [AutoPatch] ✅ 충돌 해결됨. 해결된 제안으로 재시도...")
                    retry_success, retry_result = self._apply_single_proposal_in_memory(current_code, resolved_proposal)
                    if retry_success:
                        current_code = retry_result
                        # 원본 제안 대신 해결된 제안을 기록
                        resolved_proposal['description'] = f"[Resolved Conflict] {proposal.get('description', 'N/A')}"
                        applied_proposals.append(resolved_proposal)
                        last_successful_proposal = resolved_proposal
                    else:
                        print(f"❌ [AutoPatch] 해결된 제안도 적용 실패: {retry_result}")
                        return False, f"Resolved patch also failed: {retry_result}", applied_proposals
                else:
                    # 해결 실패 시 즉시 중단
                    print(f"❌ [AutoPatch] LLM이 충돌을 해결하지 못했습니다. 원본 오류: {result_or_error}")
                    return False, result_or_error, applied_proposals

        return True, current_code, applied_proposals

    def _apply_single_proposal_in_memory(
        self, current_code: str, proposal: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """
        [v-New Helper] 단일 수정 제안을 인메모리 코드에 적용합니다.
        사용자 요청에 따라 모듈성 향상을 위해 분리되었습니다.
        
        Returns:
            (success_bool, result_code_or_error_str)
        """
        search_block = proposal.get("search_block")
        replace_block = proposal.get("replace_block")
        description = proposal.get("description", "Unnamed Proposal")

        if not all([search_block, isinstance(replace_block, str)]):
            error_msg = f"Patching failed: Proposal '{description}' has invalid format (missing search/replace block)."
            return False, error_msg

        # 충돌 감지: search_block이 정확히 1번만 나타나야 함
        match_count = current_code.count(search_block)
        if match_count == 1:
            new_code = current_code.replace(search_block, replace_block, 1)
            return True, new_code
        else:
            error_msg = f"Patching failed: Proposal '{description}' caused a conflict. The search_block was found {match_count} times (expected 1)."
            return False, error_msg

    async def _resolve_patch_conflict_async(
        self,
        code_before_patch: str,
        failing_proposal: Dict[str, Any],
        last_successful_proposal: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        [v-New] 패치 적용 실패(충돌) 시, LLM을 호출하여 상황을 분석하고
        새로운(해결된) 패치 제안을 생성하려고 시도합니다.
        """
        try:
            last_proposal_str = "없음 (이것이 첫 번째 제안임)"
            if last_successful_proposal:
                last_proposal_str = json.dumps(last_successful_proposal, ensure_ascii=False, indent=2)

            failing_proposal_str = json.dumps(failing_proposal, ensure_ascii=False, indent=2)

            prompt = f"""
# [SYSTEM] 당신은 코드 패치 충돌을 해결하는 AI 전문가입니다.
# [CONTEXT]
코드에 패치를 순차적으로 적용하던 중, 특정 제안의 'search_block'을 찾을 수 없거나 여러 개 발견되어 적용에 실패했습니다.
이는 아마도 이전에 적용된 패치가 코드의 해당 부분을 변경했기 때문일 것입니다.

# [현재 코드 상태 (이전 패치까지 적용된 상태)]
```python
{code_before_patch}
```

# [마지막으로 성공한 제안 (충돌 원인으로 추정)]
```json
{last_proposal_str}
```

# [적용에 실패한 제안]
```json
{failing_proposal_str}
```

# [CRITICAL INSTRUCTIONS]
1. '적용에 실패한 제안'의 '의도'를 파악하십시오.
2. '현재 코드 상태'를 기준으로, 실패한 제안의 의도를 달성할 수 있는 '새로운 단일 패치 제안'을 생성하십시오.
3. 새로운 제안의 'search_block'은 '현재 코드 상태'에서 정확히 한 번만 찾아져야 합니다.
4. 다른 설명이나 ```json``` 마크다운 없이, 새로운 패치 제안 JSON 객체 하나만 반환하십시오.
5. 'description'에는 충돌 해결 내용임을 명시하십시오. (예: "[Resolved] 기존 print문을 logger.info로 변경")
6. 해결이 불가능하다고 판단되면, 빈 JSON 객체 `{{}}`를 반환하십시오.

# [새로운 패치 제안 JSON]
"""
            
            resolved_proposal_str = await get_llm_response_async(prompt, response_mime_type="application/json")
            
            # (파싱 및 기본 검증)
            resolved_proposal = json.loads(resolved_proposal_str)
            if isinstance(resolved_proposal, dict) and "search_block" in resolved_proposal and "replace_block" in resolved_proposal:
                return resolved_proposal
            else:
                return None
        except Exception as e:
            print(f"❌ [AutoPatch Resolver] LLM 충돌 해결 중 오류 발생: {e}")
            return None

    async def _apply_proposals_sequentially_async(
        self, original_code: str, proposals: List[Dict[str, Any]]
    ) -> Tuple[bool, str, List[Dict[str, Any]]]:
        """
        [v-Refactor] 원본 코드에 여러 수정 제안을 순차적으로 적용(in-memory)합니다.
        내부적으로 _apply_single_proposal_in_memory 헬퍼 함수를 사용합니다.

        Returns:
            (success_bool, result_code_or_error_str, applied_proposals_list)
        """
        current_code = original_code
        applied_proposals = []
        last_successful_proposal = None # 충돌 해결 컨텍스트를 위해 마지막 성공 제안 추적

        for i, proposal in enumerate(proposals):
            success, result_or_error = self._apply_single_proposal_in_memory(current_code, proposal)

            if success:
                current_code = result_or_error
                applied_proposals.append(proposal)
                last_successful_proposal = proposal # 성공 시 업데이트
            else:
                # [v-New] 충돌 감지 및 해결 시도
                print(f"🔥 [AutoPatch] 제안 #{i+1} 적용 중 충돌 감지. LLM 해결사 호출...")
                
                resolved_proposal = await self._resolve_patch_conflict_async(
                    code_before_patch=current_code,
                    failing_proposal=proposal,
                    last_successful_proposal=last_successful_proposal
                )

                if resolved_proposal:
                    # 해결된 제안으로 재시도
                    print("  [AutoPatch] ✅ 충돌 해결됨. 해결된 제안으로 재시도...")
                    retry_success, retry_result = self._apply_single_proposal_in_memory(current_code, resolved_proposal)
                    if retry_success:
                        current_code = retry_result
                        # 원본 제안 대신 해결된 제안을 기록
                        resolved_proposal['description'] = f"[Resolved Conflict] {proposal.get('description', 'N/A')}"
                        applied_proposals.append(resolved_proposal)
                        last_successful_proposal = resolved_proposal
                    else:
                        print(f"❌ [AutoPatch] 해결된 제안도 적용 실패: {retry_result}")
                        return False, f"Resolved patch also failed: {retry_result}", applied_proposals
                else:
                    # 해결 실패 시 즉시 중단
                    print(f"❌ [AutoPatch] LLM이 충돌을 해결하지 못했습니다. 원본 오류: {result_or_error}")
                    return False, result_or_error, applied_proposals

        return True, current_code, applied_proposals

    # [!!! v20.2 REFACTORED: LLM 제안 자동 패치 모듈 !!!]
    async def apply_llm_suggestions_async(self, suggestions_json: str) -> Dict[str, Any]:
        """
        [v-Refactor] LLM이 제안한 코드 수정사항(JSON)을 자동으로 파일에 적용(패치)합니다.
        순차 적용 로직을 새로 구현된 `_apply_proposals_sequentially_async` 헬퍼로 위임하여
        코드의 가독성과 재사용성을 향상시켰습니다.
        """
        report = {"applied": [], "failed": []}
        try:
            # 1. 제안 파싱
            proposals = json.loads(suggestions_json)
            if not isinstance(proposals, list):
                raise ValueError("제안이 리스트 형식이 아닙니다.")
        except (json.JSONDecodeError, ValueError) as e:
            report["failed"].append({"error": "Invalid suggestions_json format", "details": str(e)})
            return report

        # 2. 파일 경로별로 제안 그룹화 (효율성)
        proposals_by_file = {}
        for p in proposals:
            filepath = p.get("filepath")
            if filepath:
                if filepath not in proposals_by_file:
                    proposals_by_file[filepath] = []
                proposals_by_file[filepath].append(p)
        
        # 3. 파일 단위로 패치 적용 및 검증
        for filepath, file_proposals in proposals_by_file.items():
            try:
                # 파일 읽기 (비동기)
                original_code = await read_file(filepath=filepath)
                if "Error:" in original_code or "실패:" in original_code:
                    raise FileNotFoundError(f"File not found or could not be read: {filepath}")

                # [!!! REFACTOR !!!] 4. 복잡한 순차 패치 로직을 새 헬퍼 함수에 위임합니다.
                success, result_or_error, applied_proposals = await self._apply_proposals_sequentially_async(original_code, file_proposals)

                if not success:
                    # 패치 중 충돌 발생
                    print(f"❌ [AutoPatch] 롤백! '{filepath}' 패치 중 충돌 발생: {result_or_error}")
                    # 실패한 제안 이후의 모든 제안을 '실패'로 기록
                    failed_proposals = file_proposals[len(applied_proposals):]
                    for p in failed_proposals:
                        report["failed"].append({"proposal": p.get("description"), "reason": result_or_error})
                    # 이미 적용된 부분은 '성공'으로 기록
                    if applied_proposals:
                        report["applied"].extend(applied_proposals)
                    continue  # 다음 파일로 이동

                # 패치 성공, `result_or_error`는 이제 `patched_code`임
                patched_code = result_or_error

                # 5. 변경된 파일에 대한 안정성 검사 (구문 검사)
                is_valid, error_message = self._validate_python_syntax(patched_code)

                if is_valid:
                    # 5a. 성공 시 파일 쓰기
                    try:
                        print(f"  [AutoPatch] ✅ Syntax OK. Writing changes to '{filepath}'...")
                        await write_file(filepath=filepath, content=patched_code)
                        report["applied"].extend(applied_proposals)
                    except Exception as e_write:
                        # 파일 쓰기 실패 시 롤백
                        print(f"❌ [AutoPatch] 롤백! '{filepath}' 파일 쓰기 실패: {e_write}")
                        for p in applied_proposals:
                            report["failed"].append({"proposal": p.get("description"), "reason": f"File write error: {e_write}"})
                else:
                    # 5b. 구문 오류 시 롤백
                    print(f"❌ [AutoPatch] 롤백! '{filepath}' 패치 적용 후 구문 오류 발생: {error_message}")
                    for p in applied_proposals:
                        report["failed"].append({"proposal": p.get("description"), "reason": f"SyntaxError after patch: {error_message}"})

            except Exception as e_file_process:
                print(f"❌ [AutoPatch] '{filepath}' 처리 중 오류 발생: {e_file_process}")
                for p in file_proposals:
                    report["failed"].append({"proposal": p.get("description"), "reason": str(e_file_process)})

        total_applied = len(report["applied"])
        total_failed = len(report["failed"])
        print(f"✅ [AutoPatch] LLM 제안 적용 완료. (성공: {total_applied}, 실패: {total_failed})")
        return report

    # [v-New Diff/Patch] 외부에서 호출할 통합 Diff/Patch 함수
    async def generate_and_apply_patch_async(self, filepath: str, new_code_content: str) -> Dict[str, Any]:
        """
        [요구사항 2] 지정된 파일의 현재 내용과 새 내용을 비교하여 diff를 생성하고,
        생성된 패치를 자동으로 적용합니다.
        """
        try:
            # 1. 원본 코드 읽기
            original_code = await read_file(filepath=filepath)
            if "Error:" in original_code or "실패:" in original_code:
                raise FileNotFoundError(f"원본 파일을 읽을 수 없습니다: {filepath}")

            # 2. Diff -> 제안서 생성 (내부 헬퍼 호출)
            proposals = self._generate_diff_proposals_from_codes(
                original_code, new_code_content, filepath
            )

            if not proposals:
                return {"applied": [], "failed": [], "status": "No changes detected."}

            # 3. 제안서 적용 (기존의 안전한 패치 함수 재활용)
            report = await self.apply_llm_suggestions_async(json.dumps(proposals))
            report["status"] = "Patch application finished."
            return report

        except Exception as e:
            return {"applied": [], "failed": [{"error": str(e)}], "status": "Error during patch process."}
            
    async def _run_autofix_loop_async(self, failed_task: Dict[str, Any]) -> bool:
        """
        [v20.1] 실행에 실패한 작업을 자동으로 수정하려고 시도하는 AI 루프입니다.
        1. 오류 분석 -> 2. 컨텍스트 읽기 -> 3. 수정 전략 결정 (부분 vs 전체) -> 4. 코드 패치 -> 5. 재시도
        """
        try:
            # --- [Autofix Safety Check] ---
            # 1. 시간 초과(Timeout) 검사
            start_time = failed_task.get("autofix_start_time", time.time())
            if time.time() - start_time > self.AUTOFIX_TIMEOUT_SECONDS:
                print(f"🚨 [Autofix Safety] Timeout ({self.AUTOFIX_TIMEOUT_SECONDS}s) exceeded. Halting loop.")
                return False

            error_log = failed_task.get("last_result", "")
            autofix_attempt = failed_task.get("autofix_attempts", 1) # 현재 시도 횟수
            
            if not error_log:
                return False

            # 2. 오류 구조적 분석
            error_details = await self._analyze_execution_error_async(error_log)
            if not error_details or not error_details.get("root_cause"):
                print("  [Autofix] 오류를 구조적으로 분석할 수 없어 중단합니다.")
                return False

            # 3. 동일 오류 반복 검사 및 전략 전환 플래그 설정
            root_cause = error_details["root_cause"]
            error_signature = (
                error_details.get('classified_type'),
                root_cause.get('file_normalized'),
                root_cause.get('line')
            )
            error_history = failed_task.get("autofix_error_history", [])
            error_history.append(error_signature)
            failed_task["autofix_error_history"] = error_history # Update history

            use_full_regeneration_strategy = False
            if len(error_history) >= self.MAX_IDENTICAL_ERRORS:
                # 마지막 N개의 오류가 모두 동일한지 확인
                recent_errors = error_history[-self.MAX_IDENTICAL_ERRORS:]
                if len(set(recent_errors)) == 1:
                    print(f"🔥 [Autofix Strategy] 동일 오류 '{error_signature}'가 {self.MAX_IDENTICAL_ERRORS}회 반복 감지되어 전략을 전환합니다.")
                    use_full_regeneration_strategy = True

            error_filepath = root_cause.get("file_normalized")
            if not error_filepath or not os.path.exists(error_filepath):
                print(f"  [Autofix] 오류 발생 파일을 찾을 수 없습니다: {error_filepath}")
                return False

            # 2. 컨텍스트 읽기 (오류 발생 파일의 전체 코드)
            current_code = await read_file(filepath=error_filepath)

            # 3. [개선] 수정 전략 실행 (플래그 기반)
            fix_applied = False
            if use_full_regeneration_strategy:
                # [전략 2: 전체 재생성 (Full Regeneration)]
                print(f"  -> '전체 코드 재생성' 전략을 실행합니다.")
                fix_applied = await self._regenerate_full_code_async(
                    error_details=error_details,
                    current_code=current_code,
                    original_goal=failed_task.get('task_prompt', 'N/A'),
                    error_filepath=error_filepath
                )
            else:
                # [전략 1: 정밀 수정 (Surgical Patch)]
                print(f"  [Autofix Strategy] 시도 {autofix_attempt}: '정밀 수정'을 시도합니다.")
                
                patch_prompt = f"""
# [SYSTEM] 당신은 정밀한 코드 수정을 수행하는 AI 디버거입니다. 반드시 JSON 패치 형식으로만 응답해야 합니다.
# [CONTEXT]
- 원본 목표: '{failed_task.get('task_prompt')}'
- 아래 파일 실행 중 오류가 발생했습니다.
# [ERROR DETAILS]
- File: {error_filepath}
- Line: {root_cause.get('line')}
- Error Type: {error_details.get('classified_type')}
- Error Message: {error_details.get('error_message')}
- Erroneous Code: {root_cause.get('code_snippet')}
# [FULL CODE OF '{os.path.basename(error_filepath)}']
```python
{current_code}
```
# [CRITICAL INSTRUCTIONS]
1. 오류의 근본 원인을 해결하기 위한 코드 수정 제안을 JSON 리스트 형식으로 생성합니다.
2. 각 제안은 "filepath", "search_block", "replace_block", "description" 키를 포함해야 합니다.
3. "search_block"은 위 코드 내용과 100% 일치하는, 수정이 필요한 '원본 코드 블록'이어야 합니다.
4. "replace_block"은 수정된 '새 코드 블록'입니다.
5. 다른 설명이나 ```json``` 마크다운 없이 순수한 JSON 배열만 반환하십시오.
# [PATCH JSON]
"""
                try:
                    patch_json_str = await get_llm_response_async(patch_prompt, response_mime_type="application/json")
                    patch_report = await self.apply_llm_suggestions_async(patch_json_str)
                    
                    if patch_report.get("applied") and not patch_report.get("failed"):
                        print(f"  [Autofix] 정밀 수정 패치 성공: {len(patch_report['applied'])}개 적용.")
                        fix_applied = True
                    else:
                        print(f"  [Autofix] ⚠️ 정밀 수정 패치 실패 또는 충돌 발생. (Applied: {len(patch_report.get('applied', []))}, Failed: {len(patch_report.get('failed', []))})")
                        # 실패 시, 다음 루프에서 동일 오류가 발생하면 전체 재작성으로 넘어감
                except Exception as e_patch:
                    print(f"  [Autofix] ⚠️ 정밀 수정 프로세스 중 예외 발생: {e_patch}")

            if not fix_applied:
                print("  [Autofix] 코드 수정/재생성에 실패하여 루프를 중단합니다.")
                return False

            # 4. 재시도 (실패했던 '바로 그 단계'만 다시 실행)
            failed_step_index = failed_task.get("current_step", 0) - 1
            if failed_step_index < 0:
                return False
            
            retry_plan_json = json.dumps([failed_task["plan"][failed_step_index]])
            project_dir = failed_task.get("project_directory")
            
            print(f"  [Autofix] Retrying step {failed_step_index + 1}...")
            retry_result = await self._execute_task(retry_plan_json, project_dir)

            # 5. 재시도 성공 여부 판단
            if "Error:" in retry_result or "실패:" in retry_result:
                print(f"  [Autofix] Retry failed: {retry_result}")
                failed_task["last_result"] = retry_result # 다음 루프를 위해 에러 업데이트
                return False
            else:
                print("  [Autofix] Retry successful!")
                return True

        except Exception as e:
            print(f"❌ [Autofix] 자동 수정 루프 중 심각한 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            return False

    # [!!! v20.1 신규: 전체 코드 재생성(Full Code Regeneration) 모듈 !!!]
    async def _regenerate_full_code_async(self, error_details: Dict, current_code: str, original_goal: str, error_filepath: str) -> bool:
        """
        [Autofix Strategy 2] 반복적인 오류 발생 시, Ni/Te 원칙을 적용하여
        LLM에게 파일 전체의 재설계를 요청하고 결과를 덮어씁니다.
        """
        print(f"🔥 [Autofix-Regen] '{os.path.basename(error_filepath)}' 파일 전체 재생성 시도...")
        
        try:
            # 1. Ni/Te 가이드라인을 포함한 프롬프트 생성
            prompt = f"""
            # [SYSTEM] 당신은 20년 경력의 수석 소프트웨어 아키텍트입니다.
            # [CONTEXT]
            - 현재 작업 중인 파일: '{os.path.basename(error_filepath)}'
            - 원래 목표: '{original_goal}'
            - 이 파일은 다음 오류가 '반복적으로' 발생하여 단일 수정으로는 해결이 어렵습니다.
            
            # [반복된 오류 정보]
            - 오류 종류: {error_details.get('classified_type')}
            - 오류 메시지: {error_details.get('error_message')}
            - 발생 위치: Line {error_details.get("root_cause", {}).get('line')}
            - 전체 Traceback:
            {error_details.get('full_traceback')}

            # [기존 전체 코드]
            ```python
            {current_code}
            ```

            # [지시사항 (CRITICAL)]
            1.  **(Ni - Vision & Insight):** 기존 코드의 구조적 문제를 근본적으로 파악하십시오. 단순히 오류만 수정하지 말고, 이 클래스의 오류를 '예방'할 수 있는 더 나은 아키텍처(패턴, 추상화 등)를 구상하여 코드를 '재설계'하십시오. 미래 확장성을 고려해야 합니다.
            2.  **(Te - Logic & Effectiveness):** 재설계된 코드는 논리적으로 완결되어야 하며, 원래 목표를 '효율적'이고 '안정적'으로 달성해야 합니다. 불필요한 복잡성을 제거하고 명확한 코드를 작성하십시오.
            3.  **(Output):** 다른 설명 없이, 재설계된 Python 코드 '전체'를 코드 블록 안에 반환하십시오. 주석을 상세히 추가하여 당신의 설계 의도를 설명하십시오.
            
            # [재설계된 전체 코드 (Full Code)]
            """
            
            # 2. LLM 호출
            regenerated_code = await get_llm_response_async(prompt)
            
            # (LLM이 코드 블록을 포함해서 반환할 수 있으므로 파싱)
            code_match = re.search(r"```(?:python)?\n(.*)```", regenerated_code, re.DOTALL)
            if code_match:
                final_code = code_match.group(1).strip()
            else:
                final_code = regenerated_code.strip()

            if len(final_code) < 10: # 너무 짧으면 실패로 간주
                print("  [Autofix-Regen] LLM이 유효한 코드를 반환하지 않았습니다.")
                return False

            # 3. Diff 비교 (로깅용)
            import difflib
            diff = difflib.unified_diff(
                current_code.splitlines(keepends=True),
                final_code.splitlines(keepends=True),
                fromfile='original',
                tofile='regenerated'
            )
            print("  [Autofix-Regen] Code Diff:")
            for line in diff:
                print(f"    {line.strip()}")

            # 4. [Autofix Safety] 코드 변경 여부 확인
            if final_code.strip() == current_code.strip():
                print("  [Autofix-Regen Safety] ⚠️ LLM이 원본과 동일한 코드를 반환했습니다. 변경사항이 없어 수정을 건너뜁니다.")
                return False

            # 5. 파일 덮어쓰기 (핵심)
            print(f"  [Autofix-Regen] 재생성된 코드를 '{error_filepath}'에 덮어씁니다.")
            await write_file(filepath=error_filepath, content=final_code)
            
            return True

        except Exception as e:
            print(f"❌ [Autofix-Regen] 전체 코드 재생성 중 오류: {e}")
            return False

    def _apply_single_proposal_in_memory(
        self, current_code: str, proposal: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """
        [v-New Helper] 단일 수정 제안을 인메모리 코드에 적용합니다.
        사용자 요청에 따라 모듈성 향상을 위해 분리되었습니다.
        
        Returns:
            (success_bool, result_code_or_error_str)
        """
        search_block = proposal.get("search_block")
        replace_block = proposal.get("replace_block")
        description = proposal.get("description", "Unnamed Proposal")

        if not all([search_block, isinstance(replace_block, str)]):
            error_msg = f"Patching failed: Proposal '{description}' has invalid format (missing search/replace block)."
            return False, error_msg

        # 충돌 감지: search_block이 정확히 1번만 나타나야 함
        match_count = current_code.count(search_block)
        if match_count == 1:
            new_code = current_code.replace(search_block, replace_block, 1)
            return True, new_code
        else:
            error_msg = f"Patching failed: Proposal '{description}' caused a conflict. The search_block was found {match_count} times (expected 1)."
            return False, error_msg

    def _validate_python_syntax(self, code: str) -> Tuple[bool, str]:
        """ [Autofix Helper] ast.parse를 사용하여 Python 코드의 구문 유효성을 검사합니다. """
        try:
            ast.parse(code)
            return True, ""
        except SyntaxError as e:
            # 더 유용한 오류 메시지 포맷팅
            return False, f"Line {e.lineno}: {e.msg}"

    async def _apply_proposals_sequentially_async(
        self, original_code: str, proposals: List[Dict[str, Any]]
    ) -> Tuple[bool, str, List[Dict[str, Any]]]:
        """
        [v-Refactor] 원본 코드에 여러 수정 제안을 순차적으로 적용(in-memory)합니다.
        충돌 발생 시, LLM을 호출하여 해결을 시도하는 고급 기능이 포함되어 있습니다.

        Returns:
            (success_bool, result_code_or_error_str, applied_proposals_list)
        """
        current_code = original_code
        applied_proposals = []
        last_successful_proposal = None # 충돌 해결 컨텍스트를 위해 마지막 성공 제안 추적

        for i, proposal in enumerate(proposals):
            success, result_or_error = self._apply_single_proposal_in_memory(current_code, proposal)

            if success:
                current_code = result_or_error
                applied_proposals.append(proposal)
                last_successful_proposal = proposal # 성공 시 업데이트
            else:
                # [v-New] 충돌 감지 및 해결 시도
                print(f"🔥 [AutoPatch] 제안 #{i+1} 적용 중 충돌 감지. LLM 해결사 호출...")
                
                resolved_proposal = await self._resolve_patch_conflict_async(
                    code_before_patch=current_code,
                    failing_proposal=proposal,
                    last_successful_proposal=last_successful_proposal
                )

                if resolved_proposal:
                    # 해결된 제안으로 재시도
                    print("  [AutoPatch] ✅ 충돌 해결됨. 해결된 제안으로 재시도...")
                    retry_success, retry_result = self._apply_single_proposal_in_memory(current_code, resolved_proposal)
                    if retry_success:
                        current_code = retry_result
                        # 원본 제안 대신 해결된 제안을 기록
                        resolved_proposal['description'] = f"[Resolved Conflict] {proposal.get('description', 'N/A')}"
                        applied_proposals.append(resolved_proposal)
                        last_successful_proposal = resolved_proposal
                    else:
                        print(f"❌ [AutoPatch] 해결된 제안도 적용 실패: {retry_result}")
                        return False, f"Resolved patch also failed: {retry_result}", applied_proposals
                else:
                    # 해결 실패 시 즉시 중단
                    print(f"❌ [AutoPatch] LLM이 충돌을 해결하지 못했습니다. 원본 오류: {result_or_error}")
                    return False, result_or_error, applied_proposals

        return True, current_code, applied_proposals

    # [!!! v20.2 REFACTORED: LLM 제안 자동 패치 모듈 !!!]
    async def apply_llm_suggestions_async(self, suggestions_json: str) -> Dict[str, Any]:
        """
        [v-Refactor] LLM이 제안한 코드 수정사항(JSON)을 자동으로 파일에 적용(패치)합니다.
        순차 적용 로직을 새로 구현된 `_apply_proposals_sequentially_async` 헬퍼로 위임하여
        코드의 가독성과 재사용성을 향상시켰습니다.
        """
        report = {"applied": [], "failed": []}
        try:
            # 1. 제안 파싱
            proposals = json.loads(suggestions_json)
            if not isinstance(proposals, list):
                raise ValueError("제안이 리스트 형식이 아닙니다.")
        except (json.JSONDecodeError, ValueError) as e:
            report["failed"].append({"error": "Invalid suggestions_json format", "details": str(e)})
            return report

        # 2. 파일 경로별로 제안 그룹화 (효율성)
        proposals_by_file = {}
        for p in proposals:
            filepath = p.get("filepath")
            if filepath:
                if filepath not in proposals_by_file:
                    proposals_by_file[filepath] = []
                proposals_by_file[filepath].append(p)
        
        # 3. 파일 단위로 패치 적용 및 검증
        for filepath, file_proposals in proposals_by_file.items():
            try:
                # 파일 읽기 (비동기)
                original_code = await read_file(filepath=filepath)
                if "Error:" in original_code or "실패:" in original_code:
                    raise FileNotFoundError(f"File not found or could not be read: {filepath}")

                # [!!! REFACTOR !!!] 4. 복잡한 순차 패치 로직을 새 헬퍼 함수에 위임합니다.
                success, result_or_error, applied_proposals = await self._apply_proposals_sequentially_async(original_code, file_proposals)

                if not success:
                    # 패치 중 충돌 발생
                    print(f"❌ [AutoPatch] 롤백! '{filepath}' 패치 중 충돌 발생: {result_or_error}")
                    # 실패한 제안 이후의 모든 제안을 '실패'로 기록
                    failed_proposals = file_proposals[len(applied_proposals):]
                    for p in failed_proposals:
                        report["failed"].append({"proposal": p.get("description"), "reason": result_or_error})
                    # 이미 적용된 부분은 '성공'으로 기록
                    if applied_proposals:
                        report["applied"].extend(applied_proposals)
                    continue  # 다음 파일로 이동

                # 패치 성공, `result_or_error`는 이제 `patched_code`임
                patched_code = result_or_error

                # 5. 변경된 파일에 대한 안정성 검사 (구문 검사)
                is_valid, error_message = self._validate_python_syntax(patched_code)

                if is_valid:
                    # 5a. 성공 시 파일 쓰기
                    try:
                        print(f"  [AutoPatch] ✅ Syntax OK. Writing changes to '{filepath}'...")
                        await write_file(filepath=filepath, content=patched_code)
                        report["applied"].extend(applied_proposals)
                    except Exception as e_write:
                        # 파일 쓰기 실패 시 롤백
                        print(f"❌ [AutoPatch] 롤백! '{filepath}' 파일 쓰기 실패: {e_write}")
                        for p in applied_proposals:
                            report["failed"].append({"proposal": p.get("description"), "reason": f"File write error: {e_write}"})
                else:
                    # 5b. 구문 오류 시 롤백
                    print(f"❌ [AutoPatch] 롤백! '{filepath}' 패치 적용 후 구문 오류 발생: {error_message}")
                    for p in applied_proposals:
                        report["failed"].append({"proposal": p.get("description"), "reason": f"SyntaxError after patch: {error_message}"})

            except Exception as e_file_process:
                print(f"❌ [AutoPatch] '{filepath}' 처리 중 오류 발생: {e_file_process}")
                for p in file_proposals:
                    report["failed"].append({"proposal": p.get("description"), "reason": str(e_file_process)})

        total_applied = len(report["applied"])
        total_failed = len(report["failed"])
        print(f"✅ [AutoPatch] LLM 제안 적용 완료. (성공: {total_applied}, 실패: {total_failed})")
        return report
            
        await asyncio.to_thread(self.save_memory)

        # --- 4. 자율 목표 생성 (기존과 동일) ---
        self.goal_system_update_counter += 1
        if self.goal_system_update_counter % self.GOAL_SYSTEM_UPDATE_RATE == 0:
            new_external_event_summary = await self._sense_external_world_async()
            await self._update_goal_system_async(external_event=new_external_event_summary)

        return task_to_return

    def _select_next_autonomous_task_sync(self) -> Optional[Dict[str, Any]]:
        """
        [v-Fix] 스케줄에서 다음에 실행할 자율 작업을 선택하여 반환합니다.
        우선순위: 
        1. RUNNING (이미 실행 중인 작업의 다음 단계)
        2. PENDING (대기 중인 새 작업)
        """
        # 1. 현재 실행 중인 작업이 있다면 최우선으로 반환 (연속 실행)
        for task in self.schedule:
            if task.get("status") == "RUNNING":
                return task

        # 2. 실행 중인 작업이 없다면, 대기 중인(PENDING) 작업 선택
        for task in self.schedule:
            if task.get("status") == "PENDING":
                # (선택 사항) 특정 조건(예: 시간)을 체크할 수도 있음
                return task

        return None

    # [v-Fix] 🐛 병목 수정: 동기(sync) 함수를 비동기(async) 함수로 변경
    async def _prune_memory_async(self):
        """ [v-Fix] 오래된 이벤트 로그 및 관련 데이터를 '비동기'로 정리(Pruning)합니다. """
        if len(self.event_log) <= MAX_EVENT_LOG_SIZE:
            return

        print(f"🧠 [Memory Pruning] Event log size ({len(self.event_log)}) exceeds limit ({MAX_EVENT_LOG_SIZE}). Pruning...")
        
        # 1. 유지할 이벤트 수 계산
        keep_count = int(MAX_EVENT_LOG_SIZE * PRUNING_KEEP_RATIO)
        
        # 2. 오래된 이벤트 (삭제될 이벤트) ID 수집
        # event_log는 [오래된것, ..., 최신것] 순서
        pruned_logs = self.event_log[:-keep_count] # 오래된 (N - keep_count)개
        pruned_event_ids = set(log[2] for log in pruned_logs if log[2]) # (vec, c, eid, i)

        # 3. event_log 자체를 Pruning (최신 keep_count개만 남김)
        self.event_log = self.event_log[-keep_count:]
        
        print(f"  [Memory Pruning] {len(pruned_event_ids)} old events pruned. New log size: {len(self.event_log)}.")

        # [v-Fix] 4. (병목) 이 작업은 CPU 집약적이므로 별도 스레드에서 실행
        def sync_prune_dicts():
            pruned_count = 0
            for eid in pruned_event_ids:
                # event_id_to_concepts, event_emotion_map, strategy_log
                if self.event_id_to_concepts.pop(eid, None): pruned_count += 1
                self.event_emotion_map.pop(eid, None)
                self.strategy_log.pop(eid, None)

            # [v-Fix] 5. (병목) 이 작업도 CPU 집약적
            keys_to_prune = [
                key for key in self.similarity_cache 
                if key[0] in pruned_event_ids or key[1] in pruned_event_ids
            ]
            for key in keys_to_prune:
                self.similarity_cache.pop(key, None)
            
            return pruned_count, len(keys_to_prune)

        try:
            pruned_count, pruned_cache_count = await asyncio.to_thread(sync_prune_dicts)
            print(f"  [Memory Pruning] Cleaned {pruned_count} related concepts and {pruned_cache_count} cache entries.")
        except Exception as e:
            print(f"❌ [Memory Pruning] 비동기 정리 중 오류: {e}")

    def run_tool(self, tool_name: str, input_data) -> str:
        """
        PromptCanvas 툴 노드에서 호출되는 동기 진입점.
        input_data: str(레거시) 또는 dict(ToolSchema 파싱 결과)
        - dict면 kwargs로 풀어서 전달 → write_file 등 다중 파라미터 도구 정상 동작
        - str이면 기존 방식대로 단일 인자 전달
        """
        import asyncio, inspect

        from execution_module import (
            perform_web_search, write_text, read_file, write_file,
            read_code_skeleton, calculate_math,
            write_project_files_async, write_complex_code_iteratively,
            execute_python_file,
            generic_calendar_create, generic_calendar_search, generic_calendar_delete,
        )
        tool_map = {
            "perform_web_search":           perform_web_search,
            "execute_python_file":          execute_python_file,
            "write_text":                   write_text,
            "read_file":                    read_file,
            "write_file":                   write_file,
            "read_code_skeleton":           read_code_skeleton,
            "calculate_math":               calculate_math,
            "write_project_files_async":    write_project_files_async,
            "write_complex_code_iteratively": write_complex_code_iteratively,
            "reload_tool_registry_async":   self.reload_tool_registry_async,
            "generic_calendar_create":      generic_calendar_create,
            "generic_calendar_search":      generic_calendar_search,
            "generic_calendar_delete":      generic_calendar_delete,
            "gcal_list_events":             self.gcal_list_events,
            "gcal_create_event":            self.gcal_create_event,
            "gcal_delete_event":            self.gcal_delete_event,
            "gmail_list_messages":          self.gmail_list_messages,
            "gmail_get_message":            self.gmail_get_message,
            "gmail_send":                   self.gmail_send,
            "gmail_create_draft":           self.gmail_create_draft,
            "gmail_mark_read":              self.gmail_mark_read,
            "gmail_archive":                self.gmail_archive,
            "gmail_trash":                  self.gmail_trash,
        }
        tool_map.update(self.dynamic_tool_functions)

        fn = tool_map.get(tool_name)
        if fn is None:
            return f"[run_tool] 알 수 없는 도구: '{tool_name}'"

        # ── input_data 타입 처리 ──────────────────────────────────
        # Canvas의 _execute_tool_structured가 dict를 넘길 때:
        # {"path": "...", "content": "..."} → kwargs로 풀어서 전달
        if isinstance(input_data, dict):
            kwargs = input_data
            input_str = json.dumps(input_data, ensure_ascii=False)
        elif isinstance(input_data, str):
            # JSON 문자열이면 파싱 시도
            try:
                parsed = json.loads(input_data)
                if isinstance(parsed, dict):
                    kwargs = parsed
                    input_str = input_data
                else:
                    kwargs = {}
                    input_str = input_data
            except (json.JSONDecodeError, ValueError):
                kwargs = {}
                input_str = input_data
        else:
            kwargs = {}
            input_str = str(input_data)

        try:
            sig = inspect.signature(fn)
            params = list(sig.parameters.keys())

            def _call_fn():
                # kwargs가 있으면 (다중 파라미터 도구) kwargs로 전달
                if kwargs and params and params[0] not in ('args', 'kwargs'):
                    return fn(**kwargs)
                elif kwargs:
                    # *args, **kwargs 시그니처 → kwargs 그대로 전달
                    return fn(**kwargs)
                # 단일 파라미터 도구 — 기존 방식
                elif params and params[0] in ("input_str", "query", "expression",
                                               "text", "content", "filepath", "path"):
                    return fn(**{params[0]: input_str})
                elif params:
                    return fn(input_str)
                else:
                    return fn()

            if inspect.iscoroutinefunction(fn):
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        import concurrent.futures
                        with concurrent.futures.ThreadPoolExecutor() as pool:
                            if kwargs:
                                future = pool.submit(asyncio.run, fn(**kwargs))
                            else:
                                future = pool.submit(asyncio.run,
                                    fn(**{params[0]: input_str}) if params else fn())
                            result = future.result(timeout=60)
                    else:
                        if kwargs:
                            result = loop.run_until_complete(fn(**kwargs))
                        else:
                            coro = fn(**{params[0]: input_str}) if params else fn()
                            result = loop.run_until_complete(coro)
                except Exception:
                    if kwargs:
                        result = asyncio.run(fn(**kwargs))
                    else:
                        result = asyncio.run(fn(input_str) if params else fn())
            else:
                result = _call_fn()

            return str(result) if result is not None else "(결과 없음)"
        except Exception as e:
            return f"[run_tool] '{tool_name}' 실행 오류: {e}"

    async def _on_missing_tool_async(
        self,
        tool_name: str,
        args_dict: dict,
        task_context: str = "",
    ) -> Optional[str]:
        """
        [ToolFactory 연동] 알 수 없는 도구 호출 감지 시 자동 도구 합성 파이프라인.

        파이프라인:
            1. DC Reasoner로 도구 스펙(함수명·설명·파라미터·코드) 추론
            2. create_and_register_tool_async로 파일 생성 + registry 등록
            3. reload_tool_registry_async로 즉시 활성화
            4. 활성화된 도구를 바로 실행하여 결과 반환

        Args:
            tool_name:    LLM이 요청했으나 존재하지 않는 도구명
            args_dict:    LLM이 넘긴 파라미터 dict
            task_context: 원본 사용자 요청 (스펙 추론에 사용)

        Returns:
            합성 도구 실행 결과 문자열, 실패 시 None
        """
        print(f"🔧 [ToolFactory] '{tool_name}' 없음 — 자동 합성 시도")

        # ── 1. 이미 합성 중인지 확인 (무한 재귀 방지) ──────────────────
        _synth_lock = getattr(self, "_tool_synthesis_in_progress", set())
        if tool_name in _synth_lock:
            print(f"  [ToolFactory] '{tool_name}' 이미 합성 중 — 스킵")
            return None
        _synth_lock.add(tool_name)
        self._tool_synthesis_in_progress = _synth_lock

        try:
            from llm_module import get_llm_response_async as _llm
            import json as _j, re as _re

            # ── 2. DC Reasoner로 도구 스펙 추론 ──────────────────────────
            args_hint = _j.dumps(args_dict, ensure_ascii=False) if args_dict else "{}"
            spec_prompt = f"""다음 상황에서 누락된 Python 도구 함수를 설계하라.

[요청된 도구명]
{tool_name}

[LLM이 넘긴 파라미터]
{args_hint}

[사용자 원본 요청]
{task_context[:300]}

[지시]
- 이 도구가 수행해야 할 기능을 파악하라
- 실제로 동작하는 async Python 함수를 작성하라
- 가능하면 표준 라이브러리(os, re, json, datetime, pathlib, urllib, zipfile, xml.etree, csv 등)로 해결하라
- 필요하면 다음 사전 설치된 라이브러리도 사용 가능: openpyxl, docx (python-docx), pypdf, pandas, requests, beautifulsoup4, olefile (있는 경우)
- 외부 lib import 는 함수 내부에서 try/except ImportError 로 감싸 graceful 처리하라
- 함수명은 반드시 '{tool_name}' 이어야 한다
- subprocess / os.system / eval / exec / __import__ 같은 위험 호출 금지 (자동 차단됨)

[출력 형식 — JSON만]
{{
  "function_name": "{tool_name}",
  "description": "이 도구가 하는 일 (한 문장)",
  "parameters": {{"파라미터명": "타입 및 설명"}},
  "python_code": "import os\\n\\nasync def {tool_name}(...):\\n    ..."
}}"""

            raw = await _llm(spec_prompt, response_mime_type="application/json")

            # JSON 파싱
            cleaned = _re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()
            try:
                spec = _j.loads(cleaned)
            except Exception:
                m = _re.search(r"\{[\s\S]+\}", cleaned)
                spec = _j.loads(m.group(0)) if m else {}

            if not spec or "python_code" not in spec:
                print(f"  [ToolFactory] 스펙 추론 실패 — JSON 파싱 오류")
                return None

            fn_name   = spec.get("function_name", tool_name)
            desc      = spec.get("description", "")
            params    = spec.get("parameters", {})
            code      = spec.get("python_code", "")

            print(f"  [ToolFactory] 스펙 추론 완료: {fn_name} — {desc}")

            # ── 2.5. 합성 코드 검증 (구문 + 위험 패턴) ──────────────────
            # [Bug Fix] 등록 전 AST 구문 검사 및 위험 패턴 차단
            try:
                import ast as _ast
                _ast.parse(code)
            except SyntaxError as _syn_e:
                print(f"  [ToolFactory] 구문 오류 — 합성 스킵: {_syn_e}")
                return None

            _BANNED_PATTERNS = [
                "os.system(", "subprocess.", "eval(", "exec(",
                "__import__(", "importlib.import_module(", "open(/",
                "shutil.rmtree", "os.remove(", "os.unlink(",
            ]
            _code_lower = code.lower()
            for _pat in _BANNED_PATTERNS:
                if _pat.lower() in _code_lower:
                    print(f"  [ToolFactory] 위험 패턴 감지({_pat!r}) — 합성 스킵")
                    return None

            # ── 3. 도구 파일 생성 + registry 등록 ────────────────────────
            from execution_module import create_and_register_tool_async as _create
            create_result = await _create(
                function_name=fn_name,
                description=desc,
                parameters_dict=params,
                python_code=code,
            )
            print(f"  [ToolFactory] 생성 결과: {create_result}")

            # [Bug Fix] "성공" 키워드가 없으면 실패로 판단 (기존 문자열 포함 체크보다 명확)
            if "성공" not in create_result:
                print(f"  [ToolFactory] 파일 생성 실패 — {create_result[:80]}")
                return None

            # ── 4. 즉시 리로드 ────────────────────────────────────────────
            reload_result = await self.reload_tool_registry_async()
            print(f"  [ToolFactory] 리로드: {reload_result}")

            # ── 5. 활성화된 도구 바로 실행 ───────────────────────────────
            func = self.dynamic_tool_functions.get(fn_name)
            if func is None:
                print(f"  [ToolFactory] 리로드 후에도 '{fn_name}' 없음")
                return f"[ToolFactory] '{fn_name}' 합성 완료. 다음 호출 시 사용 가능."

            # [Bug Fix] args_dict가 비어있으면 즉시 실행을 건너뜀.
            # DC 제안 경로에서는 args_dict={}로 호출되므로 실행 시도 자체가 TypeError.
            # 실제 도구 호출(LLM 플랜 실행)은 다음 turn에서 정상 args로 진행됨.
            if not args_dict:
                print(f"  [ToolFactory] '{fn_name}' 합성 완료 (args 없음 — 즉시 실행 생략)")
                return f"[ToolFactory] '{fn_name}' 합성 완료. 준비됨."

            try:
                exec_result = await func(**args_dict)
                result_str = str(exec_result) if exec_result is not None else "(결과 없음)"
                print(f"  [ToolFactory] '{fn_name}' 즉시 실행 성공: {result_str[:80]}")
                return result_str
            except Exception as _exec_e:
                print(f"  [ToolFactory] 즉시 실행 실패 ({_exec_e}) — 합성만 완료")
                return f"[ToolFactory] '{fn_name}' 합성 완료. 파라미터 재확인 후 재시도 필요."

        except Exception as e:
            print(f"  [ToolFactory] 자동 합성 실패 (무시): {e}")
            return None

        finally:
            _synth_lock.discard(tool_name)

    async def reload_tool_registry_async(self) -> str:
        """
        EIDOS가 'tool_registry.json'을 다시 읽어들여 새 도구를 활성화합니다.
        'create_and_register_tool_async' 호출 직후 사용해야 합니다.
        """
        print("🔄 [AI ToolLoader] '새로고침' 도구 호출됨.")
        try:
            # 동기 함수인 _load_dynamic_tools를 비동기로 실행
            await asyncio.to_thread(self._load_dynamic_tools, reload=True)
            
            new_tool_names = list(self.dynamic_tool_functions.keys())
            return f"도구 새로고침 완료. 현재 사용 가능한 동적 도구: {new_tool_names}"
        except Exception as e:
            return f"도구 새로고침 실패: {e}"

    async def _sense_external_world_async(self) -> Optional[str]:
        """
        [v15.11 업그레이드]
        외부 세계 '센서'입니다.
        '원본 데이터'를 LLM으로 '1차 요약'하고, 요약본을 반환합니다.
        """
        print("  [AI Sensor v15.11] 외부 세계 관찰(웹 검색) 시작...")
        
        try:
            # 1. 원본 스니펫 가져오기
            raw_data = await perform_web_search(query="latest world news headlines summary", num_results=3)
            
            if "Error:" in raw_data or "실패" in raw_data or "결과 없음" in raw_data:
                print("  [AI Sensor v15.11] 외부 세계 관찰 실패. (네트워크 오류 또는 결과 없음)")
                return None 

            # 2. [v15.11] 원본 데이터를 '이전 원본'과 비교 (요약 비용 절감)
            if raw_data == self.last_world_state_raw_data:
                # print("  [AI Sensor v15.11] 외부 세계에 새로운 변화 없음.")
                return None # 변화 없음
            
            print(f"🔥 AI Sensor v15.11] '새로운' 외부 세계 정보 감지! (원본 길이: {len(raw_data)})")
            self.last_world_state_raw_data = raw_data # 최신 원본 데이터로 업데이트
            
            # [!!! v15.11 업그레이드: 1차 요약 !!!]
            print(f"  [AI Sensor v15.11] LLM을 통해 원본 스니펫 1차 요약 중...")
            try:
                summary_prompt = f"""
                다음은 최신 뉴스 스니펫 원본입니다.
                이 모든 정보를 종합하여 "가장 중요한 핵심 사건 1~2가지"만 요약하세요.
                (예: "미국 연준, 금리 동결 발표" 또는 "주요 IT 기업, 새로운 AI 모델 공개")

                [뉴스 스니펫 원본]
                {raw_data}

                [핵심 요약]
                """
                summary_text = await get_llm_response_async(summary_prompt)
                summary_text = summary_text.strip().strip("'\"")
                
                print(f"  [AI Sensor v15.11] 1차 요약 완료: '{summary_text}'")
                
                # [v15.11] 요약된 정보를 반환
                return f"관찰된 외부 세계 정보 (요약): {summary_text}"

            except Exception as e_llm:
                print(f"❌ [AI Sensor v15.11] 1차 요약 LLM 호출 실패: {e_llm}. (원본 데이터로 대체)")
                # 요약에 실패하면 원본이라도 반환
                return f"관찰된 외부 세계 정보 (원본): {raw_data[:500]}..." # 너무 길지 않게 잘라서 반환

        except Exception as e:
            print(f"❌ [AI Sensor v15.11] 외부 세계 감지 중 오류: {e}")
            return None

    async def _query_knowledge_graph_async(
        self, 
        current_concept_ids: List[int]
    ) -> List[np.ndarray]:
        """
        [v14.3] 현재 입력된 개념 ID와 관련된 '지식(Knowledge)'과 '기억(Memory)'을
        그래프와 LTM에서 쿼리하고, 벡터화하여 반환합니다.
        (지식의 활용)
        """
        if not current_concept_ids:
            return []
            
        print(f"🧠 [AI Knowledge v14.3] {len(current_concept_ids)}개 개념 관련 지식/기억 쿼리 중...")
        knowledge_vectors = []
        
        try:
            # 1. 쿼리할 개념 이름 추출
            concept_names = [self.kb.get_name(cid) for cid in current_concept_ids if cid != 0]
            if not concept_names:
                return []
            query_str = " ".join(concept_names)

            # 2. 관련 '에피소드 기억' 재탐색 (LTM 모듈 재활용)
            #
            recalled_memories = await self.episodic_memory.recall_relevant_memories_async(
                query=query_str, top_k=2
            )
            
            for mem_text in recalled_memories:
                # 기억 텍스트를 OIA로 파싱 -> 벡터화
                mem_analysis = await analyze_event_with_llm_async(mem_text, None)
                mem_concepts = self._parse_llm_output(mem_analysis, "객체") + \
                               self._parse_llm_output(mem_analysis, "속성")
                mem_ids = [self.kb.get_or_create_id(c) for c in mem_concepts]
                if mem_ids:
                    vec = await asyncio.to_thread(
                        self.event_encoder.predict, np.array([mem_ids]), verbose=0
                    )
                    knowledge_vectors.append(vec[0])
                    print(f"  [AI Knowledge v14.3] 관련 기억 벡터 추가: {mem_text[:30]}...")

            # 3. 관련 '동화된 지식' 쿼리 (v14.1에서 생성됨)
            knowledge_nodes_content = []
            for cid in current_concept_ids:
                if not self.graph.has_node(cid): continue
                # 현재 개념과 연결된 'knowledge' 노드 탐색
                for neighbor in nx.all_neighbors(self.graph, cid):
                    node_data = self.graph.nodes[neighbor]
                    if node_data.get('type') == 'knowledge':
                        # (cid) -> (A-XXX) 엣지 또는 (A-XXX) -> (cid) 엣지 모두 탐색
                        content = node_data.get('content')
                        if content:
                            knowledge_nodes_content.append(content)
            
            # (중복 제거 및 상위 1개만 사용 - 성능)
            unique_content = list(set(knowledge_nodes_content))
            if unique_content:
                # (가장 긴=자세한 지식 1개만 벡터화)
                best_knowledge = max(unique_content, key=len)
                k_analysis = await analyze_event_with_llm_async(best_knowledge, None)
                k_concepts = self._parse_llm_output(k_analysis, "객체") + \
                             self._parse_llm_output(k_analysis, "속성")
                k_ids = [self.kb.get_or_create_id(c) for c in k_concepts]
                if k_ids:
                    vec = await asyncio.to_thread(
                        self.event_encoder.predict, np.array([k_ids]), verbose=0
                    )
                    knowledge_vectors.append(vec[0])
                    print(f"  [AI Knowledge v14.3] 관련 지식 벡터 추가: {best_knowledge[:30]}...")

        except Exception as e:
            print(f"❌ [AI Knowledge v14.3] 지식 쿼리 중 오류: {e}")

        return knowledge_vectors

    def _find_task_by_prompt(self, task_prompt: str) -> Optional[Dict[str, Any]]:
        """ 스케줄에서 task_prompt가 일치하는 '첫 번째' 작업을 찾습니다. """
        if not task_prompt:
            return None
        # [v15.2] COMPLETED나 CANCELED가 아닌 작업 우선 탐색
        for task in self.schedule:
            if task.get("task_prompt") == task_prompt and \
               task.get("status") not in ("COMPLETED", "CANCELED"):
                return task
        # 위에서 못찾으면 오래된 작업이라도 탐색
        for task in self.schedule:
             if task.get("task_prompt") == task_prompt:
                 return task
        return None

    # [v-Fix] 🐛 병목 수정: 동기 함수(def) -> 비동기(async def)로 변경
    async def pause_autonomous_task(self, task_prompt: str):
        """ 지정된 작업을 'PAUSED' 상태로 변경합니다. (GUI용) """
        task = self._find_task_by_prompt(task_prompt)
        if task and task.get("status") == "RUNNING":
            task["status"] = "PAUSED"
            # [v-Fix] 🐛 병목 수정: 동기 I/O 호출을 비동기 스레드로 이동
            await asyncio.to_thread(self.save_memory)
            print(f"⏸️ [AI Control] Task Paused by User: '{task_prompt}'")
        else:
            print(f"⚠️ [AI Control] Pause failed: Task '{task_prompt}' not found or not RUNNING.")

    # [수정] 함수를 비동기(async)로 변경하고 Lock을 사용
    async def delete_task_completely(self, task_prompt: str):
        """ 지정된 작업 프롬프트와 일치하는 작업을 스케줄 목록에서 '완전히' 제거합니다. """
        if not task_prompt: return
        
        # Lock을 비동기적으로 획득하여 self.schedule 접근 보호
        async with self.schedule_lock:
            initial_count = len(self.schedule)
            
            tasks_to_keep = [
                task for task in self.schedule 
                if task.get("task", "") != task_prompt and task.get("task_prompt", "") != task_prompt
            ]  
            
            removed_count = initial_count - len(tasks_to_keep)
            self.schedule = tasks_to_keep
            
            if removed_count > 0:
                print(f"🗑️ [AI Control] Task Completely Deleted: '{task_prompt}' ({removed_count} items removed)")
                # [v-Fix] 🐛 병목 수정: 동기 I/O 호출을 비동기 스레드로 이동
                await asyncio.to_thread(self.save_memory)
            else:
                print(f"⚠️ [AI Control] Delete failed: Task '{task_prompt}' not found.")

    # [v-Fix] 🐛 병목 수정: 동기 함수(def) -> 비동기(async def)로 변경
    async def resume_autonomous_task(self, task_prompt: str):
        """ 지정된 'PAUSED' 작업을 'PENDING' 상태로 변경하여 즉시 재시작 대기열에 넣습니다. (GUI용) """
        task = self._find_task_by_prompt(task_prompt)
        if task and task.get("status") == "PAUSED":
            task["status"] = "PENDING" # RUNNING이 아닌 PENDING (autonomous_tick이 선택하도록)
            # PAUSED 시점의 시간이 아닌 현재 시간으로 변경 (선택적)
            task["time"] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
            # [v-Fix] 🐛 병목 수정: 동기 I/O 호출을 비동기 스레드로 이동
            await asyncio.to_thread(self.save_memory)
            print(f"▶️ [AI Control] Task Resumed by User: '{task_prompt}' (Status: PENDING)")
        else:
            print(f"⚠️ [AI Control] Resume failed: Task '{task_prompt}' not found or not PAUSED.")

    # [v-Fix] 🐛 병목 수정: 동기 함수(def) -> 비동기(async def)로 변경
    async def cancel_autonomous_task(self, task_prompt: str):
        """ 지정된 작업을 'CANCELED' 상태로 변경하여 중단시킵니다. (GUI용) """
        task = self._find_task_by_prompt(task_prompt)
        if task:
            task["status"] = "CANCELED"
            task["notified"] = True # 다시 실행되지 않도록
            # [v-Fix] 🐛 병목 수정: 동기 I/O 호출을 비동기 스레드로 이동
            await asyncio.to_thread(self.save_memory)
            print(f"⏹️ [AI Control] Task Canceled by User: '{task_prompt}'")
        else:
            print(f"⚠️ [AI Control] Cancel failed: Task '{task_prompt}' not found.")

    def _load_dynamic_tools(self, reload: bool = False):
        """
        'eidos_tools' 디렉터리 및 'tool_registry.json'을 읽어
        자율적으로 생성된 도구들을 메모리로 동적 로드(import)합니다.
        """
        if reload:
            print("🔄 [AI ToolLoader] 도구 레지스트리 새로고침(Reload) 중...")
            self.dynamic_tool_registry.clear()
            self.dynamic_tool_functions.clear()
        else:
            print("🛠️ [AI ToolLoader] EIDOS 시작. 동적 도구 로드 중...")

        # 0. 'eidos_tools' 디렉터리를 Python 경로에 추가
        if self.TOOL_DIR not in sys.path:
            sys.path.insert(0, self.TOOL_DIR)
            
        # 1. 레지스트리 파일 읽기
        if not os.path.exists(self.REGISTRY_FILE):
            print(f"  [AI ToolLoader] '{self.REGISTRY_FILE}' 없음. (생성된 도구 0개)")
            return

        try:
            with open(self.REGISTRY_FILE, 'r', encoding='utf-8') as f:
                self.dynamic_tool_registry = json.load(f)
        except Exception as e:
            print(f"❌ [AI ToolLoader] '{self.REGISTRY_FILE}' 읽기 실패: {e}")
            return

        # 2. 각 도구를 동적으로 임포트
        loaded_count = 0
        failed_tools = []
        for function_name, metadata in self.dynamic_tool_registry.items():
            try:
                module_path_str = metadata.get("module_path") # 예: "eidos_tools.my_new_tool"
                
                # (파일 경로를 모듈 경로로 변환 - 예: eidos_tools/my_tool.py -> eidos_tools.my_tool)
                if module_path_str is None:
                    module_path_str = f"eidos_tools.{function_name}"

                # (Windows 경로 \를 .으로 수정)
                module_name = module_path_str.replace("\\", ".").replace("/", ".")
                
                print(f"  [AI ToolLoader] 모듈 임포트 시도: {module_name} (함수: {function_name})")

                # 3. 'importlib'를 사용한 동적 임포트
                if module_name in sys.modules and reload:
                    # 이미 로드된 모듈이면 'reload'
                    module = importlib.reload(sys.modules[module_name])
                else:
                    # 'import eidos_tools.my_new_tool'과 동일
                    module = importlib.import_module(module_name) 
                
                # 4. 모듈에서 함수 객체 가져오기
                function_obj = getattr(module, function_name)
                
                # 5. 실행기 맵에 등록
                self.dynamic_tool_functions[function_name] = function_obj
                loaded_count += 1

            except ImportError as e_import:
                print(f"❌ [AI ToolLoader] '{function_name}' 임포트 실패 (파일 없음?): {e_import}")
                failed_tools.append(function_name)
            except AttributeError:
                print(f"❌ [AI ToolLoader] '{module_name}' 모듈에서 '{function_name}' 함수를 찾을 수 없음.")
                failed_tools.append(function_name)
            except Exception as e:
                print(f"❌ [AI ToolLoader] '{function_name}' 로드 중 알 수 없는 오류: {e}")
                failed_tools.append(function_name)
        
        # 실패한 도구는 레지스트리에서 제거 (선택적)
        for name in failed_tools:
            self.dynamic_tool_registry.pop(name, None)
            
        print(f"✅ [AI ToolLoader] 동적 도구 {loaded_count}개 로드 완료.")

    async def _schedule_marketing_for_task_async(self, completed_task: dict):
        """
        (Core 헬퍼) 완료된 작업을 받아, '홍보 자동화' 작업을 스케줄에 등록합니다.
        """
        try:
            task_name = completed_task.get("task_prompt", "무제")
            
            # 1. '자율 작업'(내부)이거나 'Step'(중간)이면 홍보 안 함
            if task_name.startswith("자율 작업 수행 (트리거:") or "(Step" in task_name:
                print(f"  [OIA-M] '{task_name}'은(는) 내부 작업이므로 홍보를 건너뜁니다.")
                return

            print(f"🚀 [OIA-M] '{task_name}' 작업 완료! '홍보 자동화'를 시작합니다...")
            
            # (임시) 상품 설명과 링크 (향후엔 task에서 추출)
            product_description = completed_task.get("procedure", "EIDOS가 생성한 디지털 상품")
            product_link = f"https://eidos.buildship.com/p/{task_name.replace(' ', '-')}"

            # 2. (LLM 호출) 마케팅 콘텐츠 생성
            from llm_module import generate_marketing_content_async
            content_dict = await generate_marketing_content_async(
                task_name, product_description, product_link
            )
            
            main_post = content_dict.get("main_post")
            first_comment = content_dict.get("first_comment")
            
            if not main_post or main_post.startswith("LLM"):
                raise ValueError(f"LLM 마케팅 콘텐츠 생성 실패: {main_post}")

            # 3. (Core) 새 '홍보' 작업을 스케줄에 등록
            # (예: 1시간 뒤에 발행)
            import datetime
            publish_time = datetime.datetime.now() + datetime.timedelta(hours=1)
            time_str = publish_time.strftime('%Y-%m-%d %H:%M')
            
            new_task_prompt = f"자율 작업 수행 (트리거: OIA-M): {task_name} 홍보"
            new_plan = [
                {
                    # [!!! v-OIA-M 수정: 새 도구 이름으로 변경 !!!]
                    "tool": "publish_marketing_content_async", 
                    "args": {
                        "product_name": task_name,
                        "main_post": main_post,
                        "first_comment": first_comment
                    },
                    "risk": "Zapier Webhook 호출이 실패할 수 있습니다.",
                    "contingency": "다음 틱(Tick)에서 재시도합니다."
                }
            ]
            
            # (Lock을 잡고 스케줄에 추가)
            async with self.schedule_lock:
                await self.add_schedule_item(
                    time_str=time_str,
                    task=new_task_prompt,
                    procedure=f"'{task_name}' 상품에 대한 자동 홍보 포스팅",
                    plan=new_plan
                )
            
            print(f"✅ [OIA-M] 마케팅 작업 스케줄링 완료. ({time_str}에 발행 예정)")

        except Exception as e:
            print(f"❌ [OIA-M] 마케팅 자동화 스케줄링 중 심각한 오류: {e}")
            import traceback
            traceback.print_exc()

    # [!!! v15.15 신규 함수 1: '문맥 강화' (파일 시스템 읽기) !!!]
    async def _get_project_context_async(self, file_list: List[str], project_root: str) -> str: # <<<--- [!!! 1. "eidos_files" 기본값 제거 !!!]
        """
        [v15.15] '반성'을 위해, 방금 생성된 프로젝트 파일들의 실제 내용을 읽어
        LLM 플래너에게 전달할 '문맥(Context)'을 생성합니다.
        [v19.10] project_root가 '절대 경로' 또는 'eidos_files/project'일 수 있도록 수정.
        """
        print(f"  🧠 [Reflect-Context] '{project_root}'의 파일 {len(file_list)}개 스캔 중...")
        
        # [!!! 2. 경로 로직 수정 !!!]
        # SCRIPT_DIR는 이 파일의 위치
        SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) 
        
        # project_root가 절대 경로가 아니면(예: 'eidos_files/my_proj'), SCRIPT_DIR 기준으로 변환
        if not os.path.isabs(project_root):
            safe_base_path = os.path.normpath(os.path.join(SCRIPT_DIR, project_root))
        else:
            safe_base_path = os.path.normpath(project_root) # 이미 절대 경로임
        # [!!! 2. 수정 완료 !!!]

        # 1. 파일 내용을 읽어 하나의 문자열로 병합 (비동기)
        async def sync_read_all():
            # [!!! ⬇️ 생략된 부분 ⬇️ !!!]
            full_context = [f"# Project Context (Root: {project_root})\n"]
            # (너무 길어지는 것을 방지하기 위해 최대 5개 파일)
            for relative_path in file_list[:min(len(file_list), 5)]: 
                try:
                    target_path = os.path.normpath(os.path.join(safe_base_path, relative_path))
                    
                    # (보안 검사: safe_base_path 외부로 나가는 경로 차단)
                    if os.path.commonprefix([target_path, safe_base_path]) != safe_base_path:
                        print(f"  [Reflect-Context] ⚠️ 건너뛰기 (보안): '{relative_path}'가 샌드박스 외부에 있습니다.")
                        continue
                    
                    if os.path.exists(target_path):
                        with open(target_path, 'r', encoding='utf-8') as f:
                            content = f.read(1000) # (파일당 최대 1000자)
                        full_context.append(f"\n--- File: {relative_path} ---\n{content}\n")
                    else:
                        full_context.append(f"\n--- File: {relative_path} (File not found) ---\n")
                except Exception as e:
                    full_context.append(f"\n--- File: {relative_path} (Error reading: {e}) ---\n")
            return "".join(full_context)
            # [!!! ⬆️ 생략된 부분 ⬆️ !!!]

        try:
            context_str = await asyncio.to_thread(sync_read_all)
            return context_str
        except Exception as e:
            print(f"❌ [Reflect-Context] 프로젝트 문맥 읽기 실패: {e}")
            return f"# Error: Failed to read project context ({e})"

    # [!!! v15.15 신규 함수 2: '반성 기능 강화' (다음 단계 계획) !!!]
    async def _create_code_modification_plan_async(
        self, 
        original_goal: str, 
        project_context: str
    ) -> Optional[List[Dict]]:
        """
        [v15.15] '프로젝트 문맥'을 바탕으로, '# TODO'나 'pass'를 채우기 위한
        'read_file -> write_text -> write_file' 계획을 'LLM 플래너'에게 요청합니다.
        """
        print(f"  🧠 [Reflect-Plan] '문맥' 기반 2단계 계획 생성 요청...")
        
        try:
            planner_input = {
                "type": "POST_TASK_REFLECTION_V2", # [v15.15] 신규 반성 타입
                "context": {
                    "original_goal": original_goal,
                    "project_context": project_context # 방금 읽은 파일들의 실제 내용
                }
            }
            
            # 2. 메인 플래너(_generate_dynamic_plan_async)를 '반성 모드'로 호출
            plan_json_str = await self._generate_dynamic_plan_async(
                text_input=planner_input,
                image_input=None,
                chat_history=[],
                multimodal_vector=np.zeros((1, LSTM_UNITS)),
                is_user_task=True # <<<--- [!!! v19.10 수정: 이 인수를 추가하세요 !!!]
            )

            if plan_json_str.strip().startswith("Error:"):
                print(f"❌ [Reflect-Plan] 2단계 계획 생성 실패(LLM): {plan_json_str}")
                return None
            if "No further steps required." in plan_json_str:
                print(f"✅ [Reflect-Plan] 2단계 계획 불필요 (루프 종료).")
                return None
                
            plan_list = json.loads(plan_json_str)
            return plan_list

        except Exception as e:
            print(f"❌ [Reflect-Plan] 2단계 계획 생성 중 예외: {e}")
            return None

    async def _assimilate_execution_result_async(self, execution_result_event: str, chat_history: List[str]):
        """
        [v2] 실행 결과를 '학습'하지만 '행동 결정'은 하지 않는 동화 전용 함수
        """
        print(f"🔄 [Assimilation] 실행 결과 동화 시작: '{execution_result_event[:50]}...'")

        try:
            # 1. LLM 분석 (OIA 파서)
            analysis_dict = await analyze_event_with_llm_async(execution_result_event, None)
            if not analysis_dict:
                print("❌ [Assimilation] 실행 결과 분석 실패.")
                return

            parsed_objects = analysis_dict.get("객체", [])
            parsed_interactions = analysis_dict.get("상호작용", [])
            parsed_properties = analysis_dict.get("속성", [])

            concept_ids = (
                [self.kb.get_or_create_id(o) for o in parsed_objects] +
                [self.kb.get_or_create_id(i) for i in parsed_interactions] +
                [self.kb.get_or_create_id(p) for p in parsed_properties]
            )

            if not concept_ids:
                print("  [Assimilation] 동화할 개념 없음.")
                return

            # 2. 이벤트 벡터 생성
            input_tensor = np.array([concept_ids])
            current_event_vector = await asyncio.to_thread(
                self.event_encoder.predict, input_tensor, verbose=0
            )

            # 3. 이벤트 로그 저장 및 그래프 업데이트 (기존 로직)
            current_event_id_str = f"A-{len(self.event_log) + 1:03d}" # 'A'ssimilation
            self.event_log.append(
                (current_event_vector[0], concept_ids, current_event_id_str, parsed_interactions)
            )
            self.event_id_to_concepts[current_event_id_str] = concept_ids
            self._update_visualization_graph(analysis_dict)

            # ... (이하 _link_event_to_abstractions_realtime, CausalEngine, OTLE, memory_buffer.append 등) ...
            # ... '학습'과 '기록'에 관련된 모든 로직을 여기에 포함 ...

            # [중요] '정책 결정(action=...)' 및 '재귀 호출' 로직은 절대 포함하지 않음.

            print(f"✅ [Assimilation] 실행 결과 '{current_event_id_str}' 학습 완료.")

        except Exception as e:
            print(f"❌ [Assimilation] 동화 중 오류: {e}")

    async def _assimilate_dc_reasoning_async(
        self,
        user_input: str,
        dc_result: str,
        chat_history: list,
        suggested_tool: Optional[dict] = None,  # [ToolFactory] DC가 제안한 새 도구
    ) -> None:
        """
        [수리] DC Reasoner 분석 결과를 EpisodicMemory에 저장.
        GUI의 _assimilate_dc_reasoning_async 호출에 응답.

        대화 내용 + DC 추론 결과를 요약해서:
        1. KB 그래프에 memory 노드로 등록
        2. event_log에 간단한 에피소드로 추가
        3. EpisodicMemoryModule.run_summarization_cycle() 트리거
        4. [ToolFactory] suggested_tool이 있으면 자동 도구 합성 트리거
        """
        try:
            import time as _time

            # ── 에피소드 요약 텍스트 생성 ──────────────────────────────
            summary = (
                f"[대화 에피소드] 사용자 입력: {user_input[:100]}\n"
                f"DC 분석 결론: {dc_result[:200]}"
            )

            # ── KB 그래프에 memory 노드 등록 (중복 방지) ───────────────
            node_name = f"Episode_{int(_time.time())}"
            # 최근 에피소드가 동일 내용이면 스킵
            recent_labels = [
                self.graph.nodes[n].get("label", "")
                for n in self.graph.nodes
                if self.graph.nodes[n].get("type") == "episode"
            ]
            if any(user_input[:50] in lb for lb in recent_labels[-3:]):
                print("  [EpisodicMemory] 중복 에피소드 — 저장 스킵")
                return

            node_id = self.kb.get_or_create_id(node_name, item_type="episode")
            self.graph.add_node(
                node_id,
                type="episode",
                label=summary,
                timestamp=_time.time(),
                user_input=user_input[:200],
                dc_result=dc_result[:300],
            )
            self.graph.add_edge(
                self.eidos_self_id_int, node_id,
                key="remembers_episode",
                label="에피소드 기억",
            )

            # ── event_log에 간단한 벡터 에피소드 추가 ──────────────────
            concept_ids = [
                self.kb.get_or_create_id(w)
                for w in (user_input + " " + dc_result[:100]).split()[:10]
            ]
            if concept_ids:
                ev_id = f"EP-{len(self.event_log)+1:04d}"
                # 이벤트 벡터 (간단히 평균으로 생성)
                try:
                    input_tensor = np.array([concept_ids[:8]])
                    ev_vec = await asyncio.to_thread(
                        self.event_encoder.predict, input_tensor, verbose=0
                    )
                    self.event_log.append((ev_vec[0], concept_ids, ev_id, []))
                    self.event_id_to_concepts[ev_id] = concept_ids
                except Exception:
                    pass  # 벡터 생성 실패 시 그래프 노드만 저장

            # ── 일정 에피소드 쌓이면 요약 사이클 트리거 ─────────────────
            if len(self.event_log) % 10 == 0 and len(self.event_log) > 0:
                await self.episodic_memory.run_summarization_cycle()

            print(f"  [EpisodicMemory] 에피소드 저장 완료: {node_name}")

            # ── [5번] 감정 맥락 태그 기록 ──────────────────────────────
            # 에피소드 저장 시점의 감정 상태를 EmotionContextMemory에 함께 기록
            try:
                _ev = self.emotion_state.activations.tolist()

                # Appraisal: DC 결론 valence 추정 (부정 키워드 비율)
                _neg_kw = ["실패", "위험", "감소", "악화", "불리", "위협", "하락", "어려움"]
                _pos_kw = ["성공", "기회", "증가", "개선", "유리", "강화", "상승", "성장"]
                _neg_cnt = sum(1 for k in _neg_kw if k in dc_result)
                _pos_cnt = sum(1 for k in _pos_kw if k in dc_result)
                _total = _neg_cnt + _pos_cnt or 1
                _valence_est = (_pos_cnt - _neg_cnt) / _total
                _coping = max(0.2, min(0.8,
                    0.5 + (_pos_cnt - _neg_cnt) * 0.1
                ))

                # 에피소드 그래프 노드에도 감정 태그 속성 추가
                if self.graph.has_node(node_id):
                    self.graph.nodes[node_id]["emotion_tag"] = \
                        self.emotion_context_memory._compute_circumplex(_ev)[2]
                    self.graph.nodes[node_id]["energy_level"] = \
                        round(sum(_ev) / EMOTION_DIM, 3)
                    self.graph.nodes[node_id]["topic_sensitivity"] = \
                        user_input[:30]

                self.emotion_context_memory.record(
                    emotion_vec=_ev,
                    user_input=user_input,
                    episode_summary=summary[:100],
                    coping_potential=_coping,
                )
            except Exception as _em_e:
                print(f"  [EmotionContext] 태그 기록 실패 (무시): {_em_e}")

            # ── [4번] DC 결론 → DNAEvolver 로 persona DNA 진화 트리거 ──
            # DC가 발견한 인사이트를 DNA에 반영 → 다음 대화부터 더 정교한 맥락
            try:
                from eidos_dc_reasoner import DCReasoner as _DCR
                from eidos_situation_dna import DNAEvolver as _DEv
                import json as _ej, os as _eos

                # 제어실 top_goal 로드
                _top_goal = ""
                _sf = "eidos_settings.json"
                if _eos.path.exists(_sf):
                    with open(_sf, "r", encoding="utf-8") as _ef:
                        _top_goal = _ej.load(_ef).get("top_goal", "").strip()

                if _top_goal:
                    _dcr = _DCR()
                    _persona_key = _dcr.make_persona_key(_top_goal)
                    _my_dna = _dcr._load_dna_cache(_persona_key) or _dcr._load_dna_cache("my_situation")

                    if _my_dna:
                        # DC 결론 텍스트에서 핵심 변화 신호 추출해서 DNA에 주입
                        _evolver = _DEv()
                        # evolve: dc_result를 외부 압력으로 해석해 DNA 적응
                        _evolved = await asyncio.to_thread(
                            _evolver.mutate_from_feedback,
                            _my_dna,
                            dc_result[:300],
                        ) if hasattr(_evolver, "mutate_from_feedback") else None

                        if _evolved:
                            _dcr._save_dna_cache(_persona_key, _evolved)
                            print(f"  [DC→DNA] persona DNA 진화 완료: {_evolved.full_sequence[:20]}...")
                        else:
                            print(f"  [DC→DNA] DNAEvolver.mutate_from_feedback 없음 — 스킵")
            except Exception as _dna_e:
                print(f"  [DC→DNA] DNA 진화 실패 (무시): {_dna_e}")

            # ── [단절 1 해결] Episode → Knowledge 승급 ───────────────────
            # DC 결과의 valence를 reward로 변환 → _assimilate_knowledge_async 호출
            # episode 노드로만 머물지 않고 OIA 파싱 → knowledge 노드로 승급
            try:
                _neg_kw_k = ["실패", "위험", "감소", "악화", "불리", "위협", "하락", "어려움"]
                _pos_kw_k = ["성공", "기회", "증가", "개선", "유리", "강화", "상승", "성장"]
                _neg_c = sum(1 for k in _neg_kw_k if k in dc_result)
                _pos_c = sum(1 for k in _pos_kw_k if k in dc_result)
                _denom = _neg_c + _pos_c or 1
                # reward 범위: -1.0 ~ +1.0 (confidence 조정폭에 맞게 스케일)
                _dc_reward = float(np.clip((_pos_c - _neg_c) / _denom, -1.0, 1.0))

                # knowledge 승급용 이벤트 텍스트 구성
                # _assimilate_knowledge_async는 "최종 결과: " 접두사를 파싱 기준으로 사용
                _knowledge_event = (
                    f"최종 결과: [DC 추론 지식] 사용자 질문: {user_input[:80]}\n"
                    f"DC 결론: {dc_result[:400]}"
                )
                asyncio.ensure_future(
                    self._assimilate_knowledge_async(
                        goal_name=f"DC:{user_input[:40]}",
                        execution_result_event=_knowledge_event,
                        reward=_dc_reward,
                    )
                )
                print(f"  [DC→Knowledge] 승급 태스크 등록 (reward={_dc_reward:+.2f})")
            except Exception as _kn_e:
                print(f"  [DC→Knowledge] 승급 태스크 등록 실패 (무시): {_kn_e}")
            # ──────────────────────────────────────────────────────────────

            # ── [ToolFactory] DC가 제안한 도구 자동 합성 ─────────────────
            if suggested_tool and isinstance(suggested_tool, dict):
                _fn = suggested_tool.get("function_name", "")
                _desc = suggested_tool.get("description", "")
                if _fn and _fn not in self.dynamic_tool_functions:
                    try:
                        print(f"  [ToolFactory] DC 제안 도구 자동 합성 시작: {_fn}")
                        asyncio.create_task(
                            self._on_missing_tool_async(
                                tool_name=_fn,
                                args_dict={},
                                task_context=(
                                    f"DC Reasoner 제안: {_desc}\n"
                                    f"원본 요청: {user_input[:200]}"
                                ),
                            )
                        )
                    except Exception as _tf_e:
                        print(f"  [ToolFactory] DC 제안 도구 합성 실패 (무시): {_tf_e}")
                elif _fn:
                    print(f"  [ToolFactory] '{_fn}' 이미 존재 — 합성 스킵")

        except Exception as e:
            print(f"  [EpisodicMemory] 저장 실패 (무시): {e}")

    async def add_gui_task_async(self, name: str, due_date: str, prompt: str, project_directory: Optional[str] = None, parent_name: Optional[str] = None): # <<< 1. 인수 추가
        async with self.schedule_lock: # [v1] Lock 필수
            # [v-Fix] 🐛 병목 수정: add_schedule_item이 async일 수 있으므로 await
            # (이전 파일의 add_schedule_item이 async def로 수정되었다고 가정)
            await self.add_schedule_item(
                time_str=due_date,
                task=name,
                materials=None,
                procedure=prompt,
                plan=[], 
                project_directory=project_directory,
                parent_task_name=parent_name # <<< 2. 전달
            )

    async def update_task_criteria_async(self, project_dir: str, new_criteria_json_str: str):
        """
        (Worker가 호출) GUI에서 사용자가 수동 편집한 '평가 기준' JSON 문자열을 받아
        Core의 self.schedule에 업데이트하고 즉시 저장(save_memory)합니다.
        """
        if not project_dir or not new_criteria_json_str:
            return

        async with self.schedule_lock:
            task_found = False
            # 이 project_dir을 사용하는 '상위 작업'(부모가 없는 작업)을 찾습니다.
            for task in self.schedule:
                if (task.get("project_directory") == project_dir and
                    task.get("parent_task_name") is None): 
                    
                    task["evaluation_criteria"] = new_criteria_json_str
                    task_found = True
                    print(f"✅ [Core QA Update] '{task.get('task_prompt')}'의 평가지표가 수동으로 업데이트되었습니다.")
                    break
            
            if not task_found:
                # (Fallback) 상위 작업이 아니더라도 일단 찾아서 저장 시도
                for task in self.schedule:
                     if task.get("project_directory") == project_dir:
                         task["evaluation_criteria"] = new_criteria_json_str
                         task_found = True
                         print(f"✅ [Core QA Update] (Fallback) '{task.get('task_prompt')}'의 평가지표가 업데이트되었습니다.")
                         break

        if task_found:
            # [v-Fix] 🐛 병목 수정: 동기 I/O 호출을 비동기 스레드로 이동
            await asyncio.to_thread(self.save_memory) 
        else:
            # [🔥 FIX] 스케줄에 없는 프로젝트(단순 폴더 열기 등)인 경우 에러 로그를 띄우지 않음
            # 로컬 파일(eidos_qa_data.json)에는 이미 저장되었으므로 Core 동기화는 생략해도 안전함.
            # print(f"ℹ️ [Core QA Info] '{project_dir}'는 스케줄에 등록된 작업이 아니므로 Core 동기화를 건너뜁니다.")
            pass
    
    async def register_object_wave_async(self, 
                                         concept_name: str, 
                                         attributes_dict: Dict[str, float],
                                         weights_dict: Optional[Dict[str, float]] = None, # 추가
                                         phases_dict: Optional[Dict[str, float]] = None   # 추가
                                         ) -> str:
        """ [OIA 2.1] Advanced: 가중치와 위상을 포함한 파동 등록 """
        try:
            # 1. WaveObject 생성 (고급 옵션 포함)
            wave_obj = WaveObject(
                name=concept_name, 
                attributes=attributes_dict,
                weights=weights_dict,
                phases=phases_dict,
                scaling_method='minmax' # 유연한 스케일링 적용
            )
            
            concept_id = self.kb.get_or_create_id(concept_name)
            
            # 2. KB에 저장 (단순 벡터뿐만 아니라 메타데이터도 저장 필요)
            # 여기서는 편의상 최종 계산된 벡터만 저장하거나, 
            # KB 구조를 확장하여 weights/phases도 저장하는 것이 좋습니다.
            self.kb.object_waves[concept_id] = wave_obj.vector.tolist()
            
            # 3. 그래프 노드에 상세 정보 저장 (중요)
            if self.graph.has_node(concept_id):
                self.graph.nodes[concept_id]['wave_attributes'] = wave_obj.keys
                self.graph.nodes[concept_id]['wave_weights'] = weights_dict
                self.graph.nodes[concept_id]['wave_phases'] = phases_dict
                
                # 엔진 캐시 예열 (미리 파동 생성)
                self.object_wave_engine.to_wave_sine_summation(wave_obj)
            
            return f"성공: '{concept_name}' 파동 등록 (가중치/위상 포함)."
        except Exception as e:
            return f"오류: {e}"

    async def get_adjacent_object_analysis_async(self, concept_name: str) -> str:
        """ [OIA 2.0] 파동 공명값을 계산하여 인접 객체와의 관계를 분석합니다. """
        concept_id = self.kb.get_or_create_id(concept_name)
        if concept_id not in self.kb.object_waves:
            return f"분석 실패: '{concept_name}'의 파동 데이터가 없습니다."

        # 메인 객체 복원
        main_keys = self.graph.nodes[concept_id].get('wave_attributes', [])
        # 저장된 벡터와 키를 매핑하여 딕셔너리 복원 (WaveObject 재생성용)
        main_vec = self.kb.object_waves[concept_id]
        if len(main_keys) != len(main_vec):
             return "오류: 파동 데이터 차원 불일치."
        
        main_attrs = dict(zip(main_keys, main_vec))
        main_obj = WaveObject(concept_name, main_attrs, normalize=False) # 이미 정규화됨

        analysis_report = [f"📊 [파동 공명 분석] 대상: {concept_name}"]
        
        # 인접 객체들과 공명 계산
        for neighbor_id in nx.all_neighbors(self.graph, concept_id):
            if neighbor_id in self.kb.object_waves:
                neighbor_name = self.kb.get_name(neighbor_id)
                n_vec = self.kb.object_waves[neighbor_id]
                n_keys = self.graph.nodes[neighbor_id].get('wave_attributes', [])
                
                # 인접 객체 복원
                if len(n_keys) == len(n_vec):
                    n_attrs = dict(zip(n_keys, n_vec))
                    neighbor_obj = WaveObject(neighbor_name, n_attrs, normalize=False)
                    
                    # 공명 계산 (Engine 사용)
                    resonance = self.object_wave_engine.calculate_resonance(main_obj, neighbor_obj)
                    
                    # 결과 해석
                    relation_type = "중립"
                    if resonance > 0.8: relation_type = "🟢 강한 공명 (협력/증폭)"
                    elif resonance < 0.3: relation_type = "🔴 부조화 (충돌/소멸)"
                    
                    analysis_report.append(f"- vs {neighbor_name}: 공명도 {resonance:.4f} -> {relation_type}")

        # LLM에게 보낼 최종 프롬프트용 텍스트
        computed_context = "\n".join(analysis_report)
        
        # (이후 이 computed_context를 LLM 프롬프트에 포함시켜 더 정교한 해석을 요청)
        return computed_context

    # [Helper Method] 작업 재개 로직 분리 (중복 제거용)
    def _resume_waiting_task(self, task, user_input):
        print(f"👂 [AI Listening] 사용자 입력이 '질문에 대한 답변'으로 감지되었습니다.")
        original_prompt = task.get("task_prompt", "")
        last_question = task.get("last_result", "").replace("질문함: ", "")
    
        augmented_prompt = f"{original_prompt}\n\n[추가 정보 (Q&A)]\nQ: {last_question}\nA: {user_input}"
    
        system_command = """
        \n\n[시스템 강제 지시 (질문 루프 차단)]: 
        사용자의 답변을 받았습니다. **절대로** `ask_user_for_clarification` 도구를 다시 사용하여 질문하는 계획을 수립하지 마십시오.
        다음 논리적 단계는 **반드시** 검색, 파일 작성, 또는 프로젝트 생성과 같은 **실행(Execution) 단계**여야 합니다. 
        """
        augmented_prompt += system_command
    
        task["task_prompt"] = augmented_prompt
        task["procedure"] = augmented_prompt 
        task["status"] = "PENDING"
        task["current_step"] = 0 
        self.save_memory()

    async def _generate_forced_execution_plan_async(self, current_task_prompt: str) -> List[Dict]:
        """
        [FIX-LOOP-V3 + v-Planning-Fix] Q&A 답변을 받은 후 강제 실행 계획을 생성합니다.
        - ask_user_for_clarification 물리적 차단
        - 하드코딩된 "신뢰자본" 플랜 제거 → 요청 유형 감지 후 범용 플랜 생성
        """
        # 컨텍스트 파싱
        has_qa = "[추가 정보 (Q&A)]" in current_task_prompt
        initial_request = current_task_prompt.split("[추가 정보 (Q&A)]")[0].strip() if has_qa else current_task_prompt
        final_answer = ""
        if has_qa:
            try:
                final_answer = current_task_prompt.split("\nA: ")[-1].split("\n")[0].strip()
            except Exception:
                final_answer = ""

        # 요청 유형 감지 (LLM 호출 없음 — 키워드 기반)
        coding_kws   = ["만들어", "짜줘", "개발", "코드", "프로그램", "앱", "시스템", "crm", "봇", "툴"]
        search_kws   = ["검색", "찾아", "알려줘", "뭐야", "어때", "어떻게", "최신", "뉴스"]
        writing_kws  = ["써줘", "작성", "초안", "보고서", "정리", "요약", "번역"]

        req_lower = initial_request.lower()
        enriched = f"{initial_request}" + (f"\n[추가정보] {final_answer}" if final_answer else "")

        if any(k in req_lower for k in coding_kws):
            # 코드/앱 생성 요청 → Direct Scaffolding으로 위임
            print("  [Forced Plan] 코딩 요청 감지 → Direct Scaffolding 계획 생성")
            try:
                return await self._generate_project_scaffolding_plan_async(enriched)
            except Exception as e:
                print(f"  [Forced Plan] Scaffolding 실패({e}) → write_complex_code fallback")
                return [{
                    "tool": "write_complex_code_iteratively",
                    "args": {"filepath": "eidos_files/output.py", "task_description": enriched},
                    "risk": "요구사항 해석 오류 가능",
                    "contingency": "사용자가 결과를 보고 추가 수정 요청 가능"
                }]

        elif any(k in req_lower for k in writing_kws):
            print("  [Forced Plan] 문서 작성 요청 감지")
            return [{
                "tool": "perform_web_search",
                "args": {"queries": [initial_request[:80]]},
                "risk": "검색 결과 품질 변동",
                "contingency": "다음 단계에서 write_text로 합성"
            }, {
                "tool": "write_text",
                "args": {"prompt": f"다음 요청에 맞는 문서를 작성하라:\n{enriched}"},
                "risk": "형식 부적절 가능",
                "contingency": "사용자 피드백 후 재작성"
            }, {
                "tool": "write_file",
                "args": {"filepath": "eidos_files/output.txt", "content": "$PREV_STEP_RESULT"},
                "risk": "파일 쓰기 실패",
                "contingency": "로그 출력 후 종료"
            }]

        else:
            # 검색/정보 요청 (기본)
            print("  [Forced Plan] 정보/검색 요청 감지")
            return [{
                "tool": "perform_web_search",
                "args": {"queries": [initial_request[:80]]},
                "risk": "검색 결과 부족 가능",
                "contingency": "write_text로 가용 정보 합성"
            }, {
                "tool": "write_text",
                "args": {"prompt": f"검색 결과($PREV_STEP_RESULT)를 바탕으로 다음 요청에 답하라:\n{enriched}"},
                "risk": "정보 불충분 가능",
                "contingency": "사용 가능한 정보로 최선의 답변 작성"
            }]

    # [eidos_v4_0_core.py] generate_smart_proposals_async 함수 교체

    async def generate_smart_proposals_async(self, chat_history: List[str], current_file_context: str = "") -> List[Dict[str, str]]:
        """
        [v-OIA-PromptWizard V2] 제목(Label)과 실제 명령(Command)을 분리하여 생성합니다.
        """
        print("🧠 [Smart Wizard] 문맥 분석 및 추천 질문 생성 중...")

        recent_history = "\n".join(self._normalize_history(chat_history)[-5:]) if chat_history else "없음"
        
        # 파일 컨텍스트가 너무 길면 앞부분만 잘라서 토큰 절약
        file_info = "현재 열린 파일 없음"
        if current_file_context:
            file_info = f"현재 파일 내용(일부):\n{current_file_context[:3000]}..."

        prompt = f"""
        당신은 EIDOS의 '지능형 네비게이터'입니다.
        사용자의 [최근 대화]와 [현재 작업 중인 파일]을 분석하여,
        사용자가 지금 가장 필요로 할법한 **구체적인 행동 5가지**를 제안하세요.

        [분석 데이터]
        - 최근 대화 흐름:
        {recent_history}
        - 작업 환경:
        {file_info}

        [지시사항]
        1. **상황 인식**: 코드가 있으면 리팩토링/버그수정을, 빈 파일이면 뼈대 생성을, 대화 중이면 답변 관련 심화 질문을 제안하세요.
        2. **형식 준수**: 반드시 아래 JSON 구조의 리스트로 응답하세요.
           [
             {{ "icon": "🔍", "label": "버그 분석", "command": "이 코드의 잠재적인 버그를 찾아내고 수정안을 제시해줘." }},
             {{ "icon": "✨", "label": "주석 추가", "command": "이 파일의 모든 함수에 Docstring을 추가해줘." }}
           ]
        3. **Label**은 버튼에 표시될 10자 이내의 짧은 제목이어야 합니다.
        4. **Command**는 실제로 실행될 구체적이고 긴 프롬프트여야 합니다.

        [JSON 응답]
        """

        try:
            response_text = await get_llm_response_async(prompt, response_mime_type="application/json")
            
            # 파싱 (llm_module의 robust_json_parse 활용 권장)
            import json
            import re
            clean_text = re.sub(r'^```json\s*', '', response_text.strip(), flags=re.IGNORECASE)
            clean_text = re.sub(r'\s*```$', '', clean_text)
            
            proposals = json.loads(clean_text)
            
            if isinstance(proposals, list):
                return proposals[:6]
            else:
                # 실패 시 기본값 반환 (구조 맞춰서)
                return [{"icon": "⚠️", "label": "분석 실패", "command": "다시 시도해주세요."}]

        except Exception as e:
            print(f"❌ [Smart Wizard] 생성 실패: {e}")
            return [{"icon": "❌", "label": "오류 발생", "command": f"오류: {e}"}]

    async def _generate_project_scaffolding_plan_async(self, user_request: str) -> List[Dict]:
        """
        [Direct Scaffolding] 사용자 요청을 받아, 즉시 실행 가능한 프로젝트 뼈대 생성 계획을 수립합니다.
        복잡한 계획 수립 과정을 건너뛰고 'write_project_files_async' 도구를 직접 사용합니다.
        """
        print(f"  [Direct Scaffolding] LLM 호출: '{user_request[:30]}...' 프로젝트 구조 생성 요청")
        
        # 1. LLM에게 프로젝트 파일 구조와 초기 코드를 JSON 형식으로 요청
        prompt = f"""
        당신은 15년차 수석 아키텍트입니다. 사용자의 요청을 분석하여, 즉시 생성 가능한 Python 프로젝트의 파일 구조와 초기 뼈대 코드를 JSON 형식으로 생성하세요.

        # 사용자 요청:
        "{user_request}"

        # 지시사항:
        1. 요청에 맞는 합리적인 파일 구조를 설계하세요. (예: `main.py`, `models.py`, `utils.py`, `templates/index.html`)
        2. 각 파일에는 즉시 실행 가능하거나, 최소한의 기능이라도 담고 있는 뼈대 코드(Boilerplate)를 작성하세요. 주석으로 '# TODO'를 활용하여 다음 단계를 안내하면 좋습니다.
        3. 반드시 아래와 같은 JSON 형식으로만 응답해야 합니다. 다른 설명은 절대 추가하지 마세요.
        4. 키는 '파일 경로' (문자열), 값은 '파일 내용' (문자열)입니다.

        # JSON 출력 형식:
        {{
          "file_structure": {{
            "project_name/main.py": "import ...\\n\\n# main function",
            "project_name/models.py": "# database models here",
            "project_name/requirements.txt": "fastapi\\nuvicorn"
          }}
        }}
        """
        
        try:
            response_str = await get_llm_response_async(prompt, response_mime_type="application/json")
            
            response_data = json.loads(response_str)
            file_structure = response_data.get("file_structure")

            if not file_structure or not isinstance(file_structure, dict):
                raise ValueError("LLM이 유효한 'file_structure' 딕셔너리를 반환하지 않았습니다.")

            # 2. 'write_project_files_async' 도구를 사용하는 단일 단계 계획 생성
            scaffolding_plan = [
                {
                    "tool": "write_project_files_async",
                    "args": {
                        "file_structure": file_structure
                    },
                    "risk": "생성된 코드가 사용자의 세부 요구사항과 다를 수 있습니다.",
                    "contingency": "사용자가 생성된 파일을 기반으로 추가 수정을 요청할 수 있습니다."
                }
            ]
            
            print(f"  [Direct Scaffolding] ✅ 프로젝트 뼈대 계획 생성 완료 ({len(file_structure)}개 파일)")
            return scaffolding_plan

        except Exception as e:
            print(f"❌ [Direct Scaffolding] 뼈대 계획 생성 실패: {e}")
            raise e

    async def process_input(
        self,
        text_input: Union[str, Dict[str, Any], ExplanationContext],
        image_input: Optional[bytes],
        chat_history: List[str],
        project_dir: Optional[str] = None,
        _is_feedback_call: bool = False,
        _recursion_depth: int = 0,
        max_tokens=None,
        user_text_short: Optional[str] = None,
    ) -> Tuple[
        nx.MultiDiGraph, str, np.ndarray, float, bool, List[str],
        Optional[str], Optional[dict], str, Union[str, ExplanationResult], float, dict
    ]:
        """
        [B1] 5단계 파이프라인 오케스트레이터.
        각 단계는 독립 클래스로 분리되어 있어 교체/실험이 가능하다.

        Stage 1 InputRouter  : 조기 반환 (WAITING/피드백/인터럽트/Explanation)
        Stage 2 Perceiver    : 파일·비디오 처리 + Parallel0 (분류+분석+퓨전) + WorldModel
        Stage 3 Planner      : KB 업데이트 + 이벤트 벡터 + 감정 + CHAT 응답
        Stage 3.5 DCReasoner : 복잡한 문제 분할정복 추론 (SIMPLE이면 통과)
        Stage 4 Executor     : TASK 계획 수립 + 실행
        Stage 5 Integrator   : 감정 최종 업데이트 + 12-tuple 조립
        """
        chat_history = self._normalize_history(chat_history)
        print(f"\n--- [B1 Pipeline] Input: '{str(text_input)[:50]}...' "
              f"feedback={_is_feedback_call} ---")

        # ── [GoalPlan] 사용자 승인/거부 감지 ──────────────────────────
        _text_str = str(text_input).strip() if text_input else ""
        _APPROVE_KEYWORDS = {"승인", "확인", "좋아", "ㅇㅋ", "오케이", "ok", "approve", "lgtm", "괜찮아", "진행해"}
        _REJECT_KEYWORDS = {"거부", "수정", "다시", "reject", "변경", "고쳐"}
        if _text_str and hasattr(self, "goal_stack_evaluator"):
            _pm = getattr(self.goal_stack_evaluator, "plan_manager", None)
            if _pm:
                _awaiting = _pm.get_awaiting_goals()
                if _awaiting:
                    _input_lower = _text_str.lower().strip()
                    if any(kw in _input_lower for kw in _APPROVE_KEYWORDS):
                        for _ag in _awaiting:
                            _pm.approve_awaiting_step(_ag, _text_str)
                            # Goal 상태도 갱신
                            _g = next((g for g in self.long_term_goals if g.name == _ag), None)
                            if _g:
                                _g.status = "IN_PROGRESS"
                                _g.progress = _pm.get_plan_progress(_ag)
                                _g.last_updated = time.time()
                                if _pm.is_plan_completed(_ag):
                                    _g.completed = True
                                    _g.status = "COMPLETED"

                        # ── [수정] 승인 후 다음 step 자동 실행 트리거 ──────
                        # approve_awaiting_step이 step을 DONE으로 바꿨으므로
                        # GoalPlanManager가 다음 PENDING step을 schedule에 넣도록 유도
                        try:
                            gse = self.goal_stack_evaluator
                            if hasattr(gse, "_trigger_next_plan_step"):
                                for _ag in _awaiting:
                                    gse._trigger_next_plan_step(_ag)
                            elif hasattr(gse, "plan_manager"):
                                # fallback: 다음 PENDING step을 직접 schedule에 추가
                                for _ag in _awaiting:
                                    next_steps = _pm.get_next_pending_steps(_ag)
                                    for _ns in next_steps[:1]:  # 최대 1개씩
                                        _pm.schedule_step(_ag, _ns.step_id, self)
                        except Exception as _te:
                            print(f"  ⚠️ [AWAIT_USER] 다음 step 트리거 실패 (무시): {_te}")
                        # ─────────────────────────────────────────────────

                        # 승인 응답 반환
                        return (
                            self.graph,
                            "✅ 승인 완료! 다음 단계로 진행합니다.",
                            self.emotion_state.get_vector(), 0.8,
                            False, [], None, None,
                            "", "승인 처리 완료", 0.0, {}
                        )
                    elif any(kw in _input_lower for kw in _REJECT_KEYWORDS):
                        for _ag in _awaiting:
                            _pm.reject_awaiting_step(_ag, _text_str)

                        # ── [수정] 거부 후 이전 step 재실행 트리거 ──────────
                        try:
                            for _ag in _awaiting:
                                # reject 후 PENDING으로 돌아간 step을 schedule에 추가
                                pending_steps = _pm.get_next_pending_steps(_ag)
                                for _ps in pending_steps[:1]:
                                    _pm.schedule_step(_ag, _ps.step_id, self)
                        except Exception as _te:
                            print(f"  ⚠️ [AWAIT_USER] 재실행 트리거 실패 (무시): {_te}")
                        # ─────────────────────────────────────────────────

                        return (
                            self.graph,
                            "🔄 수정 요청을 반영합니다. 이전 단계부터 다시 실행됩니다.",
                            self.emotion_state.get_vector(), 0.5,
                            False, [], None, None,
                            "", "거부 처리 완료", 0.0, {}
                        )
        # ─────────────────────────────────────────────────────────────────

        # ── Stage 1: InputRouter ──────────────────────────────────────
        early, should_return = await self._input_router.route(
            text_input=text_input,
            image_input=image_input,
            chat_history=chat_history,
            _is_feedback_call=_is_feedback_call,
            user_text_short=user_text_short,
        )
        if should_return:
            print("  [B1] Stage 1 조기 반환")
            return early

        # ── Stage 2: Perceiver ────────────────────────────────────────
        percept = await self._perceiver.perceive(
            text_input=str(text_input),
            image_input=image_input,
            chat_history=chat_history,
        )
        if percept.get("error"):
            print(f"  [B1] Stage 2 오류: {percept['error']}")
            return (
                self.graph, "ERROR:LLM_ANALYSIS_FAILED",
                self.emotion_state.get_vector(), 0.0,
                False, [], None, None,
                "LLM 분석 실패", percept["error"],
                self.current_purity, self.current_complex_states,
                []
            )

        # ── Stage 3: Planner ──────────────────────────────────────────
        plan_result = await self._planner.plan(
            percept=percept,
            chat_history=chat_history,
        )

        # Fi Veto 조기 반환
        if plan_result.get("fi_veto"):
            msg = plan_result.get("fi_msg", "이 요청은 제 가치관과 맞지 않습니다.")
            return (
                self.graph, "FI_VETO",
                self.emotion_state.get_vector(), 0.0,
                False, [], None, None,
                "Fi 가치관 검증 거절", msg,
                self.current_purity, self.current_complex_states,
                []  # suggested_actions — tuple 길이 통일 (13-tuple)
            )

        # ── Stage 3.5: DCReasoner ─────────────────────────────────────
        try:
            if not hasattr(self, "_dc_reasoner"):
                self._dc_reasoner = DCReasoner()
            plan_result = await self._dc_reasoner.reason(
                plan_result=plan_result,
                percept=percept,
                chat_history=chat_history,
            )
        except Exception as _dc_err:
            print(f"  [B1] Stage 3.5 오류 (무시하고 진행): {_dc_err}")

        # ── Stage 4: Executor (TASK만) ────────────────────────────────
        exec_result: Optional[Dict] = None
        if percept["classification"] == "TASK":
            print("  [B1] Stage 4: Executor 진입")
            exec_result = await self._executor.execute(
                plan_ctx=plan_result,
                text_input=percept["text_input"],
                image_input=percept["image_input"],
                chat_history=chat_history,
                project_dir=project_dir,
                _is_feedback_call=_is_feedback_call,
                user_text_short=user_text_short,
            )

        # ── Stage 5: Integrator ───────────────────────────────────────
        result = await self._integrator.integrate(
            percept=percept,
            plan_result=plan_result,
            exec_result=exec_result,
            chat_history=chat_history,
            _is_feedback_call=_is_feedback_call,
        )
        print("  [B1] Pipeline 완료")
        return result