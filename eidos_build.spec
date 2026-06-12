# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_submodules, collect_data_files, copy_metadata
import sys
import os

block_cipher = None

# [2026-06-05] petalbot ↔ 요구사항 구체화기 연동 빌드 지원.
#   - EIDOS_ROOT: petalbot 패키지가 있는 곳(빌드 실행 위치). SPECPATH(=spec 디렉토리).
#   - ENGINE_ROOT: 요구사항 구체화기(`app` 패키지). EIDOS 트리 밖이라 절대경로로 참조
#     (obf_dist 등에 복사할 필요 없음). 옮겼으면 환경변수 REQSPEC_ENGINE_PATH 로 지정.
try:
    EIDOS_ROOT = SPECPATH                      # PyInstaller 가 주입하는 spec 디렉토리
except NameError:
    EIDOS_ROOT = os.getcwd()
ENGINE_ROOT = os.environ.get("REQSPEC_ENGINE_PATH", r"D:\EIDOS\요구사항 구체화기")
for _p in (EIDOS_ROOT, ENGINE_ROOT):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

# 1. 필수 라이브러리만 남김 (gensim, spacy, tqdm 제거됨!)
hidden_imports = [
    'google.generativeai', 'tensorflow', 'networkx', 'moviepy',
    'cv2', 'pydub', 'transformers', 'torch', 'numpy', 'pandas',
    'sklearn', 'scipy', 'aiohttp', 'asyncio', 'sympy', 'ast', 'imageio', 'imageio_ffmpeg'
]
hidden_imports += collect_submodules('sklearn')
hidden_imports += collect_submodules('scipy')
hidden_imports += collect_submodules('xml')
hidden_imports += collect_submodules('imageio')

# [2026-06-05] petalbot 전체(새 탭/브리지/워커) + 요구사항 구체화기 엔진(app) + 그 의존성.
#   frozen exe 는 시스템 site-packages 를 못 보므로 google.genai(신SDK)·reportlab·
#   openpyxl 까지 번들 필수. (기존 google.generativeai 는 구버전 — 별개 패키지)
for _pkg in ('petalbot', 'app', 'google.genai', 'reportlab', 'openpyxl'):
    try:
        hidden_imports += [_pkg] + collect_submodules(_pkg)
    except Exception as _e:
        print(f"[spec] WARN: collect_submodules('{_pkg}') 실패 — 경로 확인: {_e}")

# [2026-06-11] AIRA 비서 기능(메일/통화/일정) — eidos_chat_gui 함수 내부 lazy import(정적분석 누락 방지).
hidden_imports += ['eidos_aira_mail', 'eidos_aira_call', 'eidos_aira_stt',
                   'eidos_aira_audio', 'eidos_aira_schedule']

# 2. 파일 포함 리스트 (모델 파일 관련 로직 제거됨)
my_datas = [
    ('eidos_v4_0_core.py', '.'),
    ('eidos_v4_0_nn_models.py', '.'),
    ('eidos_world_model.py', '.'),
    ('eidos_multimodal_encoder.py', '.'),
    ('object_temporal_loop.py', '.'),
    ('eidos_v4_0_causal_engine_v10.py', '.'),
    ('eidos_v10_0_features.py', '.'),
    ('eidos_symbolic_reasoner.py', '.'),
    ('llm_module.py', '.'),
    ('execution_module.py', '.'),
    ('key_manager.py', '.'),
    ('license_manager.py', '.'),
    ('graph_abduction.py', '.'),
    ('config.py', '.'),
    ('eidos_safety_module.py', '.'),
    # [Phase D-2 2026-05-28] AIRA 캐릭터 표정 PNG 자산 (expr_*.png).
    # onedir 빌드 시 dist/EIDOS_AI_Agent/assets/ 로 unpack.
    # 코드 (AIRA_ASSETS_DIR) 가 exe 옆 assets/ 우선 검색 → _MEIPASS fallback.
    ('assets', 'assets'),
]

# [2026-06-11] Gmail/Calendar OAuth credentials + 토큰 자동 포함(사용자 요청·본인 전용 툴).
#   ⚠ 클라이언트 시크릿·계정 토큰이라 EXE 공유 시 계정 노출(본인 전용 전제에서만 번들).
def _bundle_eidos_file(_name):
    for _p in (os.path.join(EIDOS_ROOT, 'eidos_files', _name),
               os.path.join(os.path.dirname(EIDOS_ROOT), 'eidos_files', _name)):
        if os.path.isfile(_p):
            my_datas.append((_p, 'eidos_files'))
            print(f"[spec] eidos_files/{_name} 번들 포함: {_p}")
            return True
    print(f"[spec] WARN: eidos_files/{_name} 없음 — 런타임에 직접 배치/생성 필요")
    return False

for _f in ('credentials.json', 'gmail_token.json', 'gcal_token.json'):
    _bundle_eidos_file(_f)

# 3. 메타데이터 복사
try:
    my_datas += copy_metadata('imageio')
    my_datas += copy_metadata('imageio_ffmpeg')
    my_datas += copy_metadata('regex')
    my_datas += copy_metadata('packaging')
    my_datas += copy_metadata('filelock')
    my_datas += copy_metadata('numpy')
    my_datas += copy_metadata('tokenizers')
    my_datas += copy_metadata('huggingface_hub')
    my_datas += copy_metadata('safetensors')
    my_datas += copy_metadata('pyyaml')
except Exception as e:
    print(f"Warning during metadata copy: {e}")

a = Analysis(
    ['eidos_chat_gui.py'],
    pathex=[os.getcwd()],
    binaries=[],
    datas=my_datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # PyQt5 / PyQt6 완전 제외 → PySide6만 번들링
        'PyQt5',
        'PyQt5.QtCore',
        'PyQt5.QtGui',
        'PyQt5.QtWidgets',
        'PyQt5.QtNetwork',
        'PyQt5.QtWebEngine',
        'PyQt5.QtWebEngineWidgets',
        'PyQt5.sip',
        'PyQt6',
        'PyQt6.QtCore',
        'PyQt6.QtGui',
        'PyQt6.QtWidgets',
        'PyQt6.sip',
        # qtpy도 제외 (PySide6 직접 사용)
        'qtpy',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=False,
    name='EIDOS_AI_Agent',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False, # GUI용 배포 모드
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='character.ico',
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='EIDOS_AI_Agent',
)