# EIDOS
EIDOS는 다중 AI 엔진(Helix, Polaris, Petalbot)을
통합 관리하는 개인용 AI Agent Operating System입니다.

단순 챗봇이 아니라

- 목표 생성
- 계획 수립
- 의사결정
- 이메일 처리
- 음성 통화 대응
- 컴퓨터 자동조작

을 하나의 시스템에서 수행하도록 설계되었습니다.

개인용 자율 AI 에이전트 / 음성 비서. Gemini LLM 기반으로 데스크톱에서 동작하며,
음성 통화 대행 · 메일 분류/자동 초안 · 일정 관리 · 컴퓨터 자율 조작 · 장기 목표 수렴 실행을
하나의 GUI 위에서 통합한다.

> ⚠️ 개인 프로젝트입니다. 실행에는 본인의 Gemini / Google OAuth / (선택) NCP·SerpAPI·Telegram 키가 필요하며,
> 키와 런타임 데이터(`eidos_files/`)는 저장소에 포함되지 않습니다.

## 주요 기능

- **AIRA 음성 비서**
  - 통화 대행: 음향 결합(acoustic coupling) 방식으로 걸려온 전화를 대신 받고, 반이중(half-duplex)으로 응대 후 요약 보고
  - STT/TTS: NCP CLOVA Speech 연동, 턴 분절은 RMS 기반 VAD 상태기계로 직접 구현
  - 메일: 받은 메일을 중요/답장필요 등으로 분류 → 답장 초안 자동 생성(임시보관함 저장, 발송은 사람 승인) → 일정 자동 추출/등록
  - 일정 알림: 음성 완료 처리, 연속 일정 브리핑, 스누즈
- **발명/계획 엔진** (서브프로세스로 격리 실행)
  - Helix — 발산(아이디어 생성·선별)
  - Polaris — 수렴(현실검증하며 단일 비전으로 좁혀 장기계획 수립·실행)
  - autodrive — 화면을 보고 컴퓨터를 조작하는 폐루프(감각-운동) 실행
- **무인 실행**: Windows 작업 스케줄러 등록으로 매일 정해진 시간만 자율 동작(비용 캡)

## 아키텍처 메모

- **순수 로직과 부작용 분리**: `eidos_aira_*` 모듈은 마이크/Qt/TTS 같은 부작용을 GUI로 밀어내고
  순수 함수만 남겨 단위 테스트가 가능하도록 설계. (예: 통화 흐름·메일 분류를 가짜 LLM 주입으로 테스트)
- **서브프로세스 격리**: Helix/Polaris/autodrive는 `*_launcher.py`가 subprocess로만 호출해
  본체 의존성/인코딩 함정으로부터 엔진의 독립성을 보존.
- **graceful degradation**: numpy·sounddevice 등 선택 의존성이 없으면 순수 파이썬 폴백으로 동작.

## 기술 스택

- Python 3.x, PyQt(GUI)
- Google Gemini API, Google Gmail/Calendar API (OAuth)
- NCP CLOVA Speech (STT/TTS), (선택) SerpAPI, Telegram Bot API
- PyInstaller (배포 빌드)

## 설치 / 실행

```bash
pip install -r requirements.txt   # TODO: requirements.txt 추가
# eidos_files/credentials.json 에 Google OAuth 클라이언트 배치 후
python gmail_oauth_test.py        # 최초 1회 OAuth 동의
python eidos_chat_gui.py          # 본체 실행  (TODO: 실제 진입점 확인)
```

## 프로젝트 구조 (발췌)

| 파일 | 역할 |
|------|------|
| `eidos_chat_gui.py` | 메인 GUI / 오케스트레이션 |
| `eidos_v4_0_core.py` | 코어(LLM·도구·툴 실행) |
| `eidos_aira_call.py` | 통화 대화 로직(순수) |
| `eidos_aira_mail.py` | 메일 분류/초안/일정추출(순수) |
| `eidos_aira_schedule.py` | 일정 알림 로직(순수) |
| `eidos_aira_audio.py` / `eidos_aira_stt.py` | 마이크 캡처·VAD / STT |
| `helix_launcher.py` / `polaris_launcher.py` / `autodrive_launcher.py` | 서브엔진 런처 |

## 라이선스

TODO (공개 범위 결정 후 명시)
