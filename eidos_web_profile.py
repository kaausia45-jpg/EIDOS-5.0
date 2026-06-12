# eidos_web_profile.py
# 내장 브라우저 (QWebEngineView) 의 default profile 을 영속 모드로 설정.
#
# 사용자가 네이버 / 구글 / 카카오 채널 / Gmail 등에 1회 로그인하면
# 쿠키 + 로컬 스토리지가 eidos_files/web_profile/ 에 저장되어 영구 유지.
# 다음 부터는 자동 로그인 상태 → 블로그 글쓰기·상품 등록·답변 입력 페이지로
# 곧바로 진입 가능 ([[feedback-automation-take-to-doorstep]] 원칙).
#
# 사용법: QApplication 인스턴스 생성 *직후*, 첫 QWebEngineView 생성 *전*에
#   `setup_persistent_default_profile()` 1회 호출.

from __future__ import annotations

import os

_already_setup = False


def setup_persistent_default_profile(base_dir: str = "eidos_files") -> bool:
    """앱 시작 시 1회 호출 — QWebEngineProfile.defaultProfile() 을 영속 모드로 설정.

    모든 QWebEngineView 가 default profile 을 공유하므로 한 곳만 설정하면
    PopupHandlingPage·새 탭·팝업·iframe 모두 영속 쿠키 / 캐시 혜택을 받음.

    반환: 성공 True / 이미 설정됨 또는 실패 False.
    """
    global _already_setup
    if _already_setup:
        return False

    try:
        from PySide6.QtWebEngineCore import QWebEngineProfile
    except Exception as e:
        print(f"⚠️ [web_profile] QtWebEngine 미설치 (graceful skip): {e}")
        return False

    try:
        storage_dir = os.path.abspath(os.path.join(base_dir, "web_profile"))
        cache_dir = os.path.join(storage_dir, "cache")
        os.makedirs(storage_dir, exist_ok=True)
        os.makedirs(cache_dir, exist_ok=True)

        profile = QWebEngineProfile.defaultProfile()
        if profile is None:
            print("⚠️ [web_profile] defaultProfile() = None — skip")
            return False

        profile.setPersistentStoragePath(storage_dir)
        profile.setCachePath(cache_dir)
        # 영속 쿠키 강제 — 세션 쿠키도 디스크 저장
        try:
            profile.setPersistentCookiesPolicy(
                QWebEngineProfile.PersistentCookiesPolicy.ForcePersistentCookies
            )
        except Exception:
            # PySide6 6.4 이전 호환
            try:
                profile.setPersistentCookiesPolicy(
                    QWebEngineProfile.ForcePersistentCookies
                )
            except Exception:
                pass
        # 디스크 HTTP 캐시
        try:
            profile.setHttpCacheType(QWebEngineProfile.HttpCacheType.DiskHttpCache)
        except Exception:
            try:
                profile.setHttpCacheType(QWebEngineProfile.DiskHttpCache)
            except Exception:
                pass

        # 봇 탐지 회피용 일반 데스크탑 Chrome UA — 일부 사이트 (네이버·구글) 가
        # QtWebEngine default UA 를 비표준으로 인식해 추가 보안 단계 띄울 수 있음.
        # 사용자가 직접 로그인하는 흐름이므로 UA 위장은 합법·일반 통과 위함.
        try:
            profile.setHttpUserAgent(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36"
            )
        except Exception:
            pass

        _already_setup = True
        print(
            f"🌐 [web_profile] default profile 영속화 완료\n"
            f"   • 저장 경로: {storage_dir}\n"
            f"   • 캐시 경로: {cache_dir}\n"
            f"   • 사용자 한 번 로그인 후 영구 유지 (네이버/구글/카카오/Gmail 등)"
        )
        return True
    except Exception as e:
        print(f"⚠️ [web_profile] setup 실패 (graceful skip): {e}")
        return False


def get_web_profile_dir(base_dir: str = "eidos_files") -> str:
    """영속 profile 저장 디렉터리 경로 반환 (조회용)."""
    return os.path.abspath(os.path.join(base_dir, "web_profile"))


def clear_web_profile(base_dir: str = "eidos_files") -> bool:
    """영속 profile 전체 삭제 — 로그인 세션 모두 초기화 (디버그/리셋용).
    주의: 앱 재시작 후 적용. 실행 중에는 파일 잠금으로 일부 실패 가능."""
    import shutil
    storage_dir = get_web_profile_dir(base_dir)
    try:
        if os.path.isdir(storage_dir):
            shutil.rmtree(storage_dir, ignore_errors=True)
        print(f"🧹 [web_profile] cleared: {storage_dir}")
        return True
    except Exception as e:
        print(f"⚠️ [web_profile] clear 실패: {e}")
        return False
