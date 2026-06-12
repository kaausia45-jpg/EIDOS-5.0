# -*- coding: utf-8 -*-
"""
Gmail 연동 라이브 검증 스크립트 (1회 OAuth 동의 + 최근 메일 3개 출력).
Core의 _get_gmail_service / gmail_list_messages 와 동일한 로직·경로·스코프를 사용.

실행:  ! python gmail_oauth_test.py
  - 처음엔 브라우저가 열려 Google 로그인·권한 동의를 요청합니다.
  - 동의하면 eidos_files/gmail_token.json 이 생성되고, 이후 EIDOS GUI는 재동의 없이 동작합니다.
"""
import os
import sys

CRED  = "eidos_files/credentials.json"
TOKEN = "eidos_files/gmail_token.json"
SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]


def main() -> int:
    if not os.path.exists(CRED):
        print(f"[X] credentials.json 없음: {CRED}")
        return 1
    try:
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from google.auth.transport.requests import Request
        from googleapiclient.discovery import build
    except ImportError as e:
        print(f"[X] 패키지 누락: {e}\n    pip install google-api-python-client google-auth-oauthlib")
        return 1

    creds = None
    if os.path.exists(TOKEN):
        creds = Credentials.from_authorized_user_file(TOKEN, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            print("[*] 토큰 갱신 중...")
            creds.refresh(Request())
        else:
            print("[*] 브라우저에서 Google 로그인·권한 동의를 진행하세요...")
            flow = InstalledAppFlow.from_client_secrets_file(CRED, SCOPES)
            creds = flow.run_local_server(port=0)
        os.makedirs(os.path.dirname(TOKEN) or ".", exist_ok=True)
        with open(TOKEN, "w") as f:
            f.write(creds.to_json())
        print(f"[OK] 토큰 저장: {TOKEN}")

    service = build("gmail", "v1", credentials=creds)

    # 프로필 + 최근 메일 3개
    prof = service.users().getProfile(userId="me").execute()
    print(f"[OK] 인증 성공 - 계정: {prof.get('emailAddress')} "
          f"(총 메일 {prof.get('messagesTotal')}개)")

    resp = service.users().messages().list(userId="me", maxResults=3).execute()
    refs = resp.get("messages", []) or []
    print(f"[OK] 최근 메일 {len(refs)}개:")
    for ref in refs:
        msg = service.users().messages().get(
            userId="me", id=ref["id"], format="metadata",
            metadataHeaders=["From", "Subject"]).execute()
        hdrs = {h["name"].lower(): h["value"] for h in
                (msg.get("payload", {}) or {}).get("headers", [])}
        unread = "UNREAD" in (msg.get("labelIds", []) or [])
        mark = "[안읽음] " if unread else ""
        print(f"   - {mark}{hdrs.get('subject', '(제목 없음)')}  "
              f"<- {hdrs.get('from', '?')}")
    print("\n[DONE] Gmail 연동 검증 완료. 이제 EIDOS 채팅에서 메일 명령을 쓸 수 있습니다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
