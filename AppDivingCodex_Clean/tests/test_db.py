from data import db
import base64
import json


def _fake_jwt(role: str) -> str:
    def encode(payload: dict) -> str:
        raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")

    return f"{encode({'alg': 'HS256', 'typ': 'JWT'})}.{encode({'role': role})}.signature"


def test_resolve_supabase_credentials_prefers_service_role_key(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_KEY", "anon-key")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "service-key")

    url, key = db._resolve_supabase_credentials()

    assert url == "https://example.supabase.co"
    assert key == "service-key"


def test_format_db_error_explains_is_staff_or_admin_permission():
    error = {
        "code": "42501",
        "message": "permission denied for function is_staff_or_admin",
    }

    message = db._format_db_error(error)

    assert "is_staff_or_admin" in message
    assert "SUPABASE_SERVICE_ROLE_KEY" in message
    assert "grant execute" in message.lower()


def test_format_db_error_explains_users_table_permission():
    error = {
        "code": "42501",
        "message": "permission denied for table users",
    }

    message = db._format_db_error(error)

    assert "public.users" in message
    assert "SUPABASE_SERVICE_ROLE_KEY" in message
    assert "service_role" in message


def test_get_client_rejects_service_role_secret_with_anon_jwt(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", _fake_jwt("anon"))
    monkeypatch.delenv("SUPABASE_SECRET_KEY", raising=False)
    monkeypatch.delenv("SUPABASE_KEY", raising=False)
    monkeypatch.setattr(
        db,
        "create_client",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("should not create client")),
    )
    db.clear_last_db_error()

    client = db.get_client()

    assert client is None
    assert "role anon" in db.get_last_db_error()
