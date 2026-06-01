from ui import auth_session


class FakeStreamlit:
    def __init__(self):
        self.secrets = {"STAFF_PASSWORD": "staff-pass"}
        self.errors = []
        self.warnings = []
        self.session_state = {}

    def error(self, message):
        self.errors.append(message)

    def warning(self, message):
        self.warnings.append(message)


def test_login_staff_shows_database_error_when_profile_lookup_fails(monkeypatch):
    fake_st = FakeStreamlit()
    monkeypatch.setattr(auth_session, "st", fake_st)
    monkeypatch.setattr(auth_session, "get_perfil_staff", lambda _email: None)
    monkeypatch.setattr(auth_session, "get_last_db_error", lambda: "DB config error", raising=False)

    logged_in = auth_session.login_staff("staff@example.com", "staff-pass")

    assert logged_in is False
    assert fake_st.errors == ["DB config error"]
