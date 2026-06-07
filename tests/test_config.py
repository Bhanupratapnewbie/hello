from command_center.config import Settings


def _settings(**kw):
    base = dict(
        telegram_bot_token="t",
        telegram_admin_chat_id=42,
        model_api_key="k",
        workdir="./workspace",
    )
    base.update(kw)
    return Settings(**base)


def test_role_model_mapping():
    s = _settings(
        model_planner="p",
        model_reasoner="r",
        model_coder="c",
        model_fixer="f",
    )
    assert s.role_model("planner") == "p"
    assert s.role_model("reasoner") == "r"
    assert s.role_model("coder") == "c"
    assert s.role_model("fixer") == "f"


def test_role_model_unknown():
    s = _settings()
    try:
        s.role_model("nope")
    except KeyError:
        return
    raise AssertionError("expected KeyError")


def test_effective_api_key_fallback():
    s = _settings(model_api_key="", openrouter_api_key="or-key")
    assert s.effective_api_key == "or-key"


def test_effective_api_key_priority():
    s = _settings(model_api_key="primary", openrouter_api_key="fallback")
    assert s.effective_api_key == "primary"


def test_validate_runtime_reports_missing():
    s = Settings(telegram_bot_token="", telegram_admin_chat_id=0, model_api_key="")
    problems = s.validate_runtime()
    assert any("TELEGRAM_BOT_TOKEN" in p for p in problems)
    assert any("TELEGRAM_ADMIN_CHAT_ID" in p for p in problems)
    assert any("API_KEY" in p for p in problems)


def test_validate_runtime_clean():
    assert _settings().validate_runtime() == []
