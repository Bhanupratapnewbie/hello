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


def test_role_endpoint_falls_back_to_global():
    s = _settings(model_base_url="http://gpu:11434/v1", model_api_key="g")
    assert s.role_endpoint("planner") == ("http://gpu:11434/v1", "g")
    assert s.role_endpoint("reasoner") == ("http://gpu:11434/v1", "g")


def test_role_endpoint_per_role_override():
    s = _settings(
        model_base_url="http://gpu:11434/v1",
        model_api_key="g",
        model_reasoner_base_url="https://openrouter.ai/api/v1",
        model_reasoner_api_key="or",
    )
    # Overridden role uses its own endpoint + key.
    assert s.role_endpoint("reasoner") == ("https://openrouter.ai/api/v1", "or")
    # Non-overridden role still falls back to the global endpoint.
    assert s.role_endpoint("planner") == ("http://gpu:11434/v1", "g")


def test_role_endpoint_unknown():
    s = _settings()
    try:
        s.role_endpoint("nope")
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
    assert any("API_KEY" in p for p in problems)


def test_validate_runtime_admin_id_optional():
    # Admin id 0 is fine: the bot auto-claims the first /start sender.
    s = Settings(telegram_bot_token="t", telegram_admin_chat_id=0, model_api_key="k")
    assert s.validate_runtime() == []


def test_validate_runtime_clean():
    assert _settings().validate_runtime() == []
