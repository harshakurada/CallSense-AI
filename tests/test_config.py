from configs.settings import get_config, get_settings


def test_settings_load_from_env():
    settings = get_settings()
    assert settings.api_port > 0
    assert settings.database_url.startswith("postgresql://")


def test_config_yaml_loads_audio_params():
    config = get_config()
    assert config["audio"]["sample_rate"] == 16000
    assert "asr" in config and "diarization" in config and "nlp" in config


def test_config_yaml_has_thresholds():
    config = get_config()
    assert 0.0 < config["thresholds"]["escalation_risk_high"] <= 1.0
