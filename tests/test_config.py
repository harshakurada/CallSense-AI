from configs.settings import get_audio_config, get_config, get_diarization_config, get_settings


def test_settings_load_from_env():
    settings = get_settings()
    assert settings.api_port > 0
    assert settings.database_url.startswith("postgresql://")


def test_config_yaml_loads_pipeline_params():
    config = get_config()
    assert "asr" in config and "nlp" in config


def test_config_yaml_has_thresholds():
    config = get_config()
    assert 0.0 < config["thresholds"]["escalation_risk_high"] <= 1.0


def test_audio_config_loads():
    audio_config = get_audio_config()
    assert audio_config["io"]["target_sample_rate"] == 16000
    assert "vad" in audio_config and "chunking" in audio_config


def test_diarization_config_loads():
    config = get_diarization_config()
    assert config["clustering"]["expected_speakers"] == 2
    assert "segmentation" in config and "embedding" in config
