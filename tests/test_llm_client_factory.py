from src.agents.llm_client import LLMClient, create_llm_client


def test_create_llm_client_returns_llmclient(monkeypatch):
    monkeypatch.delenv("ANSPG_LLM_MODEL", raising=False)
    client = create_llm_client(api_key="test", model="gpt-4o-mini")
    assert isinstance(client, LLMClient)
    assert client.api_key == "test"
    assert client.model == "gpt-4o-mini"


def test_create_llm_client_env_model(monkeypatch):
    monkeypatch.setenv("ANSPG_LLM_MODEL", "env-model")
    client = create_llm_client(api_key="k", base_url="u")
    assert client.model == "env-model"
    assert client.api_key == "k"
    assert client.base_url == "u"


def test_create_llm_client_env_defaults(monkeypatch):
    monkeypatch.setenv("ANSPG_LLM_API_KEY", "env-key")
    monkeypatch.setenv("ANSPG_LLM_BASE_URL", "env-url")
    monkeypatch.setenv("ANSPG_LLM_MODEL", "env-model")
    client = create_llm_client()
    assert client.api_key == "env-key"
    assert client.base_url == "env-url"
    assert client.model == "env-model"


def test_create_llm_client_openai_env(monkeypatch):
    monkeypatch.delenv("ANSPG_LLM_API_KEY", raising=False)
    monkeypatch.delenv("ANSPG_LLM_BASE_URL", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "openai-url")
    monkeypatch.setenv("ANSPG_LLM_MODEL", "env-model2")
    client = create_llm_client()
    assert client.api_key == "openai-key"
    assert client.base_url == "openai-url"
    assert client.model == "env-model2"
