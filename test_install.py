"""Tests for the one-command install.

The end-to-end run is a verification gate, not a unit test - these pin the
decisions: which torch wheel, and that a second run costs nothing.
"""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


I = _load("install")


def test_it_installs_the_wheel_bootstrap_chose(monkeypatch):
    calls = []
    monkeypatch.setattr(I, "pip_install", lambda *a: calls.append(a))
    monkeypatch.setattr(I, "_dependencies_installed", lambda: False)
    monkeypatch.setattr(I.bootstrap, "torch_install_args",
                        lambda: ["torch", "--index-url", "CU"])
    monkeypatch.setattr(I.bootstrap, "fetch_all", lambda log=None: None)
    assert I.main() == 0
    assert calls[0] == ("torch", "--index-url", "CU")


def test_it_installs_the_runtime_dependencies(monkeypatch):
    calls = []
    monkeypatch.setattr(I, "pip_install", lambda *a: calls.append(a))
    monkeypatch.setattr(I, "_dependencies_installed", lambda: False)
    monkeypatch.setattr(I.bootstrap, "torch_install_args", lambda: ["torch"])
    monkeypatch.setattr(I.bootstrap, "fetch_all", lambda log=None: None)
    I.main()
    rest = calls[1]
    for pkg in ("safetensors", "huggingface_hub", "sentencepiece", "numpy"):
        assert pkg in rest


def test_a_second_run_fetches_nothing(monkeypatch):
    """Idempotence is bootstrap's, not install's - this pins that install
    delegates rather than reimplementing the skip logic."""
    monkeypatch.setattr(I, "pip_install", lambda *a: None)
    monkeypatch.setattr(I, "_dependencies_installed", lambda: False)
    monkeypatch.setattr(I.bootstrap, "torch_install_args", lambda: ["torch"])
    fetched = []
    monkeypatch.setattr(I.bootstrap, "fetch_all",
                        lambda log=None: fetched.append(1))
    I.main()
    I.main()
    assert len(fetched) == 2      # called twice; bootstrap decides to no-op


def test_it_skips_pip_if_dependencies_installed(monkeypatch):
    """If all dependencies are already installed, pip_install is not called."""
    calls = []
    monkeypatch.setattr(I, "pip_install", lambda *a: calls.append(a))
    monkeypatch.setattr(I.bootstrap, "torch_install_args", lambda: ["torch"])
    monkeypatch.setattr(I.bootstrap, "fetch_all", lambda log=None: None)

    # Mock find_spec to return truthy for all required packages
    def mock_find_spec(name):
        if name in ("torch", "safetensors", "huggingface_hub", "sentencepiece", "numpy"):
            return object()  # Mock spec object
        return None

    monkeypatch.setattr(importlib.util, "find_spec", mock_find_spec)

    assert I.main() == 0
    assert len(calls) == 0  # pip_install should not be called
