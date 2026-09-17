"""`yue spec --json` is the contract other front-ends (snd yue) generate from."""
import json

from yuecli import cli
from yuecli.verbs import VERBS


def test_spec_lists_every_verb_and_field(capsys):
    assert cli.main(["spec"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert set(data["verbs"]) == set(VERBS)
    render = data["verbs"]["render"]
    assert render["remote"] is True
    bars = next(f for f in render["fields"] if f["name"] == "bars")
    assert bars["switch"] == "--bars" and bars["kind"] == "str"
    cfg = next(f for f in data["verbs"]["generate"]["fields"] if f["name"] == "cfg")
    assert cfg["maximum"] == 20 and cfg["short"] is None
    style = next(f for f in data["verbs"]["generate"]["fields"] if f["name"] == "style")
    assert style["short"] == "-s"
    assert data["verbs"]["status"]["remote"] is False


def test_spec_hash_is_stable_and_tracks_the_surface(capsys, monkeypatch):
    first = cli.spec()["hash"]
    assert cli.spec()["hash"] == first
    from yuecli.args import Field, Verb
    monkeypatch.setitem(VERBS, "zzz", Verb("zzz", "x", (Field("a", "str", "b"),)))
    assert cli.spec()["hash"] != first


def test_yue_remote_env_makes_runpod_the_default_and_local_overrides(tmp_path, monkeypatch):
    from yuecli.remote import client
    called = []
    monkeypatch.setattr(client, "run_remote_with_hooks", lambda *a, **k: called.append(a[0].name) or 0)
    monkeypatch.setenv("YUE_REMOTE", "runpod")
    assert cli.main(["status", "--workspace", str(tmp_path)]) in (0, 1)  # status is local-only: never routed
    assert called == []
    assert cli.main(["decode", "--workspace", str(tmp_path / "ws")]) == 0
    assert called == ["decode"]
    # --remote local wins over the env default (and then fails locally: nothing to decode)
    assert cli.main(["decode", "--workspace", str(tmp_path / "ws"), "--remote", "local"]) == 1
    assert called == ["decode"]
