"""Every steer config must build. The class split moved ablation-only kwargs off `COBRAS`,
so a config left pointing at the wrong `type` fails here rather than an hour into a sweep.
"""

from pathlib import Path

import pytest
import yaml

from cobras.steer import get_steer_model


CONFS = sorted((Path(__file__).resolve().parents[1] / "confs" / "steer").glob("*.yaml"))
assert CONFS, "no steer configs found"


@pytest.mark.parametrize("conf", CONFS, ids=lambda p: p.stem)
def test_steer_config_instantiates(conf):
    cfg = yaml.safe_load(conf.read_text())
    model = get_steer_model(cfg["type"], **(cfg.get("kwargs") or {}))
    assert model is not None or cfg["type"] == "NoSteer"
