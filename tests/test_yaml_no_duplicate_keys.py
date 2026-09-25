"""PyYAML silently keeps the last duplicate key; fail instead (e.g. the old
second `direction_training:` block in run.yaml dropped half the settings)."""

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = sorted((ROOT / "config").glob("*.yaml")) + sorted((ROOT / "config" / "models").glob("*.yaml"))


class _StrictLoader(yaml.SafeLoader):
    pass


def _no_duplicates(loader, node, deep=False):
    seen = set()
    for key_node, _ in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in seen:
            raise yaml.constructor.ConstructorError(
                None, None, f"duplicate key {key!r}", key_node.start_mark
            )
        seen.add(key)
    return loader.construct_mapping(node, deep)


_StrictLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _no_duplicates)


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_config_has_no_duplicate_keys(path):
    yaml.load(path.read_text(encoding="utf-8"), Loader=_StrictLoader)
