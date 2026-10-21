"""Exercise the actual checkpoint writer without running a GPU epoch."""
import ast
import copy
import json
from pathlib import Path


def test_checkpoint_config_save_without_optional_features_report(tmp_path):
    path = Path(__file__).resolve().parents[1]/'training'/'supervised_loop.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    # Isolate the real config-writing statement; supply a small payload instead
    # of constructing the model, optimizer and fold needed by its full payload.
    writer = next(node for node in ast.walk(tree)
                  if isinstance(node, ast.With)
                  and any(isinstance(item.optional_vars, ast.Name)
                          and item.optional_vars.id == '_cfg_fp' for item in node.items))
    writer = copy.deepcopy(writer)
    writer.body[0].value.args[0] = ast.Name(id='payload', ctx=ast.Load())
    module = ast.fix_missing_locations(ast.Module(body=[writer], type_ignores=[]))
    target = tmp_path/'checkpoint_config.json'
    payload = {'fold_id': 0, 'epoch': 0, 'n_features': 146}
    # No local `json` binding from the optional features-report branch exists.
    exec(compile(module, str(path), 'exec'),
         {'_json_hash': json, 'cfg_path': target, 'payload': payload})
    assert json.loads(target.read_text()) == payload
