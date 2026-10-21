import hashlib
import json

import numpy as np
import pytest

from scripts.research_dataset_pipeline import load_resume_state, single_pair_training_config


def test_single_pair_training_reads_scalar_cache(tmp_path):
    from training.gpu_cache_io import _zarr_create, _zarr_open_group
    from training.gpu_datasets import ZarrStreamDataset

    cfg = single_pair_training_config({'data': {'pairs': ['EURUSD']},
                                      'training': {'per_pair_heads': True}})
    path = str(tmp_path/'cache.zarr')
    z = _zarr_open_group(path, mode='w')
    for name, values in {'X': np.ones((8, 4, 3), dtype=np.float32),
                         'y': np.arange(8, dtype=np.float32),
                         'y_cls': np.ones(8, dtype=np.float32)}.items():
        _zarr_create(z, name, shape=values.shape, chunks=values.shape, dtype=values.dtype)
        z[name][:] = values
    ds = ZarrStreamDataset(path, np.arange(8), shuffle_chunks=False,
                           pair_targets=cfg['training']['per_pair_heads'])
    rows = list(ds)
    assert len(rows) == 8
    assert sorted(float(row[1]) for row in rows) == list(range(8))


def test_multi_pair_config_rejected():
    with pytest.raises(ValueError, match='exactly one pair'):
        single_pair_training_config({'data': {'pairs': ['EURUSD', 'USDJPY']}, 'training': {}})


@pytest.mark.parametrize('resume', [True, False])
def test_restart_uses_explicit_native_training_resume(resume):
    cfg = single_pair_training_config({'data': {'pairs': ['EURUSD']},
                                      'training': {'resume': not resume}}, resume=resume)
    assert cfg['training']['resume'] is resume
    assert cfg['training']['per_pair_heads'] is False


@pytest.mark.parametrize('change', ['none', 'source', 'reference', 'running'])
def test_resume_validates_source_and_reference(tmp_path, change):
    ref = tmp_path/'reference'/'EURUSD'
    ref.mkdir(parents=True)
    (ref/'haelt_best.pt').write_bytes(b'original')
    state = {'status': 'failed', 'source_fingerprint': 'original-source',
             'completed': [], 'reference_artifacts': {'EURUSD': {
                 'haelt_best.pt': hashlib.sha256(b'original').hexdigest()}}}
    if change == 'reference':
        (ref/'haelt_best.pt').write_bytes(b'changed')
    if change == 'running':
        state['status'] = 'running'
    (tmp_path/'status.json').write_text(json.dumps(state))
    if change == 'none':
        resumed = load_resume_state(tmp_path, 'original-source', ['EURUSD'])
        assert resumed['status'] == 'running'
        assert resumed['completed'] == []
        assert list(tmp_path.glob('status_before_resume_*.json'))
    else:
        with pytest.raises(RuntimeError):
            load_resume_state(tmp_path, 'changed-source' if change == 'source' else 'original-source', ['EURUSD'])
