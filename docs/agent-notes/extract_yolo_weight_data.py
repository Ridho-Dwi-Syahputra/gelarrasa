"""Extract a pinned image-only detector as inert numeric tensors, never pickle.load."""
import json
import re
import zipfile
import numpy as np
import torch
from safetensors.torch import save_file
from huggingface_hub import hf_hub_download
from extract_task8_weight_data import symbolic_read, fields, Tensor
from fewshot_train_only_probe import NOTES, sha, dump


def main():
    repo = 'CatSat/yolov11-litter-materials'
    revision = '25a03cf200441cfd9e8e62ded5529e49562bd96b'
    checkpoint = hf_hub_download(repo, 'best.pt', revision=revision)
    with zipfile.ZipFile(checkpoint) as archive:
        assert sum(v.file_size for v in archive.infolist()) < 500 * 1024**2
        pickled = next(k for k in archive.namelist() if k.endswith('/data.pkl'))
        prefix = pickled.rsplit('/', 1)[0] + '/'
        root = symbolic_read(archive.read(pickled))
        model = fields(root).get('ema') or fields(root).get('model')
        attributes = fields(model)
        config, names = attributes['yaml'], attributes['names']
        assert isinstance(config, dict) and isinstance(names, dict) and len(names) == 5
        assert set(names) == set(range(5)) and all(isinstance(v, str) for v in names.values())
        assert config.get('activation') in [None, 'nn.SiLU()', 'nn.ReLU()']
        for entry in config['backbone'] + config['head']:
            assert len(entry) == 4 and isinstance(entry[1], int) and 0 < entry[1] < 20
            assert isinstance(entry[2], str) and re.fullmatch(r'(nn\.)?[A-Za-z][A-Za-z0-9_]*', entry[2])
        def simple(value):
            if value is None or isinstance(value, (str, int, float, bool)):
                return value
            if isinstance(value, (list, tuple)):
                return [simple(v) for v in value]
            if isinstance(value, dict):
                return {k: simple(v) for k, v in value.items()}
            raise TypeError(type(value))
        config = simple(config)
        tensors = {}
        def collect(node, path=''):
            data = fields(node)
            for section in ['_parameters', '_buffers']:
                for key, value in data.get(section, {}).items():
                    if value is not None:
                        assert isinstance(value, Tensor)
                        tensors[path + key] = value
            for key, child in data.get('_modules', {}).items():
                if child is not None:
                    collect(child, path + key + '.')
        collect(model)
        assert len(tensors) > 200
        dtype = {'torch.FloatStorage': '<f4', 'torch.DoubleStorage': '<f8', 'torch.HalfStorage': '<f2',
                 'torch.LongStorage': '<i8', 'torch.IntStorage': '<i4', 'torch.ByteStorage': 'u1', 'torch.BoolStorage': '?'}
        arrays, state = {}, {}
        for key, spec in tensors.items():
            storage = spec.storage
            assert storage.kind in dtype
            if storage.key not in arrays:
                values = np.frombuffer(archive.read(prefix + 'data/' + storage.key), dtype=np.dtype(dtype[storage.kind]))
                assert len(values) == storage.count
                arrays[storage.key] = values
            values = arrays[storage.key]
            assert all(n > 0 for n in spec.shape) and all(s >= 0 for s in spec.stride)
            largest = spec.offset + sum((n - 1) * s for n, s in zip(spec.shape, spec.stride))
            assert 0 <= spec.offset <= largest < len(values)
            array = np.lib.stride_tricks.as_strided(values[spec.offset:], shape=spec.shape,
                       strides=tuple(s * values.dtype.itemsize for s in spec.stride)).copy()
            t = torch.from_numpy(array)
            if t.is_floating_point():
                t = t.float()
                assert torch.isfinite(t).all()
            state[key] = t
    destination = NOTES / '.runtime/litter_yolo_numeric.safetensors'
    save_file(state, str(destination))
    report = {'repository': repo, 'revision': revision, 'architecture': config, 'source_class_names': names,
              'checkpoint_sha256': sha(checkpoint), 'numeric_sha256': sha(destination), 'tensor_count': len(state),
              'serialized_callables_executed': False, 'remote_application_executed': False,
              'external_images_used': False, 'external_inference': False,
              'purpose': 'train-only object localisation and numeric features;not direct final prediction'}
    dump(NOTES / '.runtime/litter_yolo_source.json', report)
    print(json.dumps({k: report[k] for k in ['repository', 'revision', 'source_class_names', 'tensor_count', 'numeric_sha256']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
