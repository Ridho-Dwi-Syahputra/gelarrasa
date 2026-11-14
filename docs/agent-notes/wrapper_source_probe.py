"""Local non-VLM image-only wrapper feature probe on the 34 allowed train anchors.

The downloaded weights are a frozen feature source. No competition test image,
review table, or archive label is read. This file does not publish categories.
"""
import hashlib
import json
import os
from pathlib import Path

os.environ['KERAS_BACKEND'] = 'torch'
import keras
import numpy as np
import pandas as pd
from keras.applications.resnet import preprocess_input
from PIL import Image, ImageOps
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
NOTES = ROOT / 'docs/agent-notes'
MODEL = NOTES / '.runtime/wrapper_model/trashclassify_13.keras'
SOURCE_NAMES = ['apples', 'bananas', 'bottles', 'cans', 'cardboard', 'cups',
                'eggshells', 'generalcompost', 'mixers', 'peels',
                'plasticbags', 'plastics', 'tissues']


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def image(path):
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert('RGB').resize((224, 224), Image.Resampling.BILINEAR)
        return np.asarray(im, dtype=np.float32)


def main():
    plan = {
        'method': 'local_pretrained_nonVLM_wrapper_feature_probe',
        'source_repo': 'dvk65/trash-classifier-resnet50',
        'revision': '7a6e53cab51b9a8ba6677eb3c83ad84808e11a68',
        'weight_sha256': digest(MODEL),
        'source_class_order': SOURCE_NAMES,
        'predeclared_plastic_source_score': 'plasticbags+plastics',
        'screen': 'roc_auc_on34allowed_train_sources>=0.80 before full-train extraction',
        'restriction': 'weights only; no external images, test, review or archive; source prediction never copied to final CSV',
        'evaluation_limit': '34 repeatedly reused anchors; diagnostic source weights may overlap unknown external data'
    }
    path = NOTES / 'json/wrapper_source_probe_plan.json'
    path.write_text(json.dumps(plan, indent=2) + '\n', encoding='utf-8')
    source = pd.read_csv(ROOT / 'data/processed/manifest.csv')
    source = source[(source.split == 'train') & (~source.is_crop)].set_index('parent_key')
    seeds = pd.read_csv(ROOT / 'data/processed/manual_seeds.csv')
    assert len(seeds) == 34 and seeds.assigned_class.value_counts().le(5).all()
    model = keras.models.load_model(str(MODEL), compile=False, safe_mode=True,
                                    custom_objects={'preprocess_input': preprocess_input})
    embedder = keras.Model(model.inputs, [model.get_layer('dense_6').output, model.output])
    records = []
    embeddings = []
    for start in range(0, len(seeds), 8):
        batch = seeds.iloc[start:start+8]
        x = np.stack([image(ROOT / source.loc[row.key, 'source_path']) for row in batch.itertuples()])
        feat, probs = embedder(x, training=False)
        if start == 0:
            print('output types', type(feat).__name__, type(probs).__name__,
                  [type(z).__name__ for z in feat] if isinstance(feat, list) else '', flush=True)
        def array(value):
            while isinstance(value, (list, tuple)):
                value = value[0]
            return np.asarray(keras.ops.convert_to_numpy(value))
        for row, f, p in zip(batch.itertuples(), array(feat), array(probs)):
            embeddings.append(np.asarray(f, dtype=np.float32))
            assert digest(ROOT / source.loc[row.key, 'source_path']) == source.loc[row.key, 'source_sha256']
            records.append({'key': row.key, 'assigned_class': row.assigned_class,
                            'wrapper_score': float(p[10] + p[11]), 'bag_score': float(p[10]),
                            'top_source_class': SOURCE_NAMES[int(np.argmax(p))],
                            'source_top_score': float(np.max(p))})
    table = pd.DataFrame(records)
    table.to_csv(NOTES / 'table-csv/wrapper_source_probe_34.csv', index=False)
    np.savez_compressed(NOTES / '.runtime/wrapper_source_probe_34.npz',
                        keys=table.key.to_numpy(str), embeddings=np.stack(embeddings))
    y = table.assigned_class.eq('plastic').to_numpy()
    auc = roc_auc_score(y, table.wrapper_score)
    diagnostic = {'status': 'screen_passed' if auc >= .80 else 'screen_failed',
                  'source_plastic_auc_on_34': float(auc), 'positive_sources': int(y.sum()),
                  'weight_sha256': plan['weight_sha256'], 'train_sources_only': True,
                  'rows': len(table)}
    (NOTES / 'json/wrapper_source_probe_result.json').write_text(json.dumps(diagnostic, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(diagnostic, indent=2), flush=True)
    print(table[['key', 'assigned_class', 'wrapper_score', 'top_source_class']].to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
