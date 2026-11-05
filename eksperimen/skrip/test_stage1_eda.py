import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

spec = importlib.util.spec_from_file_location('stage1_eda', Path(__file__).with_name('stage1_eda.py'))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class ImageContractTests(unittest.TestCase):
    def test_alpha_composites_white(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'a.png'
            a = np.array([[[255, 0, 0, 0], [255, 0, 0, 255]]], dtype=np.uint8)
            Image.fromarray(a, 'RGBA').save(p)
            result = np.asarray(m.canonical_rgb(p))
            np.testing.assert_array_equal(result, [[[255, 255, 255], [255, 0, 0]]])

    def test_exif_rotation_applied_once(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'orient.png'
            im = Image.new('RGB', (30, 10), (20, 80, 140))
            exif = Image.Exif()
            exif[274] = 6
            im.save(p, exif=exif)
            result = m.canonical_rgb(p)
            self.assertEqual(result.size, (10, 30))
            q = Path(d) / 'normalized.png'
            result.save(q)
            self.assertEqual(m.canonical_rgb(q).size, (10, 30))

    def test_letterbox_retains_full_frame(self):
        im = Image.new('RGB', (400, 100), 'red')
        result, geo = m.letterbox(im, 224)
        self.assertEqual(result.size, (224, 224))
        self.assertEqual((geo['resized_width'], geo['resized_height']), (224, 56))
        self.assertEqual(geo['padding_fraction'], .75)
        self.assertEqual(result.getpixel((0, 84)), (255, 0, 0))
        self.assertEqual(result.getpixel((223, 139)), (255, 0, 0))

    def test_hamming_bit_not_hex_character_distance(self):
        arr = m.hash_array(['0000000000000000', '000000000000000f', 'ffffffffffffffff'])
        distances = m.hamming_matrix(arr, arr)
        np.testing.assert_array_equal(distances, [[0, 4, 64], [4, 0, 60], [64, 60, 0]])

    def test_pixel_hash_includes_geometry(self):
        self.assertNotEqual(m.pixel_digest(Image.new('RGB', (4, 2), 'red')),
                            m.pixel_digest(Image.new('RGB', (2, 4), 'red')))

    def test_union_chain_representative(self):
        uf = m.UnionFind(5)
        uf.union(4, 3)
        uf.union(3, 1)
        self.assertEqual(uf.find(4), uf.find(1))
        self.assertNotEqual(uf.find(4), uf.find(0))


if __name__ == '__main__':
    unittest.main()
