import hashlib
import struct
import tempfile
import unittest
from pathlib import Path
from rmala.lm100_data import HEADER, INDEX, TokenStream, check_header, select_eval


def header(magic):
    return struct.pack('<8sII', magic, 1, HEADER) + bytes(HEADER-16)


class CommonStreamTests(unittest.TestCase):
    def test_cross_shard_exact_targets_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'shards').mkdir()
            a = struct.pack('<4H', 3, 2, 5, 6)
            b = struct.pack('<4H', 2, 8, 9, 2)
            (root/'shards/a').write_bytes(header(b'NDTKTOK1')+a)
            (root/'shards/b').write_bytes(header(b'NDTKTOK1')+b)
            m = dict(dataset_root=tmp, train_segments=[dict(tokens_file='a', take_tokens=4), dict(tokens_file='b', take_tokens=3)])
            s = TokenStream(m)
            self.assertEqual(s.read(2, 4), (a+b)[4:12])
            self.assertEqual(s.digest(), hashlib.sha256(a+b[:6]).hexdigest())
            # Targets on each resumed minibatch concatenate to exactly one stream.
            self.assertEqual(s.read(0, 4)[2:] + s.read(3, 4)[2:], s.read(1, 6))
            with self.assertRaises(ValueError):
                s.read(6, 2)
            check_header(root/'shards/a', b'NDTKTOK1', 104)
            with self.assertRaises(ValueError):
                check_header(root/'shards/a', b'NDTKIDX1', 104)

    def test_complete_document_bytes_and_eos(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'shards').mkdir()
            (root/'shards/t').write_bytes(header(b'NDTKTOK1')+struct.pack('<5H', 5, 6, 2, 9, 2))
            raw = INDEX.pack(0, 0, 3, 8, 0, 0)+INDEX.pack(1, 3, 2, 3, 0, 0)
            ip = root/'shards/i'
            ip.write_bytes(header(b'NDTKIDX1')+raw)
            r = dict(index_file='i', tokens_file='t', tokens=5, documents=2,
                     text_utf8_bytes=11, index_sha256=hashlib.sha256(ip.read_bytes()).hexdigest())
            result = select_eval(root, [r], 4, 12)
            self.assertEqual(result['tokens_including_eos'], 5)
            self.assertEqual(result['content_tokens'], 3)
            self.assertEqual(result['text_utf8_bytes'], 11)
            self.assertEqual(len(result['documents']), 2)
            # Index byte counts are validated, not estimated from corpus averages.
            r['text_utf8_bytes'] = 12
            with self.assertRaises(ValueError):
                select_eval(root, [r], 4, 12)


if __name__ == '__main__':
    unittest.main()
