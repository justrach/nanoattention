import unittest

from tune_decode import profile_metrics, raw_workload, summarize


class RawWorkload(unittest.TestCase):
    def test_deterministic_valid_ids_and_changed_seed(self):
        first = raw_workload(150144, 512, 256, 42)
        self.assertEqual(first, raw_workload(150144, 512, 256, 42))
        self.assertNotEqual(first, raw_workload(150144, 512, 256, 43))
        self.assertEqual(tuple(map(len, first)), (512, 256))
        self.assertTrue(all(0 <= token < 150144 for seq in first for token in seq))
        self.assertGreater(len(set(first[0])), 500)

    def test_decode_stream_independent_of_prefix_length(self):
        short = raw_workload(150144, 512, 256, 42)
        long = raw_workload(150144, 1024, 256, 42)
        self.assertEqual(short[0], long[0][:512])
        self.assertEqual(short[1], long[1])
        self.assertNotEqual(short[0][:256], short[1])

    def test_profiler_excludes_arbitrary_stderr(self):
        self.assertIsNone(profile_metrics('private diagnostic\n{"kernels": []}\n'))
        report = '{"type":"flint_kernel_profile","benchmark":false,"kernels":[]}'
        self.assertEqual(profile_metrics('not json\n' + report)['kernels'], [])

    def test_invalid_sizes(self):
        for values in ((0, 1, 1), (2**32, 1, 1), (4, 0, 1), (4, 1, 0)):
            with self.assertRaises(ValueError):
                raw_workload(*values, seed=42)


class DecodeSweepReports(unittest.TestCase):
    def rows(self):
        rows = [dict(phase='validation', candidate=i, positions=2, vocabulary=16,
                     logits_bit_exact=True) for i in range(2)]
        for repeat in range(3):
            for candidate in ((repeat % 2), ((repeat + 1) % 2)):
                rows.append(dict(phase='timing', candidate=candidate, repeat=repeat,
                                 predictions=[3, 5], decode_seconds=2.0-candidate/2,
                                 decode_gpu_seconds=1.0-candidate/4))
        return rows

    def test_rotated_order_and_paired_results(self):
        result = summarize(self.rows(), [{}, {}], 3, 2)
        self.assertAlmostEqual(result[1]['wall_speedup_over_baseline'], 4/3)
        self.assertAlmostEqual(result[1]['gpu_speedup_over_baseline'], 4/3)

    def test_rejected_candidate_cannot_be_timed(self):
        rows = self.rows()
        rows[1]['logits_bit_exact'] = False
        with self.assertRaisesRegex(ValueError, 'rejected candidate was timed'):
            summarize(rows, [{}, {}], 3, 2)
        rows = [r for r in rows if r.get('phase') != 'timing' or r['candidate'] == 0]
        self.assertFalse(summarize(rows, [{}, {}], 3, 2)[1]['admitted'])

    def test_predictions_and_missing_repeats_are_rejected(self):
        rows = self.rows()
        rows[-1]['predictions'] = [3, 6]
        with self.assertRaisesRegex(ValueError, 'predictions changed'):
            summarize(rows, [{}, {}], 3, 2)
        with self.assertRaisesRegex(ValueError, 'repeat count mismatch'):
            summarize(self.rows()[:-1], [{}, {}], 3, 2)


if __name__ == '__main__':
    unittest.main()
