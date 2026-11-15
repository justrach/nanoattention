import unittest

from tune_decode import summarize


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
