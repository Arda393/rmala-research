import unittest
from rmala.lm100_bench import spec, oracle, choice_rows, cell_specs, wilson


class BenchmarkTests(unittest.TestCase):
    def test_oracle_and_balance(self):
        for task, condition, length in cell_specs():
            counts = [0]*4
            for i in range(64):
                r = spec(task, condition, i)
                counts[r['options'].index(oracle(r))] += 1
            self.assertEqual(counts, [16]*4)
        a = spec('distractors', 'clean', 7)
        b = spec('distractors', 'similar_keys', 7)
        self.assertEqual(oracle(a), oracle(b))
        r = spec('two_hop', 'two_facts', 3)
        ref = oracle(r)
        alias = next(f for f in r['facts'] if f['kind'] == 'alias')
        alias['value'] = next(f['key'] for f in r['facts'] if f['kind']=='color' and f['value'] != ref.strip())
        self.assertNotEqual(oracle(r), ref)

    def test_answer_shift_mask_and_padding(self):
        x, y = choice_rows(dict(prompt_ids=[2,11,12], choice_ids=[[21,22],[31,32]], length=6))
        self.assertEqual(x[0], [2,11,12,21,2,2])
        self.assertEqual(y[0], [-100,-100,21,22,-100,-100])
        self.assertEqual(x[1][:3], x[0][:3])
        self.assertEqual(y[1], [-100,-100,31,32,-100,-100])

    def test_multiple_retrieval_requires_each_fact(self):
        for i in range(64):
            r = spec('multi_retrieval', 'three_keys', i)
            truth = oracle(r).strip().split(', ')
            foils = [x.strip().split(', ') for x in r['options'] if x != oracle(r)]
            self.assertEqual(len(foils), 3)
            changed = []
            for foil in foils:
                positions = [j for j in range(3) if foil[j] != truth[j]]
                self.assertEqual(len(positions), 1)
                changed.extend(positions)
            self.assertEqual(sorted(changed), [0,1,2])

    def test_state_latest_relevant_record_and_interval(self):
        r = spec('state_tracking', 'three_states', 2)
        self.assertEqual(oracle(r), ' mor')
        # Last unrelated event must not overwrite the queried object's state.
        r['facts'][-1]['value'] = 'mavi'
        self.assertEqual(oracle(r), ' mor')
        lo, hi = wilson(16, 64)
        self.assertLess(lo, .25)
        self.assertGreater(hi, .25)


if __name__ == '__main__':
    unittest.main()
