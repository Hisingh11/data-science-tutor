import unittest

from utils.calculation_engine import _fd_projection, _fd_total_gain, _fd_total_tax


class FixedDepositProjectionTests(unittest.TestCase):
    def test_projection_reinvests_after_tax_closing_balance(self):
        taxes, balances = _fd_projection(0, 100_000, 0.0625, 0.3, 10)

        self.assertEqual(len(taxes), 10)
        self.assertEqual(len(balances), 10)
        self.assertTrue(all(tax > 0 for tax in taxes))
        self.assertTrue(all(current > previous for current, previous in zip(
            balances, [0, *balances[:-1]]
        )))
        self.assertAlmostEqual(_fd_total_tax(0, 100_000, 0.0625, 0.3, 10), sum(taxes))
        self.assertAlmostEqual(
            _fd_total_gain(0, 100_000, 0.0625, 0.3, 10),
            balances[-1] - 12_000_000,
        )

    def test_projection_rejects_invalid_inputs(self):
        with self.assertRaisesRegex(ValueError, "cannot be negative"):
            _fd_projection(0, -100, 0.05, 0.2, 1)
        with self.assertRaisesRegex(ValueError, "hypothetical tax rate"):
            _fd_projection(0, 100, 0.05, 1.2, 1)
        with self.assertRaisesRegex(ValueError, "at least one year"):
            _fd_projection(0, 100, 0.05, 0.2, 0)


if __name__ == "__main__":
    unittest.main()
