import math
import unittest

from utils.calculation_engine import (
    SIMULATION_COUNT,
    _plan_validation_issues,
    _simulate_plan,
    is_calculation_request,
    monte_carlo_report,
)


class CalculationEngineTests(unittest.TestCase):
    def test_calculation_intent_detects_numeric_questions_without_misrouting_concepts(self):
        self.assertTrue(is_calculation_request("What is 2 plus 2?"))
        self.assertTrue(is_calculation_request("How much interest on 1000 at 5%?"))
        self.assertFalse(is_calculation_request("What does the mean mean in statistics?"))

    def test_fixed_inputs_produce_reconciled_exact_result(self):
        plan = {
            "assumptions": ["All inputs are fixed values from the question."],
            "inputs": [
                {"name": "principal", "low": 100, "likely": 100, "high": 100,
                 "unit": "currency"},
                {"name": "payment", "low": 50, "likely": 50, "high": 50,
                 "unit": "currency"},
                {"name": "periods", "low": 2, "likely": 2, "high": 2,
                 "unit": "months"},
            ],
            "outputs": [
                {"label": "Final balance", "formula": "principal + payment * periods",
                 "unit": "currency"},
            ],
        }

        report = _simulate_plan(plan, seed=10)

        self.assertIn(f"{SIMULATION_COUNT:,} simulations", report)
        self.assertIn("| Final balance | 200 | 200 | 200 | currency |", report)
        self.assertIn("not confidence intervals", report)

    def test_illustrative_ranges_produce_reproducible_uncertainty_report(self):
        plan = {
            "assumptions": ["x is an illustrative triangular input."],
            "inputs": [{"name": "x", "low": 1, "likely": 2, "high": 3, "unit": "units"}],
            "outputs": [{"label": "Doubled value", "formula": "x * 2", "unit": "units"}],
        }

        first = _simulate_plan(plan, seed=42)
        second = _simulate_plan(plan, seed=42)

        self.assertEqual(first, second)
        self.assertIn("| Doubled value |", first)
        self.assertIn("LLM-proposed", first)

    def test_outputs_can_reference_a_previously_computed_output(self):
        plan = {
            "assumptions": ["Annual rate is fixed from the question."],
            "inputs": [{"name": "annual_rate", "low": 0.12, "likely": 0.12,
                        "high": 0.12, "unit": "fraction"}],
            "outputs": [
                {"label": "Monthly rate before tax", "formula": "annual_rate / 12",
                 "unit": "fraction"},
                {"label": "Annual rate check", "formula": "monthly_rate_before_tax * 12",
                 "unit": "fraction"},
            ],
        }

        report = _simulate_plan(plan, seed=11)

        self.assertIn("| Annual rate check | 0.12 | 0.12 | 0.12 | fraction |", report)

    def test_fd_functions_apply_monthly_deposits_and_annual_after_tax_reinvestment(self):
        plan = {
            "assumptions": ["Tax rate is a hypothetical scenario."],
            "inputs": [
                {"name": "initial_balance", "low": 0, "likely": 0, "high": 0, "unit": "currency"},
                {"name": "monthly_payment", "low": 100, "likely": 100, "high": 100, "unit": "currency"},
                {"name": "annual_rate", "low": 0.12, "likely": 0.12, "high": 0.12, "unit": "fraction"},
                {"name": "tax_rate", "low": 0.1, "likely": 0.1, "high": 0.1, "unit": "fraction"},
            ],
            "outputs": [
                {"label": "Year 1 closing balance after tax",
                 "formula": "fd_closing_balance(initial_balance, monthly_payment, annual_rate, tax_rate, 1)",
                 "unit": "currency"},
                {"label": "Year 1 tax paid",
                 "formula": "fd_annual_tax(initial_balance, monthly_payment, annual_rate, tax_rate, 1)",
                 "unit": "currency"},
            ],
        }

        report = _simulate_plan(plan, seed=12)
        gross_balance = 100 * sum(1.01**month for month in range(12))
        tax_paid = (gross_balance - 1_200) * 0.1
        after_tax_balance = gross_balance - tax_paid
        formatted_tax = f"{tax_paid:,.4f}".rstrip("0").rstrip(".")

        self.assertIn(
            f"| Year 1 closing balance after tax | {after_tax_balance:,.4f} | "
            f"{after_tax_balance:,.4f} | {after_tax_balance:,.4f} | currency |",
            report,
        )
        self.assertIn(
            f"| Year 1 tax paid | {formatted_tax} | {formatted_tax} | "
            f"{formatted_tax} | currency |",
            report,
        )

    def test_arbitrary_code_is_rejected(self):
        plan = {
            "assumptions": [],
            "inputs": [],
            "outputs": [
                {"label": "Unsafe", "formula": "__import__('os').system('whoami')", "unit": ""},
            ],
        }

        with self.assertRaisesRegex(ValueError, "disallowed function call"):
            _simulate_plan(plan)

    def test_invalid_input_ranges_are_rejected(self):
        plan = {
            "assumptions": [],
            "inputs": [
                {"name": "x", "low": 3, "likely": 2, "high": 1, "unit": ""},
            ],
            "outputs": [{"label": "x", "formula": "x", "unit": ""}],
        }

        with self.assertRaisesRegex(ValueError, "low <= likely <= high"):
            _simulate_plan(plan)

    def test_unspecified_tax_requires_a_non_degenerate_hypothetical_range(self):
        plan = {
            "assumptions": ["Tax is estimated."],
            "inputs": [
                {"name": "tax_rate", "low": 0.3, "likely": 0.3, "high": 0.3, "unit": "fraction"},
            ],
            "outputs": [{"label": "Tax", "formula": "tax_rate * 100", "unit": "currency"}],
        }

        self.assertIn(
            "Tax rate is unspecified; provide a hypothetical tax_rate range.",
            _plan_validation_issues(plan, "Estimate returns after Indian tax."),
        )

    def test_llm_plan_is_requested_with_untrusted_context_and_simulated(self):
        expected = {
            "assumptions": [],
            "inputs": [{"name": "x", "low": 4, "likely": 4, "high": 4, "unit": ""}],
            "outputs": [{"label": "Result", "formula": "x * 3", "unit": ""}],
        }

        class Model:
            def __init__(self):
                self.calls = []

            def complete_json(self, prompt, schema, **kwargs):
                self.calls.append((prompt, schema, kwargs))
                if kwargs["name"] == "numeric_calculation_review":
                    return {"valid": True, "issues": []}
                return expected

        model = Model()
        report = monte_carlo_report("Calculate 4 times 3.", model)

        self.assertIn("| Result | 12 | 12 | 12 |", report)
        plan_prompt, plan_schema, plan_kwargs = model.calls[0]
        self.assertEqual(plan_kwargs["name"], "numeric_calculation_plan")
        self.assertEqual(plan_kwargs["max_tokens"], 8192)
        self.assertIn("triangular distribution", plan_prompt)
        self.assertEqual(plan_schema["required"], ["assumptions", "inputs", "outputs"])
        self.assertEqual(model.calls[1][2]["name"], "numeric_calculation_review")

    def test_plan_is_revised_when_tax_range_is_missing(self):
        class Model:
            def __init__(self):
                self.plans = 0
                self.plan_contexts = []

            def complete_json(self, prompt, schema, **kwargs):
                if kwargs["name"] == "numeric_calculation_review":
                    return {"valid": True, "issues": []}
                self.plans += 1
                self.plan_contexts.append(kwargs["context"])
                tax_range = (0.3, 0.3, 0.3) if self.plans == 1 else (0.1, 0.2, 0.3)
                low, likely, high = tax_range
                return {
                    "assumptions": ["Tax rates are hypothetical scenarios."],
                    "inputs": [
                        {"name": "amount", "low": 100, "likely": 100, "high": 100, "unit": "currency"},
                        {"name": "tax_rate", "low": low, "likely": likely, "high": high, "unit": "fraction"},
                    ],
                    "outputs": [
                        {"label": "Tax", "formula": "amount * tax_rate", "unit": "currency"},
                    ],
                }

        model = Model()

        report = monte_carlo_report("Estimate tax on an amount under a tax regime.", model)

        self.assertEqual(model.plans, 2)
        self.assertIn('"previous_plan"', model.plan_contexts[1])
        self.assertIn('"tax_rate"', model.plan_contexts[1])
        self.assertIn("| Tax |", report)
        self.assertIn("| tax_rate | 0.1 | 0.2 | 0.3 | fraction |", report)

    def test_model_provider_errors_are_reported_without_replanning(self):
        class UnavailableModel:
            last_json_error = "Groq rate limit reached. Wait a few seconds and try again."

            def __init__(self):
                self.calls = 0

            def complete_json(self, prompt, schema, **kwargs):
                self.calls += 1
                return None

        model = UnavailableModel()

        with self.assertRaisesRegex(ValueError, "Groq rate limit reached"):
            monte_carlo_report("Calculate 1 + 1.", model)

        self.assertEqual(model.calls, 1)

    def test_fd_plan_builds_yearly_chart_from_validated_inputs(self):
        class Model:
            def complete_json(self, prompt, schema, **kwargs):
                if kwargs["name"] == "numeric_calculation_review":
                    return {"valid": True, "issues": []}
                return {
                    "assumptions": ["Tax rate is a hypothetical scenario."],
                    "inputs": [
                        {"name": "initial_balance", "low": 0, "likely": 0, "high": 0, "unit": "INR"},
                        {"name": "monthly_payment", "low": 100000, "likely": 100000,
                         "high": 100000, "unit": "INR"},
                        {"name": "annual_rate", "low": 0.0625, "likely": 0.0625,
                         "high": 0.0625, "unit": "fraction"},
                        {"name": "tax_rate", "low": 0.1, "likely": 0.2,
                         "high": 0.3, "unit": "fraction"},
                        {"name": "years", "low": 10, "likely": 10, "high": 10, "unit": "years"},
                    ],
                    "outputs": [
                        {"label": "Placeholder", "formula": "fd_total_gain(0, 0, 0, 0, 1)",
                         "unit": "INR"},
                    ],
                }

        report = monte_carlo_report(
            "Calculate monthly fixed deposit tax for the next 10 years.", Model()
        )

        self.assertIn("| Year 10 tax paid |", report)
        self.assertIn("| Year 10 closing balance after tax |", report)
        self.assertIn("| Total tax paid |", report)
        self.assertIn("| Total after-tax gain |", report)
        self.assertIn("| monthly_payment | 100,000 |", report)


if __name__ == "__main__":
    unittest.main()
