"""Safe Monte Carlo calculations from structured LLM-proposed assumptions."""

import ast
import hashlib
import json
import logging
import math
import operator
import random
import re
import statistics
from typing import Dict, List


SIMULATION_COUNT = 10_000
_MAX_VARIABLES = 20
_MAX_OUTPUTS = 64
_MAX_EXPRESSION_LENGTH = 500
_MAX_AST_NODES = 128
_MAX_ABS_VALUE = 1e100

_CALCULATION_ACTIONS = re.compile(
    r"\b(?:calculate|calculation|compute|work out|estimate|project|projection|"
    r"sum|multiply|divide|convert)\b",
    re.IGNORECASE,
)
_QUANTITY_CUES = re.compile(
    r"\b(?:how much|how many|how old|total|average|mean|median|percentage|percent|"
    r"interest|tax|invest(?:ment)?|balance|gain|profit|loss|growth|return|"
    r"compound(?:ing)?|probability|ratio|rate|forecast|range)\b",
    re.IGNORECASE,
)
_ARITHMETIC_EXPRESSION = re.compile(
    r"\d\s*(?:[+\-*/%^]|\bplus\b|\bminus\b|\btimes\b|"
    r"\bdivided by\b|\bmultiplied by\b)\s*\d",
    re.IGNORECASE,
)

CALCULATION_PLAN_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "assumptions": {"type": "array", "items": {"type": "string"}},
        "inputs": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "name": {"type": "string"},
                    "low": {"type": "number"},
                    "likely": {"type": "number"},
                    "high": {"type": "number"},
                    "unit": {"type": "string"},
                },
                "required": ["name", "low", "likely", "high", "unit"],
            },
        },
        "outputs": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "label": {"type": "string"},
                    "formula": {"type": "string"},
                    "unit": {"type": "string"},
                },
                "required": ["label", "formula", "unit"],
            },
        },
    },
    "required": ["assumptions", "inputs", "outputs"],
}

_PLAN_PROMPT = """Create a structured calculation plan for the numeric task in the untrusted context.
Return assumptions, numeric inputs, and one or more requested outputs as specified by the JSON schema.

For each input, provide low, likely, and high values for a triangular distribution. Use the exact
user-provided value for all three when the value is known exactly. When information is missing,
propose a clearly labeled illustrative range and explain it in assumptions; do not present it as fact.
For tax, legal, or regulatory quantities, ranges are hypothetical scenarios only, never a statement
of applicable law. If the request asks for tax but does not give an explicit tax rate, include a
non-degenerate tax_rate range between 0 and 1 and label it hypothetical in assumptions. Express
rates as fractions, not percentages. An unspecified tax rate must be modeled as an illustrative
scenario, not inferred as the user's applicable rate. For monthly recurring deposits, model the
monthly payment timing and monthly compounding explicitly; do not replace monthly deposits with one
annual lump sum. When tax is deducted annually, tax only the interest earned for that year, then
reinvest the after-tax closing balance. If the request asks for a multi-year annual chart with
annual tax and reinvestment, create sequential tax-paid and closing-balance-after-tax outputs for
each year, referencing the prior year's closing balance, plus total tax and total gain outputs.
For an end-of-month monthly deposit with monthly rate i and 12 deposits per year, the year-end
value of that year's deposits is payment * (((1+i)^12 - 1) / i). Include all material numeric
assumptions, even when that makes the result less precise. Do not add irrelevant random variation
to exact inputs.

Each output formula must be a single scalar expression using only input names, numeric literals,
the constants pi and e, parentheses, +, -, *, /, //, %, **, unary signs, and these functions: abs,
min, max, sum, mean, median, sqrt, log, log10, exp, sin, cos, tan, asin, acos, atan, ceil, floor,
factorial, comb, perm, and round, plus the FD functions described below. Do not return Python code,
statements, loops, attributes, indexing, imports, or function definitions. Put every material
assumed quantity in the input list rather than hard-coding it in a formula. Every identifier used
by a formula must be an input name, pi, e, an allowed function name, or a previously listed
output's snake_case name derived from its label. If a formula uses a derived value, list its output
first with a clear label, then reference that output in later formulas. Check payment timing and
compounding frequency, and ensure related outputs reconcile (for example, final balance =
contributions + after-tax gain). Use one output for every distinct requested result.

For a monthly fixed-deposit plan with annual tax and reinvestment, use the provided functions
fd_annual_tax(initial_balance, monthly_payment, annual_rate, tax_rate, year),
fd_closing_balance(initial_balance, monthly_payment, annual_rate, tax_rate, year),
fd_total_tax(initial_balance, monthly_payment, annual_rate, tax_rate, years), and
fd_total_gain(initial_balance, monthly_payment, annual_rate, tax_rate, years). They apply monthly
compounding with end-of-month deposits and deduct tax from that year's interest before reinvestment.
Use inputs named exactly initial_balance, monthly_payment, annual_rate, tax_rate, and years. Return
only these five inputs and a single output using fd_total_gain; the calculation engine builds the
requested annual tax/balance chart itself. Pass the year number as an integer literal.
If there is not enough information to express the calculation safely, return no outputs and explain
what is missing in assumptions."""

_PLAN_REVIEW_PROMPT = """Independently audit the proposed numeric calculation plan against the
original request. Verify formula algebra, units, input ranges, payment timing, compounding frequency,
and reconciliation among related outputs. All material assumptions must be explicit inputs; reject
hard-coded rates or tax assumptions that are not stated in the request. If tax is requested without
an explicit rate, a non-degenerate 0-to-1 tax_rate input clearly labeled as hypothetical is the
required behavior; do not reject the plan merely because the user did not provide or explicitly
approve that illustrative range. Never present it as an applicable Indian tax rate or legal advice.
For monthly recurring deposits, reject a plan that treats the deposits as one annual lump sum or
calculates interest using annual compounding; check monthly payment timing and compounding.
When annual tax is modeled, ensure it applies to that year's interest, not the entire balance.
For a multi-year annual chart with annual tax and reinvestment, require sequential yearly tax and
after-tax closing balance outputs, with each year's balance feeding the next year, plus totals for
tax and gain. For monthly fixed-deposit plans, accept the provided fd_annual_tax,
fd_closing_balance, fd_total_tax, and fd_total_gain calculation functions because they explicitly
model monthly compounding, end-of-month deposits, annual tax on that year's interest, and
reinvestment. Do not accept a single aggregate tax or final-balance calculation in place of the
requested yearly schedule.
Output formulas may only reference inputs, allowed constants/functions, and outputs listed before
them by their snake_case label name. Return valid=true only if the plan is internally consistent
and directly answers the request; otherwise list concrete corrective issues."""

_PLAN_REVIEW_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "valid": {"type": "boolean"},
        "issues": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["valid", "issues"],
}

_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPERATORS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _as_nonnegative_integer(value, maximum):
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Combinatorics functions require finite integer inputs.")
    integer = int(value)
    if integer != value or integer < 0 or integer > maximum:
        raise ValueError("Combinatorics input is outside the supported range.")
    return integer


def _factorial(value):
    return math.factorial(_as_nonnegative_integer(value, 170))


def _comb(n, k):
    return math.comb(
        _as_nonnegative_integer(n, 1_000),
        _as_nonnegative_integer(k, 1_000),
    )


def _perm(n, k=None):
    n = _as_nonnegative_integer(n, 1_000)
    return math.perm(n, None if k is None else _as_nonnegative_integer(k, 1_000))


def _fd_projection(initial_balance, monthly_payment, annual_rate, tax_rate, years):
    parameters = (initial_balance, monthly_payment, annual_rate, tax_rate)
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        for value in parameters
    ):
        raise ValueError("FD calculation inputs must be finite numbers.")
    initial_balance, monthly_payment, annual_rate, tax_rate = map(float, parameters)
    if initial_balance < 0 or monthly_payment < 0:
        raise ValueError("FD balances and deposits cannot be negative.")
    if annual_rate <= -12 or annual_rate > 10:
        raise ValueError("The annual FD rate is outside the supported range.")
    if not 0 <= tax_rate <= 1:
        raise ValueError("The hypothetical tax rate must be between 0 and 1.")
    years = _as_nonnegative_integer(years, 100)
    if years < 1:
        raise ValueError("FD projection must include at least one year.")

    monthly_rate = annual_rate / 12
    annual_factor = (1 + monthly_rate) ** 12
    if abs(monthly_rate) < 1e-12:
        annual_deposit_value = monthly_payment * 12
    else:
        annual_deposit_value = monthly_payment * (annual_factor - 1) / monthly_rate
    balance = initial_balance
    annual_taxes = []
    annual_balances = []
    for _ in range(years):
        opening_balance = balance
        balance = balance * annual_factor + annual_deposit_value
        annual_interest = max(0.0, balance - opening_balance - 12 * monthly_payment)
        tax = annual_interest * tax_rate
        balance -= tax
        annual_taxes.append(tax)
        annual_balances.append(balance)
    return annual_taxes, annual_balances


def _fd_annual_tax(initial_balance, monthly_payment, annual_rate, tax_rate, year):
    annual_taxes, _ = _fd_projection(
        initial_balance, monthly_payment, annual_rate, tax_rate, year
    )
    return annual_taxes[-1]


def _fd_closing_balance(initial_balance, monthly_payment, annual_rate, tax_rate, year):
    _, annual_balances = _fd_projection(
        initial_balance, monthly_payment, annual_rate, tax_rate, year
    )
    return annual_balances[-1]


def _fd_total_tax(initial_balance, monthly_payment, annual_rate, tax_rate, years):
    annual_taxes, _ = _fd_projection(
        initial_balance, monthly_payment, annual_rate, tax_rate, years
    )
    return sum(annual_taxes)


def _fd_total_gain(initial_balance, monthly_payment, annual_rate, tax_rate, years):
    _, annual_balances = _fd_projection(
        initial_balance, monthly_payment, annual_rate, tax_rate, years
    )
    total_contributions = initial_balance + monthly_payment * 12 * years
    return annual_balances[-1] - total_contributions


_FUNCTIONS = {
    "abs": abs,
    "min": min,
    "max": max,
    "sum": sum,
    "mean": statistics.mean,
    "median": statistics.median,
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "ceil": math.ceil,
    "floor": math.floor,
    "factorial": _factorial,
    "comb": _comb,
    "perm": _perm,
    "round": round,
    "fd_annual_tax": _fd_annual_tax,
    "fd_closing_balance": _fd_closing_balance,
    "fd_total_tax": _fd_total_tax,
    "fd_total_gain": _fd_total_gain,
}
_CONSTANTS = {"pi": math.pi, "e": math.e}


def is_calculation_request(text: str) -> bool:
    """Identify numeric tasks that should use the calculation workflow."""
    text = str(text or "")
    if _CALCULATION_ACTIONS.search(text):
        return True
    has_number = bool(re.search(r"\d", text))
    return bool(
        _ARITHMETIC_EXPRESSION.search(text)
        or (has_number and _QUANTITY_CUES.search(text))
        or re.search(r"\bhow (?:many|much|old)\b", text, re.IGNORECASE)
    )


def _is_monthly_fd_tax_request(text: str) -> bool:
    return bool(
        re.search(r"\b(?:fd|fixed deposits?)\b", text, re.IGNORECASE)
        and re.search(r"\bmonthly\b|\bper month\b", text, re.IGNORECASE)
        and re.search(r"\btax(?:es|ation)?\b", text, re.IGNORECASE)
        and re.search(r"\byears?\b", text, re.IGNORECASE)
    )


def _expand_fd_plan(plan: Dict) -> List[str]:
    inputs = plan.get("inputs")
    if not isinstance(inputs, list):
        return ["The FD plan must include structured calculation inputs."]
    input_names = {
        str(item.get("name", "")).strip()
        for item in inputs
        if isinstance(item, dict)
    }
    required = {"initial_balance", "monthly_payment", "annual_rate", "tax_rate", "years"}
    missing = sorted(required - input_names)
    if missing:
        return [
            "The FD plan must use these exact input names: "
            + ", ".join(sorted(required))
            + f". Missing: {', '.join(missing)}."
        ]
    try:
        distributions = _validate_inputs(inputs)
    except ValueError as exc:
        return [str(exc)]
    year_range = distributions["years"]
    if not (
        year_range[0] == year_range[1] == year_range[2]
        and year_range[0].is_integer()
        and 1 <= year_range[0] <= 30
    ):
        return ["The FD annual chart requires a fixed whole-number horizon from 1 to 30 years."]
    years = int(year_range[0])
    plan["outputs"] = [
        {
            "label": f"Year {year} tax paid",
            "formula": (
                "fd_annual_tax(initial_balance, monthly_payment, annual_rate, tax_rate, "
                f"{year})"
            ),
            "unit": "currency",
        }
        for year in range(1, years + 1)
    ] + [
        {
            "label": f"Year {year} closing balance after tax",
            "formula": (
                "fd_closing_balance(initial_balance, monthly_payment, annual_rate, tax_rate, "
                f"{year})"
            ),
            "unit": "currency",
        }
        for year in range(1, years + 1)
    ] + [
        {
            "label": "Total tax paid",
            "formula": "fd_total_tax(initial_balance, monthly_payment, annual_rate, tax_rate, years)",
            "unit": "currency",
        },
        {
            "label": "Total after-tax gain",
            "formula": "fd_total_gain(initial_balance, monthly_payment, annual_rate, tax_rate, years)",
            "unit": "currency",
        },
    ]
    return []


def monte_carlo_report(prompt: str, model, context="", history=None) -> str:
    """Ask the LLM for assumptions/formulas, then evaluate only safe expressions."""
    calculation_context = {
        "current_request": str(prompt or ""),
        "prior_conversation": history or [],
        "attachments_or_reference": str(context or ""),
    }
    revision_issues = []
    previous_plan = None
    for attempt in range(3):
        planning_context = dict(calculation_context)
        planning_prompt = _PLAN_PROMPT
        if revision_issues:
            planning_context["issues_to_correct"] = revision_issues
            planning_context["previous_plan"] = previous_plan
            planning_prompt += (
                "\n\nCorrect the previous plan's listed issues. The previous plan and exact input "
                "names are in the context. Revise it minimally, declare every material numeric "
                "quantity as an input, and ensure every formula identifier is a declared input or "
                "a previously listed output's snake_case label name. Return a complete revised plan."
            )
        plan = model.complete_json(
            planning_prompt,
            CALCULATION_PLAN_SCHEMA,
            name="numeric_calculation_plan",
            model_type="reasoning",
            temperature=0.1,
            max_tokens=8192,
            context=json.dumps(planning_context, ensure_ascii=False),
        )
        if not isinstance(plan, dict):
            model_error = str(getattr(model, "last_json_error", "") or "").strip()
            if model_error:
                raise ValueError(f"The calculation planner could not return a plan: {model_error}")
            revision_issues = ["The model did not return a valid structured calculation plan."]
            previous_plan = None
            continue
        if _is_monthly_fd_tax_request(prompt):
            revision_issues = _expand_fd_plan(plan)
            if revision_issues:
                previous_plan = plan
                continue
        revision_issues = _plan_validation_issues(plan, prompt)
        if revision_issues:
            previous_plan = plan
            continue
        review_context = json.dumps(
            {"request": calculation_context, "proposed_plan": plan},
            ensure_ascii=False,
        )
        review = model.complete_json(
            _PLAN_REVIEW_PROMPT,
            _PLAN_REVIEW_SCHEMA,
            name="numeric_calculation_review",
            model_type="fast",
            temperature=0,
            max_tokens=1024,
            context=review_context,
        )
        if (
            isinstance(review, dict)
            and review.get("valid") is True
            and isinstance(review.get("issues"), list)
            and not review["issues"]
        ):
            return _simulate_plan(plan, seed=_stable_seed(calculation_context))
        previous_plan = plan
        if (
            isinstance(review, dict)
            and isinstance(review.get("issues"), list)
            and review.get("valid") is not True
        ):
            revision_issues = [str(issue)[:300] for issue in review["issues"][:8]]
            if not revision_issues:
                revision_issues = [
                    "The independent reviewer rejected the plan without details; recheck "
                    "assumptions, formulas, and output dependencies."
                ]
        elif isinstance(review, dict) and isinstance(review.get("issues"), list):
            revision_issues = [str(issue)[:300] for issue in review["issues"][:8]]
        else:
            revision_issues = ["The calculation plan could not be independently reviewed."]

    issues = "; ".join(revision_issues[:3]) or "the plan remained inconsistent"
    logging.error("Monte Carlo plan validation failed: %s", issues)
    raise ValueError(
        "I couldn't verify a consistent calculation plan after two correction attempts. "
        f"Please clarify the assumptions or simplify the calculation. Issues: {issues}"
    )


def _plan_validation_issues(plan: Dict, prompt: str) -> List[str]:
    assumptions = plan.get("assumptions")
    inputs = plan.get("inputs")
    outputs = plan.get("outputs")
    if not isinstance(assumptions, list) or not isinstance(inputs, list) or not isinstance(outputs, list):
        return ["The plan is missing assumptions, inputs, or outputs."]
    if not outputs:
        return ["The plan has no calculable outputs."]
    if len(inputs) > _MAX_VARIABLES:
        return [f"The plan has {len(inputs)} inputs; the supported maximum is {_MAX_VARIABLES}."]
    if len(outputs) > _MAX_OUTPUTS:
        return [f"The plan has {len(outputs)} outputs; the supported maximum is {_MAX_OUTPUTS}."]
    try:
        distributions = _validate_inputs(inputs)
        if any(not isinstance(output, dict) for output in outputs):
            return ["Every output must include a label, formula, and unit."]
        _validate_outputs(outputs, set(distributions))
    except ValueError as exc:
        return [str(exc)]

    if re.search(r"\b(tax|taxation|taxable)\b", prompt, re.IGNORECASE):
        explicit_tax_rate = re.search(
            r"(?:tax(?:able)?\s+rate|tax\s+at)\D{0,20}\d+(?:\.\d+)?\s*(?:%|percent)"
            r"|\d+(?:\.\d+)?\s*(?:%|percent)\s*(?:tax|of tax)",
            prompt,
            re.IGNORECASE,
        )
        tax_inputs = [
            item for item in inputs
            if isinstance(item, dict) and "tax" in str(item.get("name", "")).lower()
        ]
        if not tax_inputs:
            return ["Tax is requested but no explicit tax input is included."]
        if not explicit_tax_rate and not any(float(item["low"]) < float(item["high"]) for item in tax_inputs):
            return ["Tax rate is unspecified; provide a hypothetical tax_rate range."]
        for item in tax_inputs:
            if (
                "rate" in str(item["name"]).lower()
                and (float(item["low"]) < 0 or float(item["high"]) > 1)
            ):
                return ["Tax rates must be expressed as fractions from 0 to 1."]
    return []


def _stable_seed(value) -> int:
    digest = hashlib.sha256(repr(value).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _simulate_plan(plan: Dict, seed=0, samples=SIMULATION_COUNT) -> str:
    if not isinstance(samples, int) or not 1 <= samples <= SIMULATION_COUNT:
        raise ValueError("Simulation count must be between 1 and 10,000.")

    assumptions = plan.get("assumptions")
    inputs = plan.get("inputs")
    outputs = plan.get("outputs")
    if not isinstance(assumptions, list) or not isinstance(inputs, list) or not isinstance(outputs, list):
        raise ValueError("The calculation plan is missing assumptions, inputs, or outputs.")
    if not outputs:
        details = "; ".join(str(item)[:300] for item in assumptions[:5])
        raise ValueError(
            "The model could not express a safe calculation formula. "
            f"Missing details: {details or 'please clarify the calculation inputs.'}"
        )
    if len(inputs) > _MAX_VARIABLES or len(outputs) > _MAX_OUTPUTS:
        raise ValueError("The calculation plan contains too many inputs or outputs.")

    distributions = _validate_inputs(inputs)
    parsed_outputs = _validate_outputs(outputs, set(distributions))
    if len(parsed_outputs) != len(outputs):
        raise ValueError("Every calculated output must be a structured formula.")

    rng = random.Random(seed)
    results = [[] for _ in parsed_outputs]
    failed = [0 for _ in parsed_outputs]
    for _ in range(samples):
        values = {
            name: rng.triangular(low, high, mode)
            for name, (low, mode, high) in distributions.items()
        }
        for index, (output, tree, output_name) in enumerate(parsed_outputs):
            try:
                value = _evaluate(tree.body, values)
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError("Formula did not produce a numeric result.")
                value = float(value)
                if not math.isfinite(value) or abs(value) > _MAX_ABS_VALUE:
                    raise ValueError("Formula result is outside the supported numeric range.")
                results[index].append(value)
                if output_name:
                    values[output_name] = value
            except (ArithmeticError, KeyError, OverflowError, TypeError, ValueError):
                failed[index] += 1

    if any(not values for values in results):
        raise ValueError("The formula failed for every simulation; check its domain and inputs.")
    return _format_report(
        assumptions,
        inputs,
        parsed_outputs,
        results,
        failed,
        samples,
    )


def _validate_inputs(inputs: List[Dict]) -> Dict[str, tuple]:
    distributions = {}
    reserved = set(_FUNCTIONS) | set(_CONSTANTS)
    for item in inputs:
        if not isinstance(item, dict):
            raise ValueError("Each calculation input must be a structured object.")
        name = str(item.get("name", "")).strip()
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,39}", name):
            raise ValueError(f"Invalid calculation input name: {name!r}.")
        if name in reserved or name in distributions:
            raise ValueError(f"Duplicate or reserved calculation input name: {name}.")
        try:
            low, mode, high = (float(item[key]) for key in ("low", "likely", "high"))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Input {name} must have numeric low, likely, and high values.") from exc
        if not all(math.isfinite(value) and abs(value) <= 1e12 for value in (low, mode, high)):
            raise ValueError(f"Input {name} contains a non-finite or out-of-range value.")
        if not low <= mode <= high:
            raise ValueError(f"Input {name} must satisfy low <= likely <= high.")
        distributions[name] = (low, mode, high)
    return distributions


def _validate_expression(output: Dict, variable_names: set):
    label = str(output.get("label", "")).strip()
    expression = str(output.get("formula", "")).strip()
    if not label or not expression or len(expression) > _MAX_EXPRESSION_LENGTH:
        raise ValueError("Every output needs a label and a short numeric formula.")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid formula for {label}: {exc.msg}.") from exc
    nodes = list(ast.walk(tree))
    if len(nodes) > _MAX_AST_NODES:
        raise ValueError(f"Formula for {label} is too complex.")

    allowed_names = variable_names | set(_FUNCTIONS) | set(_CONSTANTS)
    for node in nodes:
        if isinstance(node, ast.Name) and node.id not in allowed_names:
            raise ValueError(f"Formula for {label} uses unknown name {node.id!r}.")
        if isinstance(node, ast.Call):
            if (
                not isinstance(node.func, ast.Name)
                or node.func.id not in _FUNCTIONS
                or node.keywords
            ):
                raise ValueError(f"Formula for {label} uses a disallowed function call.")
        elif isinstance(node, ast.BinOp) and type(node.op) not in _BINARY_OPERATORS:
            raise ValueError(f"Formula for {label} uses a disallowed operator.")
        elif isinstance(node, ast.UnaryOp) and type(node.op) not in _UNARY_OPERATORS:
            raise ValueError(f"Formula for {label} uses a disallowed unary operator.")
        elif isinstance(node, ast.Constant) and (
            isinstance(node.value, bool) or not isinstance(node.value, (int, float))
        ):
            raise ValueError(f"Formula for {label} contains a non-numeric constant.")
        elif not isinstance(
            node,
            (
                ast.Expression, ast.Load, ast.Name, ast.Call, ast.BinOp, ast.UnaryOp,
                ast.Constant, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv,
                ast.Mod, ast.Pow, ast.UAdd, ast.USub,
            ),
        ):
            raise ValueError(f"Formula for {label} contains unsupported syntax.")
    return output, tree


def _output_symbol(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


def _validate_outputs(outputs: List[Dict], input_names: set):
    available_names = set(input_names)
    parsed_outputs = []
    reserved_names = set(_FUNCTIONS) | set(_CONSTANTS)
    for output in outputs:
        parsed_output = _validate_expression(output, available_names)
        output_name = _output_symbol(str(output["label"]).strip())
        if output_name and output_name not in reserved_names and output_name not in available_names:
            available_names.add(output_name)
        else:
            output_name = ""
        parsed_outputs.append((*parsed_output, output_name))
    return parsed_outputs


def _evaluate(node, values):
    if isinstance(node, ast.Constant):
        return _safe_number(node.value)
    if isinstance(node, ast.Name):
        if node.id in _FUNCTIONS:
            return _FUNCTIONS[node.id]
        return _CONSTANTS[node.id] if node.id in _CONSTANTS else values[node.id]
    if isinstance(node, ast.UnaryOp):
        return _UNARY_OPERATORS[type(node.op)](_evaluate(node.operand, values))
    if isinstance(node, ast.BinOp):
        left = _evaluate(node.left, values)
        right = _evaluate(node.right, values)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("Exponent is outside the supported range.")
        return _safe_number(_BINARY_OPERATORS[type(node.op)](left, right))
    if isinstance(node, ast.Call):
        function = _FUNCTIONS[node.func.id]
        return _safe_number(function(*[_evaluate(arg, values) for arg in node.args]))
    raise ValueError("Unsupported formula syntax.")


def _safe_number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Formula produced a non-numeric value.")
    if not math.isfinite(value) or abs(value) > _MAX_ABS_VALUE:
        raise ValueError("Intermediate formula value is outside the supported range.")
    return value


def _format_report(assumptions, inputs, outputs, results, failed, samples) -> str:
    lines = [
        f"### Monte Carlo calculation ({samples:,} simulations)",
        "",
        "Results are distributions over the assumptions below, not guaranteed outcomes.",
        "Each input was sampled independently from a triangular distribution "
        "(low, most likely, high).",
        "",
        "| Output | 5th percentile | Median | 95th percentile | Unit |",
        "|---|---:|---:|---:|---|",
    ]
    for (output, _, _), values, failures in zip(outputs, results, failed):
        label = _table_text(output["label"])
        unit = _table_text(str(output.get("unit", "")))
        ordered = sorted(values)
        p05 = _percentile(ordered, 0.05)
        median = statistics.median(ordered)
        p95 = _percentile(ordered, 0.95)
        lines.append(
            f"| {label} | {_number(p05)} | {_number(median)} | {_number(p95)} | {unit} |"
        )
        if failures:
            lines.append(
                f"\nSome samples for **{label}** were outside the formula's valid domain: "
                f"{failures:,}/{samples:,} excluded."
            )

    lines.extend([
        "",
        "### Input ranges",
        "",
        "| Input | Low | Most likely | High | Unit |",
        "|---|---:|---:|---:|---|",
    ])
    for item in inputs:
        lines.append(
            f"| {_table_text(str(item['name']))} | {_number(float(item['low']))} | "
            f"{_number(float(item['likely']))} | {_number(float(item['high']))} | "
            f"{_table_text(str(item.get('unit', '')))} |"
        )

    lines.extend(["", "### Formulas", ""])
    for output, _, _ in outputs:
        lines.append(f"- **{_table_text(output['label'])}:** `{_table_text(output['formula'])}`")

    lines.extend(["", "### Assumptions", ""])
    lines.append("- Input ranges are LLM-proposed; exact values from the question may be held fixed.")
    for assumption in assumptions[:15]:
        lines.append(f"- {_table_text(str(assumption)[:500])}")

    lines.extend([
        "",
        "**Interpretation:** the percentiles summarize these illustrative input ranges; "
        "they are not confidence intervals or a guarantee. Legal/tax ranges are hypothetical, "
        "not advice about applicable rules. Verify important results independently.",
    ])
    return "\n".join(lines)


def _percentile(ordered: List[float], fraction: float) -> float:
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _number(value: float) -> str:
    if value == 0:
        return "0"
    return f"{value:,.4f}".rstrip("0").rstrip(".")


def _table_text(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ").replace("`", "'")
