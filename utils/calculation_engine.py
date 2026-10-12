"""Deterministic fixed-deposit projection helpers used by the local calculator."""

import math


def _as_nonnegative_integer(value, maximum):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError("Projection years must be a finite integer.")
    integer = int(value)
    if integer != value or integer < 0 or integer > maximum:
        raise ValueError("Projection years are outside the supported range.")
    return integer


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
