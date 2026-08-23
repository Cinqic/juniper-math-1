"""Tool-invoking generator families: unit_conversion, financial_math,
tool_use, incorrect_tool_call, tool_error.

Ground truth for every example here comes from actually executing the real
Phase 3 ``ToolRuntime`` (Sec. 9) — never a fabricated ``<tool_result>``
string. The recorded ``tool_traces`` are the literal call/result pair the
runtime produced.
"""

from __future__ import annotations

import json

from juniper_math.dataset.generators.common import (
    GENERATOR_VERSION,
    choose_template,
    difficulty_for,
    family_rng,
    rand_int,
)
from juniper_math.dataset.idgen import derive_id
from juniper_math.dataset.schema import Example, ToolTrace
from juniper_math.tools.runtime import ToolRuntime

PROTOCOL_VERSION = "1.0.0"
GENERATOR_ID = "tool_runtime_core"


def _run(runtime: ToolRuntime, tool: str, arguments: dict) -> ToolTrace:
    call_dict = {"protocol_version": PROTOCOL_VERSION, "tool": tool, "arguments": arguments}
    text = json.dumps(call_dict, sort_keys=True, separators=(",", ":"))
    result = runtime.execute_text(text)
    return ToolTrace(call=call_dict, result=result.to_dict())


# --------------------------------------------------------------------------
# unit_conversion
# --------------------------------------------------------------------------

_UNIT_TABLE = {
    "length": ["millimeter", "centimeter", "meter", "kilometer", "inch", "foot", "yard", "mile"],
    "mass": ["milligram", "gram", "kilogram", "ounce", "pound"],
    "temperature": ["celsius", "fahrenheit", "kelvin"],
    "area": ["square_meter", "square_kilometer", "square_foot", "acre", "hectare"],
    "volume": ["milliliter", "liter", "cubic_meter", "gallon_us", "quart_us", "cup_us"],
    "speed": ["meters_per_second", "kilometers_per_hour", "miles_per_hour"],
    "time": ["second", "minute", "hour", "day", "week"],
    "data_storage": ["byte", "kilobyte", "megabyte", "gigabyte", "terabyte", "kibibyte", "mebibyte"],
}

_CONVERT_TEMPLATES = [
    "Convert {value} {from_unit} to {to_unit}.",
    "How many {to_unit} are in {value} {from_unit}?",
    "What is {value} {from_unit} expressed in {to_unit}?",
]


def make_unit_conversion(index: int, master_seed: int, runtime: ToolRuntime) -> Example:
    family_id = "convert_between_units"
    rng, seed = family_rng(GENERATOR_ID, family_id, index, master_seed)
    difficulty = difficulty_for(rng)
    category = rng.choice(list(_UNIT_TABLE))
    units = _UNIT_TABLE[category]
    from_unit, to_unit = rng.sample(units, 2)
    scale = {"trivial": 20, "easy": 200, "medium": 5000, "hard": 100000}[difficulty]
    value = rand_int(rng, 1, scale)
    t_idx, template = choose_template(rng, _CONVERT_TEMPLATES)
    prompt = template.format(
        value=value, from_unit=from_unit.replace("_", " "), to_unit=to_unit.replace("_", " ")
    )
    trace = _run(
        runtime,
        "calculator.convert",
        {"category": category, "from_unit": from_unit, "to_unit": to_unit, "value": value},
    )
    success = trace.result["status"] == "success"
    return Example(
        example_id=derive_id(
            "example", GENERATOR_ID, family_id, f"t{t_idx}", category, from_unit, to_unit, value, length=24
        ),
        generator_id=GENERATOR_ID,
        generator_version=GENERATOR_VERSION,
        family_id=family_id,
        template_id=f"t{t_idx}",
        derivation_id=derive_id(family_id, category, from_unit, to_unit, value),
        seed=seed,
        category="unit_conversion",
        difficulty=difficulty,
        synthetic=True,
        split="train",
        prompt=prompt,
        expected_behavior="invoke_tool",
        expected_answer=trace.result["result"]["value"] if success else None,
        tolerance=0,
        tool_required=True,
        tool_name="calculator.convert",
        tool_traces=(trace,),
        verification={"mode": "tool", "expression": None},
        provenance=f"{GENERATOR_ID}/{family_id} v{GENERATOR_VERSION}",
        notes="ground truth executed via the real Phase 3 ToolRuntime, not recomputed independently",
    )


# --------------------------------------------------------------------------
# financial_math
# --------------------------------------------------------------------------

_FINANCE_TEMPLATES: dict[str, list[str]] = {
    "tip": ["What is a {percent}% tip on a ${bill} bill?"],
    "sales_tax": ["What is the sales tax on a ${price} purchase at a {percent}% tax rate?"],
    "discount": ["A ${price} item is discounted {percent}%. What is the discount amount?"],
    "final_price": [
        "A ${price} item has a {percent}% discount and then {percent2}% sales tax applied. "
        "What is the final price?"
    ],
    "split_bill": [
        "A ${bill} bill is split evenly among {people} people, with no tip. How much does each person pay?"
    ],
    "simple_interest": [
        "What is the simple interest on a ${principal} principal at {percent}% annual rate "
        "over {years} years?"
    ],
}


def make_financial_math(index: int, master_seed: int, runtime: ToolRuntime) -> Example:
    family_id = "finance_operation"
    rng, seed = family_rng(GENERATOR_ID, family_id, index, master_seed)
    difficulty = difficulty_for(rng)
    op = rng.choice(list(_FINANCE_TEMPLATES))
    scale = {"trivial": 50, "easy": 300, "medium": 3000, "hard": 50000}[difficulty]
    template = _FINANCE_TEMPLATES[op][0]

    if op == "tip":
        bill, percent = rand_int(rng, 5, scale), rng.choice([10, 15, 18, 20, 25])
        prompt = template.format(bill=bill, percent=percent)
        args = {"bill_total": bill, "tip_percent": percent}
    elif op == "sales_tax":
        price, percent = rand_int(rng, 5, scale), rng.choice([5, 6, 7, 8, 9, 10])
        prompt = template.format(price=price, percent=percent)
        args = {"price": price, "tax_rate_percent": percent}
    elif op == "discount":
        price, percent = rand_int(rng, 5, scale), rng.choice([10, 15, 20, 25, 30, 50])
        prompt = template.format(price=price, percent=percent)
        args = {"price": price, "percent": percent}
    elif op == "final_price":
        price, percent, percent2 = rand_int(rng, 5, scale), rng.choice([10, 20, 30]), rng.choice([5, 8, 10])
        prompt = template.format(price=price, percent=percent, percent2=percent2)
        args = {"price": price, "discount_percent": percent, "tax_rate_percent": percent2}
    elif op == "split_bill":
        bill, people = rand_int(rng, 10, scale), rng.randint(2, 8)
        prompt = template.format(bill=bill, people=people)
        args = {"bill_total": bill, "num_people": people}
    else:  # simple_interest
        principal = rand_int(rng, 20, max(100, scale))
        percent, years = rng.choice([2, 3, 4, 5, 6, 7]), rng.randint(1, 10)
        prompt = template.format(principal=principal, percent=percent, years=years)
        args = {"principal": principal, "annual_rate_percent": percent, "years": years}

    trace = _run(runtime, "calculator.finance", {"operation": op, **args})
    success = trace.result["status"] == "success"
    return Example(
        example_id=derive_id(
            "example", GENERATOR_ID, family_id, op, json.dumps(args, sort_keys=True), length=24
        ),
        generator_id=GENERATOR_ID,
        generator_version=GENERATOR_VERSION,
        family_id=family_id,
        template_id=op,
        derivation_id=derive_id(family_id, op, json.dumps(args, sort_keys=True)),
        seed=seed,
        category="financial_math",
        difficulty=difficulty,
        synthetic=True,
        split="train",
        prompt=prompt,
        expected_behavior="invoke_tool",
        expected_answer=trace.result["result"]["value"] if success else None,
        tolerance=0,
        tool_required=True,
        tool_name="calculator.finance",
        tool_traces=(trace,),
        verification={"mode": "tool", "expression": None},
        provenance=f"{GENERATOR_ID}/{family_id} v{GENERATOR_VERSION}",
        notes="ground truth executed via the real Phase 3 ToolRuntime, not recomputed independently",
    )


# --------------------------------------------------------------------------
# tool_use — general calculator.evaluate invocation prompts
# --------------------------------------------------------------------------

_TOOL_USE_TEMPLATES = [
    "Use the calculator to compute {expr}.",
    "I need the exact value of {expr} — please calculate it.",
]


def make_tool_use(index: int, master_seed: int, runtime: ToolRuntime) -> Example:
    family_id = "evaluate_large_expression"
    rng, seed = family_rng(GENERATOR_ID, family_id, index, master_seed)
    difficulty = difficulty_for(rng)
    scale = {"trivial": 10000, "easy": 100000, "medium": 1000000, "hard": 50000000}[difficulty]
    a, b = rand_int(rng, 1000, scale), rand_int(rng, 2, 999)
    op = rng.choice(["*", "+", "-"])
    expr = f"{a} {op} {b}"
    t_idx, template = choose_template(rng, _TOOL_USE_TEMPLATES)
    prompt = template.format(expr=expr)
    trace = _run(runtime, "calculator.evaluate", {"expression": expr})
    success = trace.result["status"] == "success"
    return Example(
        example_id=derive_id("example", GENERATOR_ID, family_id, f"t{t_idx}", expr, length=24),
        generator_id=GENERATOR_ID,
        generator_version=GENERATOR_VERSION,
        family_id=family_id,
        template_id=f"t{t_idx}",
        derivation_id=derive_id(family_id, expr),
        seed=seed,
        category="tool_use",
        difficulty=difficulty,
        synthetic=True,
        split="train",
        prompt=prompt,
        expected_behavior="invoke_tool",
        expected_answer=trace.result["result"]["value"] if success else None,
        tolerance=0,
        tool_required=True,
        tool_name="calculator.evaluate",
        tool_traces=(trace,),
        verification={"mode": "tool", "expression": None},
        provenance=f"{GENERATOR_ID}/{family_id} v{GENERATOR_VERSION}",
        notes="ground truth executed via the real Phase 3 ToolRuntime, not recomputed independently",
    )


# --------------------------------------------------------------------------
# incorrect_tool_call — a proposed call is wrong; the correct call/result is
# the recorded ground truth, and the mistake is described in notes.
# --------------------------------------------------------------------------

_WRONG_CALL_TEMPLATES = [
    "A colleague proposed converting {value} {from_unit} to {to_unit} by calling "
    "calculator.convert with category '{wrong_category}' instead of '{category}'. "
    "What is the correct conversion result?",
    "Someone tried to compute {percent}% of {whole} using calculator.finance operation "
    "'percentage_increase' instead of 'percentage_of'. What is the correct result for {percent}% of {whole}?",
]


def make_incorrect_tool_call(index: int, master_seed: int, runtime: ToolRuntime) -> Example:
    family_id = "wrong_tool_argument"
    rng, seed = family_rng(GENERATOR_ID, family_id, index, master_seed)
    difficulty = difficulty_for(rng)
    t_idx = rng.randrange(len(_WRONG_CALL_TEMPLATES))

    if t_idx == 0:
        category = rng.choice(["length", "mass", "volume"])
        other_categories = [c for c in _UNIT_TABLE if c != category and c in ("length", "mass", "volume")]
        wrong_category = rng.choice(other_categories) if other_categories else category
        from_unit, to_unit = rng.sample(_UNIT_TABLE[category], 2)
        value = rand_int(rng, 1, {"trivial": 20, "easy": 200, "medium": 2000, "hard": 20000}[difficulty])
        prompt = _WRONG_CALL_TEMPLATES[0].format(
            value=value,
            from_unit=from_unit.replace("_", " "),
            to_unit=to_unit.replace("_", " "),
            wrong_category=wrong_category,
            category=category,
        )
        trace = _run(
            runtime,
            "calculator.convert",
            {"category": category, "from_unit": from_unit, "to_unit": to_unit, "value": value},
        )
        tool_name = "calculator.convert"
        derivation_key: tuple[str | int, ...] = (
            "convert",
            category,
            wrong_category,
            from_unit,
            to_unit,
            value,
        )
    else:
        whole = rand_int(rng, 10, {"trivial": 100, "easy": 500, "medium": 5000, "hard": 50000}[difficulty])
        percent = rng.choice([5, 10, 15, 20, 25, 50])
        prompt = _WRONG_CALL_TEMPLATES[1].format(percent=percent, whole=whole)
        trace = _run(
            runtime, "calculator.finance", {"operation": "percentage_of", "number": whole, "percent": percent}
        )
        tool_name = "calculator.finance"
        derivation_key = ("finance", whole, percent)

    success = trace.result["status"] == "success"
    return Example(
        example_id=derive_id("example", GENERATOR_ID, family_id, f"t{t_idx}", *derivation_key, length=24),
        generator_id=GENERATOR_ID,
        generator_version=GENERATOR_VERSION,
        family_id=family_id,
        template_id=f"t{t_idx}",
        derivation_id=derive_id(family_id, *derivation_key),
        seed=seed,
        category="incorrect_tool_call",
        difficulty=difficulty,
        synthetic=True,
        split="train",
        prompt=prompt,
        expected_behavior="flag_incorrect_answer",
        expected_answer=trace.result["result"]["value"] if success else None,
        tolerance=0,
        tool_required=True,
        tool_name=tool_name,
        tool_traces=(trace,),
        verification={"mode": "tool", "expression": None},
        provenance=f"{GENERATOR_ID}/{family_id} v{GENERATOR_VERSION}",
        notes="the proposed call in the prompt uses the wrong argument/category; the recorded ground "
        "truth is the CORRECT call's real result, executed via the real ToolRuntime",
    )


# --------------------------------------------------------------------------
# tool_error — a legitimately-invoked tool call that the real runtime
# reports as an error (division by zero, unsupported unit, etc). The model
# should still invoke the tool; the runtime's error is the expected outcome.
# --------------------------------------------------------------------------

_TOOL_ERROR_TEMPLATES = [
    "Use the calculator to compute {expr}.",
]


def make_tool_error(index: int, master_seed: int, runtime: ToolRuntime) -> Example:
    family_id = "tool_runtime_error"
    rng, seed = family_rng(GENERATOR_ID, family_id, index, master_seed)
    difficulty = difficulty_for(rng)
    shape = rng.choice(["div_zero", "factorial_negative", "huge_pow"])
    if shape == "div_zero":
        n = rand_int(rng, 1, 1000)
        expr = f"{n} / 0"
    elif shape == "factorial_negative":
        n = rand_int(rng, 1, 20)
        expr = f"factorial(-{n})"
    else:
        base = rand_int(rng, 2, 9)
        expr = f"{base} ** 100000"
    prompt = _TOOL_ERROR_TEMPLATES[0].format(expr=expr)
    trace = _run(runtime, "calculator.evaluate", {"expression": expr})
    return Example(
        example_id=derive_id("example", GENERATOR_ID, family_id, shape, expr, length=24),
        generator_id=GENERATOR_ID,
        generator_version=GENERATOR_VERSION,
        family_id=family_id,
        template_id=shape,
        derivation_id=derive_id(family_id, shape, expr),
        seed=seed,
        category="tool_error",
        difficulty=difficulty,
        synthetic=True,
        split="train",
        prompt=prompt,
        expected_behavior="invoke_tool",
        expected_answer=None,
        tolerance=None,
        tool_required=True,
        tool_name="calculator.evaluate",
        tool_traces=(trace,),
        verification={"mode": "tool", "expression": None},
        provenance=f"{GENERATOR_ID}/{family_id} v{GENERATOR_VERSION}",
        notes=f"the real ToolRuntime is expected to report status={trace.result['status']!r} "
        f"error_code={(trace.result.get('error') or {}).get('code')!r} for this call; the model "
        "should surface that outcome, not fabricate a numeric answer",
    )


FAMILIES = [
    ("unit_conversion", make_unit_conversion),
    ("financial_math", make_financial_math),
    ("tool_use", make_tool_use),
    ("incorrect_tool_call", make_incorrect_tool_call),
    ("tool_error", make_tool_error),
]
