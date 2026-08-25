"""Sec. 25 'metric denominators correct' tests for juniper_math.sft_eval,
using hand-built CaseMetrics so the report math is tested in isolation from
generation/model behavior."""

from __future__ import annotations

from fractions import Fraction

from juniper_math.sft_eval import CaseMetrics, ToolInteractionReport, _parse_number


def _case(**overrides) -> CaseMetrics:
    defaults = dict(
        case_id="c",
        category="arithmetic",
        tool_required=False,
        expected_tool=None,
        tool_invocation_required=False,
        model_invoked_tool=False,
        correct_routing=True,
        emitted_tool_call=False,
        call_parsed=False,
        tool_name_correct=None,
        arguments_correct=None,
        execution_successful=None,
        end_to_end_success=None,
        final_answer_consistent_with_result=None,
        final_answer_correct=True,
        unnecessary_tool_call=False,
        missing_required_call=False,
        fabricated_result_attempted=False,
        terminal_tag="final",
        terminal_tag_correct=True,
        stopped_reason="terminal_tag",
    )
    defaults.update(overrides)
    return CaseMetrics(**defaults)


def test_empty_denominator_reports_none_not_zero():
    report = ToolInteractionReport(suite_id="s", n_cases=2, cases=[_case(), _case()])
    d = report.as_dict()
    # No tool_required cases at all -> tool-specific rates over that subset must be None, not 0.0.
    assert d["end_to_end_success_on_tool_required"]["rate"] is None
    assert d["missing_required_call"]["rate"] is None


def test_direct_vs_tool_denominators_partition_correctly():
    cases = [
        _case(tool_invocation_required=False, unnecessary_tool_call=False),
        _case(
            tool_invocation_required=False,
            unnecessary_tool_call=True,
            emitted_tool_call=True,
            correct_routing=False,
        ),
        _case(
            tool_invocation_required=True,
            missing_required_call=False,
            execution_successful=True,
            tool_required=True,
        ),
    ]
    report = ToolInteractionReport(suite_id="s", n_cases=3, cases=cases)
    d = report.as_dict()
    assert d["n_direct_cases"] == 2
    assert d["n_tool_required_cases"] == 1
    assert d["unnecessary_tool_call"]["denominator"] == 2
    assert d["unnecessary_tool_call"]["numerator"] == 1
    assert d["missing_required_call"]["denominator"] == 1
    assert d["required_tool_call_emitted"]["denominator"] == 1
    assert d["required_tool_call_parsed_valid"]["denominator"] == 1


def test_tool_name_correct_uses_all_required_tool_cases():
    """Malformed/missing calls are failures, not denominator exclusions."""
    cases = [
        _case(tool_name_correct=None),  # not applicable
        _case(tool_invocation_required=True, tool_required=True, tool_name_correct=True),
        _case(tool_invocation_required=True, tool_required=True, tool_name_correct=False),
        _case(tool_invocation_required=True, tool_required=True, tool_name_correct=False),
    ]
    report = ToolInteractionReport(suite_id="s", n_cases=4, cases=cases)
    d = report.as_dict()
    assert d["tool_name_correct"]["denominator"] == 3
    assert d["tool_name_correct"]["numerator"] == 1


def test_required_tool_parsing_uses_all_required_cases():
    cases = [
        _case(tool_invocation_required=True, tool_required=True, emitted_tool_call=True, call_parsed=True),
        _case(tool_invocation_required=True, tool_required=True, emitted_tool_call=True, call_parsed=False),
        _case(tool_invocation_required=True, tool_required=True, emitted_tool_call=False, call_parsed=False),
    ]
    d = ToolInteractionReport(suite_id="s", n_cases=3, cases=cases).as_dict()
    assert d["required_tool_call_emitted"] == {"numerator": 2, "denominator": 3, "rate": 2 / 3}
    assert d["required_tool_call_parsed_valid"] == {"numerator": 1, "denominator": 3, "rate": 1 / 3}


def test_numerator_never_exceeds_denominator_for_every_metric():
    cases = [
        _case(),
        _case(tool_invocation_required=True, tool_required=True),
        _case(emitted_tool_call=True, call_parsed=True, tool_name_correct=True),
    ]
    report = ToolInteractionReport(suite_id="s", n_cases=3, cases=cases)
    d = report.as_dict()
    for key, value in d.items():
        if isinstance(value, dict) and "numerator" in value and value["denominator"]:
            assert value["numerator"] <= value["denominator"], f"{key} numerator exceeds denominator"


def test_n_cases_matches_len_cases():
    report = ToolInteractionReport(suite_id="s", n_cases=5, cases=[_case() for _ in range(5)])
    d = report.as_dict()
    assert d["n_cases"] == 5
    assert d["correct_routing"]["denominator"] == 5


def test_numeric_parser_accepts_valid_final_answer_representations():
    assert _parse_number("The result is $12.50.") == Fraction(25, 2)
    assert _parse_number("1,234.5 meters") == Fraction(2469, 2)
    assert _parse_number("approximately -1.2e-3") == Fraction(-3, 2500)
    assert _parse_number("1e3") == Fraction(1000)
    assert _parse_number("1E3") == Fraction(1000)
    assert _parse_number("-1e3") == Fraction(-1000)
    assert _parse_number("1.2e3") == Fraction(1200)
    assert _parse_number("The exact result is 7 / 8.") == Fraction(7, 8)
    assert _parse_number("15%") == Fraction(15)


def test_numeric_parser_uses_terminal_number_in_an_explanation():
    assert _parse_number("We add 2 and 3, so the final answer is 5.") == Fraction(5)
