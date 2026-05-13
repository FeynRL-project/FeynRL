from rewards.llm_judge_reward_func import _parse_criteria_met


def test_parse_criteria_met_allows_extra_keys():
    assert _parse_criteria_met('{"explanation":"x","criteria met":true}') is True
    assert _parse_criteria_met('{"explanation":"x","criteria met":false}') is False


def test_parse_criteria_met_coerces_string_bool():
    assert _parse_criteria_met('{"criteria met":"true","explanation":"x"}') is True
    assert _parse_criteria_met('{"criteria met":"false","explanation":"x"}') is False


def test_parse_criteria_met_fallback_regex():
    assert _parse_criteria_met('noise {"criteria met": true} trailing') is True
    assert _parse_criteria_met('noise "criteria met": false trailing') is False

