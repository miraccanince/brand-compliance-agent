from brand_compliance.utils import extract_list, parse_json_response


def test_parse_plain_json():
    assert parse_json_response('{"a": 1}') == {"a": 1}


def test_parse_strips_markdown_fences():
    assert parse_json_response('```json\n{"a": 1}\n```') == {"a": 1}


def test_parse_repairs_malformed_json():
    assert parse_json_response('{"a": 1,}') == {"a": 1}


def test_extract_list_from_list():
    assert extract_list([1, 2]) == [1, 2]


def test_extract_list_from_dict_returns_first_list():
    assert extract_list({"count": 2, "rules": [1, 2]}) == [1, 2]


def test_extract_list_without_list_uses_fallback():
    assert extract_list({"a": 1}) == []
    assert extract_list({"a": 1}, fallback=["x"]) == ["x"]
