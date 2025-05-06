import pytest
from speedy_utils.common.utils_print import flatten_dict, print_table

def test_flatten_dict_simple():
    data = {"a": 1, "b": {"c": 2, "d": {"e": 3}}}
    assert flatten_dict(data) == {"a": 1, "b.c": 2, "b.d.e": 3}

def test_flatten_dict_empty():
    assert flatten_dict({}) == {}

def test_print_table_dict_console(capsys):
    data = {"x": 1, "y": 2}
    print_table(data, use_html=False)
    captured = capsys.readouterr()
    assert "x" in captured.out and "1" in captured.out

def test_print_table_list_of_dicts_console(capsys):
    data = [{"k": "v"}, {"k": "w"}]
    print_table(data, use_html=False)
    captured = capsys.readouterr()
    assert "k" in captured.out and "v" in captured.out and "w" in captured.out

def test_print_table_invalid_data():
    with pytest.raises(TypeError):
        print_table(123, use_html=False)
