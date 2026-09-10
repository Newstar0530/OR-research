"""A model-written script is accepted for what it still guarantees, not for who wrote it.

Each test here is one way a rewrite can pass every *other* check and still
destroy the run: it compiles but drops the results table, it fixes the error by
deleting the experiment, it reaches the network under a name a string search
would miss.
"""

import pytest

from src.core.generated_code_guard import (
    extract_code,
    validate_generated_experiment,
)


GOOD = '''"""A small experiment."""
from pathlib import Path
import json
import pandas as pd

def main() -> None:
    out = Path(__file__).resolve().parent
    rows = [{"instance_id": "i1", "method": "baseline", "objective": 10.0}]
    pd.DataFrame(rows).to_csv(out / "results.csv", index=False)
    print("SUMMARY_JSON:" + json.dumps({"status": "success"}))

if __name__ == "__main__":
    main()
'''


def test_a_faithful_rewrite_is_accepted() -> None:
    verdict = validate_generated_experiment(GOOD, GOOD)
    assert verdict.ok
    assert verdict.violations == []
    assert any("kept" in note for note in verdict.notes)


def test_the_feedback_is_written_to_be_handed_back_to_the_model() -> None:
    verdict = validate_generated_experiment("import socket\n" + GOOD, GOOD)
    feedback = verdict.to_feedback()
    assert "rejected" in feedback
    assert "socket" in feedback
    assert validate_generated_experiment(GOOD, GOOD).to_feedback() == ""


# -- the ways a rewrite breaks the run -------------------------------------


def test_a_rewrite_that_does_not_parse_is_refused_with_the_line(  ) -> None:
    verdict = validate_generated_experiment("def main(:\n    pass\n", GOOD)
    assert not verdict.ok
    assert "does not parse" in verdict.violations[0]
    assert "line" in verdict.violations[0]


@pytest.mark.parametrize("marker", ["results.csv", "SUMMARY_JSON"])
def test_dropping_a_required_output_is_refused(marker: str) -> None:
    """Both are what make a node measurable; a run without them is not a result."""

    verdict = validate_generated_experiment(GOOD.replace(marker, "gone"), GOOD)
    assert not verdict.ok
    assert any(marker in item for item in verdict.violations)


def test_fixing_the_error_by_deleting_the_experiment_is_refused() -> None:
    """This passes syntax, and would pass the contract if it wrote an empty table."""

    stub = 'import pandas as pd\npd.DataFrame([]).to_csv("results.csv")\nprint("SUMMARY_JSON:{}")\n'
    verdict = validate_generated_experiment(stub, GOOD)
    assert not verdict.ok
    assert any("does not delete the experiment" in item for item in verdict.violations)


def test_a_network_import_is_caught_on_the_parse_tree_not_by_spelling() -> None:
    """`from urllib import request as r` never contains the string `import urllib`."""

    sneaky = GOOD.replace(
        "import pandas as pd", "import pandas as pd\nfrom urllib import request as r"
    )
    verdict = validate_generated_experiment(sneaky, GOOD)
    assert not verdict.ok
    assert any("urllib" in item for item in verdict.violations)


@pytest.mark.parametrize(
    "line", ["import subprocess", "import socket", "import shutil", "import multiprocessing"]
)
def test_escaping_the_sandbox_is_refused(line: str) -> None:
    verdict = validate_generated_experiment(GOOD.replace("import json", line + "\nimport json"), GOOD)
    assert not verdict.ok


@pytest.mark.parametrize("snippet", ['eval("1+1")', 'exec("x=1")', 'os.system("ls")'])
def test_executing_text_or_shelling_out_is_refused(snippet: str) -> None:
    verdict = validate_generated_experiment(GOOD + "\n" + snippet + "\n", GOOD)
    assert not verdict.ok
    assert any("execute generated text" in item for item in verdict.violations)


@pytest.mark.parametrize("path", ["/etc/passwd", "C:\\\\Windows\\\\system32", "../../secrets.csv"])
def test_writing_outside_the_workspace_is_refused(path: str) -> None:
    verdict = validate_generated_experiment(GOOD.replace('"i1"', repr(path)), GOOD)
    assert not verdict.ok
    assert any("leave the node workspace" in item for item in verdict.violations)


def test_a_relative_path_inside_the_workspace_is_fine() -> None:
    verdict = validate_generated_experiment(GOOD.replace('"i1"', '"figures/plot.png"'), GOOD)
    assert verdict.ok


# -- the entry points ------------------------------------------------------


def test_an_empty_response_is_refused_rather_than_written() -> None:
    assert validate_generated_experiment("").violations == ["The response contained no code."]
    assert validate_generated_experiment("   \n ").ok is False


def test_code_is_extracted_from_a_fenced_response() -> None:
    response = "Here is the fix:\n\n```python\nx = 1\ny = 2\n```\n\nHope that helps."
    assert extract_code(response) == "x = 1\ny = 2"


def test_the_longest_block_wins_when_a_model_shows_a_snippet_first() -> None:
    response = "```python\nrows = []\n```\nand the full file:\n```python\nimport json\nrows = []\nprint(rows)\n```"
    assert extract_code(response) == "import json\nrows = []\nprint(rows)"


def test_an_unfenced_response_is_taken_as_the_code() -> None:
    assert extract_code("x = 1\n") == "x = 1"


def test_the_contract_check_can_be_switched_off_for_non_experiment_code() -> None:
    verdict = validate_generated_experiment("x = 1\n", require_contract=False)
    assert verdict.ok
