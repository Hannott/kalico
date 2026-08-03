import importlib.util
import pathlib

ROOT = pathlib.Path(__file__).parent.parent
CHECKER_PATH = ROOT / "scripts" / "check_gcode_params.py"


def _load_checker():
    spec = importlib.util.spec_from_file_location(
        "check_gcode_params", CHECKER_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gcode_params_match_handlers():
    # Verify every opt-in "cmd_XXX_params" declaration under klippy/ stays
    # in sync with the gcmd.get()/get_int()/get_float()/get_boolean()
    # calls actually made in the matching cmd_XXX handler. See
    # scripts/check_gcode_params.py and klippy/extras/heaters.py for the
    # convention this enforces.
    checker = _load_checker()
    errors = []
    for path in sorted((ROOT / "klippy").rglob("*.py")):
        errors.extend(checker.check_file(str(path)))
    assert not errors, "\n".join(errors)
