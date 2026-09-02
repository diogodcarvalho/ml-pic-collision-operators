import pytest


@pytest.fixture(autouse=True, scope="session")
def run_from_temp_directory(tmp_path_factory):
    """Run the whole suite from a temporary working directory.

    MLflow's SqlAlchemyStore creates its default artifact root ("./mlruns", relative
    to the current working directory) as soon as a tracking client is built, and
    writes artifacts there for any experiment created without an explicit
    artifact_location. Tests that only set a tracking URI therefore leave run
    artifacts wherever pytest happened to be started, which is normally the
    repository root. Every path used by the tests is derived from `__file__` or from
    `tmp_path`, so moving the working directory keeps those droppings out of the
    repository without affecting anything else.
    """
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.chdir(tmp_path_factory.mktemp("cwd"))
    yield
    monkeypatch.undo()
