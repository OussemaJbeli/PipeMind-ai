from app.services.log_processor import clean, extract


def test_strips_ansi_and_runner_noise() -> None:
    raw = (
        "\x1b[32;1m$ npm ci\x1b[0m\n"
        "section_start:1756563004:prepare\n"
        "##[group]Run tests\n"
        "2026-08-30T14:30:04.1234Z building\n"
        "Downloading package 45%\n"
        "done\n"
    )

    cleaned = clean(raw)

    assert "\x1b" not in cleaned
    assert "section_start" not in cleaned
    assert "##[group]" not in cleaned
    assert "Downloading" not in cleaned
    # The line prefix goes; the content stays.
    assert "building" in cleaned
    assert "2026-08-30T14:30:04" not in cleaned


def test_keeps_timestamps_inside_messages() -> None:
    # A timestamp in an error is evidence; only the runner's prefix is noise.
    cleaned = clean("ERROR: token expired at 2026-08-30T14:00:00Z\n")

    assert "2026-08-30T14:00:00Z" in cleaned


def test_windows_the_first_error_not_the_last() -> None:
    lines = (
        ["setup line"] * 300
        + ["npm ERR! ERESOLVE unable to resolve dependency tree"]
        + ["noise"] * 300
        + ["ERROR: Job failed: exit code 1"]
    )

    result = extract("\n".join(lines), context=5)

    # A dependency conflict at line 301 causes the "job failed" at 601. Leading
    # with the tail would produce a confident analysis of the symptom.
    assert "ERESOLVE" in result.excerpt
    assert result.ecosystem == "node"


def test_falls_back_to_the_tail_when_nothing_matches() -> None:
    result = extract("\n".join(f"line {i}" for i in range(500)))

    assert result.ecosystem is None
    assert "line 499" in result.excerpt
    assert result.error_message is not None


def test_extracts_exit_code_and_marks_error_lines() -> None:
    result = extract(
        "running tests\nFAIL login.test.ts\nExpected 200 Received 401\n"
        "ERROR: Job failed: exit code 1\n"
    )

    assert result.exit_code == 1
    assert result.matched_lines
    assert result.error_message is not None


def test_caps_a_runaway_error_span() -> None:
    # 400 failing assertions must not produce a 400-line excerpt.
    lines = ["setup"] * 10 + ["AssertionError: boom"] * 800

    result = extract("\n".join(lines), context=5, max_chars=60_000)

    assert result.excerpt.count("\n") < 300
