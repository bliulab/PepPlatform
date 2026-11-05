from pepplatform.profile_progress import ProfileProgressReporter, emit_profile_progress


def test_emit_profile_progress_uses_single_line_parseable_records(capsys):
    emit_profile_progress("profiles", "running", current=7, total=12, feature="pssm+psfm")

    assert capsys.readouterr().out == (
        "[profile-progress] stage=profiles status=running current=7 total=12 feature=pssm+psfm\n"
    )


def test_profile_progress_reporter_emits_start_progress_and_completion(capsys):
    reporter = ProfileProgressReporter(10, "pssm", clock=lambda: 0.0)

    reporter.start()
    reporter.update(1)
    reporter.complete()

    assert capsys.readouterr().out.splitlines() == [
        "[profile-progress] stage=profiles status=running current=0 total=10 feature=pssm elapsed_seconds=0.000",
        "[profile-progress] stage=profiles status=running current=1 total=10 feature=pssm elapsed_seconds=0.000",
        "[profile-progress] stage=profiles status=complete current=10 total=10 feature=pssm elapsed_seconds=0.000",
    ]


def test_profile_progress_reporter_supports_database_stage(capsys):
    ticks = iter([4.0, 7.5])
    reporter = ProfileProgressReporter(
        1,
        "pssm_blosum62_hybrid",
        stage="database",
        clock=lambda: next(ticks),
    )

    reporter.start()
    reporter.complete()

    assert capsys.readouterr().out.splitlines() == [
        "[profile-progress] stage=database status=running current=0 total=1 "
        "feature=pssm_blosum62_hybrid elapsed_seconds=0.000",
        "[profile-progress] stage=database status=complete current=1 total=1 "
        "feature=pssm_blosum62_hybrid elapsed_seconds=3.500",
    ]
