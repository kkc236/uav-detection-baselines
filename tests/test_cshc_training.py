def test_screen_mode_uses_five_epochs_and_never_claims_formal_results():
    from scripts.train_rtdetr_cshc import build_parser, build_settings

    settings = build_settings(build_parser().parse_args(["--screen"]))

    assert settings["epochs"] == 5
    assert settings["val"] is True
    assert settings["nms"] is False
    assert settings["max_det"] == 300
