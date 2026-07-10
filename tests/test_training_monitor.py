from pathlib import Path

from src.training_monitor import estimate_eta_seconds, read_latest_result


def test_read_latest_result_parses_last_epoch(tmp_path: Path):
    csv_path = tmp_path / "results.csv"
    csv_path.write_text(
        "\n".join(
            [
                "epoch,time,metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B)",
                "1,10,0.1,0.2,0.3,0.4",
                "2,25,0.5,0.6,0.7,0.8",
            ]
        ),
        encoding="utf-8",
    )

    latest = read_latest_result(csv_path)

    assert latest is not None
    assert latest["epoch"] == 2.0
    assert latest["metrics/mAP50(B)"] == 0.7


def test_estimate_eta_seconds_uses_average_epoch_time():
    latest = {"epoch": 25.0, "time": 1000.0}

    assert estimate_eta_seconds(latest, total_epochs=100) == 3000.0
