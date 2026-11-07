"""Run the leak-free backtest and write results/ tables."""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from soyindex import config  # noqa: E402
from soyindex.backtest import run_backtest, summarize  # noqa: E402
from soyindex.data import load_panel  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")


def main() -> None:
    panel = load_panel()
    bt = run_backtest(panel)
    config.RESULTS_DIR.mkdir(exist_ok=True)
    bt.to_csv(config.RESULTS_DIR / "backtest_county_year.csv", index=False)
    summary = summarize(bt).round(4)
    summary.to_csv(config.RESULTS_DIR / "backtest_summary.csv", index=False)
    logging.info("\n%s", summary.to_string(index=False))


if __name__ == "__main__":
    main()
