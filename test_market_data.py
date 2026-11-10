import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from market_data import download_raw


class DownloadTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.tickers = {"hyundai": "005380.KS", "gm": "GM"}
        self.start, self.end = "2025-01-01", "2025-01-31"
        dates = pd.to_datetime(["2025-01-02", "2025-01-03"])
        self.quotes = pd.DataFrame(
            {("Close", "GM"): [100.0, 102.0]}, index=dates,
        )
        self.fx = pd.DataFrame({
            "observation_date": ["2025-01-02", "2025-01-03", "2025-01-06"],
            "DEXKOUS": [1400.0, np.nan, 1405.0],
        })
        for name in self.tickers:
            pd.DataFrame({"Date": dates, "Close": [100.0, 102.0]}).to_csv(
                self.directory / f"{name}.csv", index=False,
            )
        self.fx.to_csv(self.directory / "fx.csv", index=False)
        quotes_patch = patch("market_data.yf.download", return_value=self.quotes)
        self.download = quotes_patch.start()
        self.addCleanup(quotes_patch.stop)
        self.fred = Mock(return_value=self.fx.copy())
        read_csv = pd.read_csv

        def read(path, *args, **kwargs):
            if str(path).startswith("https://"):
                return self.fred(path, *args, **kwargs)
            return read_csv(path, *args, **kwargs)

        read_patch = patch("market_data.pd.read_csv", side_effect=read)
        read_patch.start()
        self.addCleanup(read_patch.stop)

    def run_download(self):
        download_raw(self.directory, self.tickers, self.start, self.end)

    def snapshot(self):
        return {p.name: p.read_bytes() for p in self.directory.iterdir()}

    def test_valid_caches_are_unchanged_and_need_no_network(self):
        before = self.snapshot()
        self.run_download()
        self.assertEqual(before, self.snapshot())
        self.download.assert_not_called()
        self.fred.assert_not_called()

    def test_only_missing_file_is_downloaded(self):
        (self.directory / "gm.csv").unlink()
        before = self.snapshot()
        self.run_download()
        self.assertEqual(self.download.call_count, 1)
        self.assertEqual(self.download.call_args.args, ("GM",))
        self.assertTrue(self.download.call_args.kwargs["auto_adjust"])
        self.fred.assert_not_called()
        for name, data in before.items():
            self.assertEqual((self.directory / name).read_bytes(), data)
        self.assertEqual(pd.read_csv(self.directory / "gm.csv")["Close"].tolist(),
                         [100.0, 102.0])

    def test_invalid_cache_is_replaced_only_after_success(self):
        path = self.directory / "gm.csv"
        path.write_text("Date,Close\nbad,old\n")
        self.run_download()
        self.assertEqual(pd.read_csv(path)["Close"].tolist(), [100.0, 102.0])
        self.assertEqual(sorted(p.name for p in self.directory.iterdir()),
                         ["fx.csv", "gm.csv", "hyundai.csv"])

    def test_invalid_responses_preserve_existing_files(self):
        invalid = [self.quotes.iloc[:0], self.quotes.iloc[:1],
                   self.quotes.rename(columns={"Close": "Other"}, level=0)]
        for prices in ([0, 102], [-1, 102], [np.inf, 102], [np.nan, 102], ["bad", 102]):
            frame = self.quotes.copy()
            frame[frame.columns[0]] = prices
            invalid.append(frame)
        for dates in (["bad", "date"], ["2025-01-02", "2025-01-02"],
                      ["2025-01-03", "2025-01-02"], ["2024-12-30", "2025-01-02"]):
            frame = self.quotes.copy()
            frame.index = dates
            invalid.append(frame)
        for i, response in enumerate(invalid):
            with self.subTest(response=i):
                (self.directory / "gm.csv").write_text("Date,Close\nbad,old\n")
                before = self.snapshot()
                self.download.return_value = response
                with self.assertRaisesRegex(RuntimeError, "Could not download gm"):
                    self.run_download()
                self.assertEqual(before, self.snapshot())

    def test_network_failure_leaves_missing_file_missing(self):
        (self.directory / "gm.csv").unlink()
        before = self.snapshot()
        self.download.side_effect = TimeoutError("provider unavailable")
        with self.assertRaisesRegex(RuntimeError, "Could not download gm"):
            self.run_download()
        self.assertEqual(before, self.snapshot())

    def test_write_or_replace_failure_preserves_existing_file_and_cleans_up(self):
        for target in ("market_data.pd.DataFrame.to_csv", "market_data.Path.replace"):
            with self.subTest(operation=target):
                (self.directory / "gm.csv").write_text("Date,Close\nbad,old\n")
                before = self.snapshot()
                with patch(target, side_effect=OSError("write failed")):
                    with self.assertRaisesRegex(RuntimeError, "Could not download gm"):
                        self.run_download()
                self.assertEqual(before, self.snapshot())

    def test_fred_download_allows_holiday_gaps_and_rejects_empty_data(self):
        path = self.directory / "fx.csv"
        path.unlink()
        self.run_download()
        self.download.assert_not_called()
        self.assertEqual(self.fred.call_count, 1)
        self.assertTrue(pd.isna(pd.read_csv(path).loc[1, "DEXKOUS"]))
        path.write_text("observation_date,DEXKOUS\nbad,old\n")
        before = self.snapshot()
        self.fred.return_value = self.fx.iloc[:0]
        with self.assertRaisesRegex(RuntimeError, "Could not download fx"):
            self.run_download()
        self.assertEqual(before, self.snapshot())


if __name__ == "__main__":
    unittest.main()
