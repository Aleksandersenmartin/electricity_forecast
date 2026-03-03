"""Daily data refresh for the electricity forecast dashboard.

Fetches fresh data for all zones and rebuilds feature matrices.
Designed to run in GitHub Actions on a daily schedule.

Usage:
    python scripts/daily_refresh.py
"""

import glob
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.build_features import build_all_zones_feature_matrix

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

START_DATE = "2020-01-01"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def delete_stale_feature_files() -> int:
    """Delete old feature parquet files to avoid accumulation.

    Returns:
        Number of files deleted.
    """
    pattern = str(PROCESSED_DIR / "features_*.parquet")
    old_files = glob.glob(pattern)
    for f in old_files:
        os.remove(f)
        logger.info("Deleted stale file: %s", f)
    return len(old_files)


def main() -> None:
    """Run the daily data refresh pipeline."""
    yesterday = datetime.now().date() - timedelta(days=1)
    end_date = yesterday.strftime("%Y-%m-%d")

    logger.info("=" * 60)
    logger.info("DAILY DATA REFRESH")
    logger.info("Period: %s to %s", START_DATE, end_date)
    logger.info("=" * 60)

    # Step 1: Delete stale feature files
    deleted = delete_stale_feature_files()
    logger.info("Deleted %d stale feature file(s)", deleted)

    # Step 2: Build feature matrices for all zones
    logger.info("Building feature matrices for all zones...")
    results = build_all_zones_feature_matrix(START_DATE, end_date)

    # Step 3: Log summary
    logger.info("=" * 60)
    logger.info("REFRESH COMPLETE")
    logger.info("Zones built: %d/5", len(results))
    for zone, df in results.items():
        file_path = PROCESSED_DIR / f"features_{zone}_{START_DATE}_{end_date}.parquet"
        size_mb = file_path.stat().st_size / (1024 * 1024) if file_path.exists() else 0
        logger.info(
            "  %s: %s rows x %d cols, %.1f MB",
            zone, f"{df.shape[0]:,}", df.shape[1], size_mb,
        )
    logger.info("=" * 60)

    if len(results) == 0:
        logger.error("No zones were built successfully!")
        sys.exit(1)


if __name__ == "__main__":
    main()
