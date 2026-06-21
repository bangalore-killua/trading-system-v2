"""
chart_pattern_detector.py — YOLOv8 Visual Chart Pattern Detection (Phase 2)

Detects 15 classic chart patterns from OHLCV price data by:
  1. Rendering the last N bars as a candlestick PNG (via mplfinance)
  2. Running YOLOv8 inference on that image
  3. Returning binary flags for each pattern detected

Why computer vision instead of rule-based detection:
  - Patterns like H&S or cup-and-handle span many bars and require holistic
    shape recognition — hard to capture with simple if/else rules
  - YOLOv8 sees the whole chart the same way a human analyst does
  - Each detected pattern becomes a binary feature in the LightGBM feature vector

Model: fine-tuned YOLOv8n (nano — fast, ~6MB) stored at models/chart_patterns_yolov8n.pt
Training data: public annotated datasets from Roboflow (search "stock chart patterns")
  Recommended dataset: "Stock Chart Pattern" by Roboflow Universe (~2000 annotated images)

If model file does not exist: returns all-zero pattern dict (graceful degradation).
The system still trades using rule-based signals — YOLOv8 just adds extra features.

Setup (run once to download a starter model or start training):
    python -m strategy.chart_pattern_detector --setup
"""

import io
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Pattern class names (must match the YOLOv8 model's training labels) ───────
PATTERN_CLASSES: List[str] = [
    "head_and_shoulders",        # bearish reversal
    "inv_head_and_shoulders",    # bullish reversal
    "double_top",                # bearish reversal
    "double_bottom",             # bullish reversal
    "ascending_triangle",        # bullish continuation / breakout
    "descending_triangle",       # bearish continuation / breakdown
    "symmetrical_triangle",      # neutral — breakout in either direction
    "bull_flag",                 # bullish continuation
    "bear_flag",                 # bearish continuation
    "pennant",                   # continuation (direction from prior trend)
    "cup_and_handle",            # bullish continuation
    "rising_wedge",              # bearish reversal
    "falling_wedge",             # bullish reversal
    "rounding_bottom",           # bullish reversal (also called saucer)
    "channel_up",                # bullish trend channel
    "channel_down",              # bearish trend channel
]

# Bullish vs bearish classification for each pattern (used by signal basket)
BULLISH_PATTERNS = {
    "inv_head_and_shoulders", "double_bottom", "ascending_triangle",
    "bull_flag", "pennant", "cup_and_handle", "falling_wedge",
    "rounding_bottom", "channel_up",
}
BEARISH_PATTERNS = {
    "head_and_shoulders", "double_top", "descending_triangle",
    "bear_flag", "rising_wedge", "channel_down",
}


def _empty_patterns() -> Dict[str, int]:
    """Return all-zero pattern dict — used when model unavailable or data too short."""
    return {f"chart_{p}": 0 for p in PATTERN_CLASSES}


class ChartPatternDetector:
    """YOLOv8-based visual chart pattern detector.

    The model is lazy-loaded on first call to `detect()`. If the model file
    doesn't exist, all patterns return 0 and a warning is logged once.
    """

    # Image size expected by YOLOv8 (height, width)
    IMG_SIZE = 640

    def __init__(
        self,
        model_path: str = "models/chart_patterns_yolov8n.pt",
        confidence_threshold: float = 0.40,
        lookback_bars: int = 60,
    ):
        self._model_path = Path(model_path)
        self._conf = confidence_threshold
        self._lookback = lookback_bars
        self._model = None
        self._model_warned = False  # log missing-model warning only once

    def detect(self, df: pd.DataFrame) -> Dict[str, int]:
        """Detect chart patterns in the last `lookback_bars` of OHLCV data.

        Args:
            df: DataFrame with columns open/high/low/close/volume, date index.

        Returns:
            Dict: {chart_<pattern_name>: 0 or 1} for all 16 patterns.
            Returns all zeros if model unavailable or data too short.
        """
        if len(df) < 30:
            return _empty_patterns()

        # Render chart image
        img_array = self._render_chart(df.tail(self._lookback))
        if img_array is None:
            return _empty_patterns()

        # Load model (lazy, first call only)
        model = self._get_model()
        if model is None:
            return _empty_patterns()

        # Run inference
        try:
            results = model(img_array, conf=self._conf, verbose=False)
            return self._parse_results(results)
        except Exception:
            logger.exception("YOLOv8 inference error")
            return _empty_patterns()

    def detect_bullish(self, df: pd.DataFrame) -> bool:
        """Quick check: any bullish chart pattern present?"""
        patterns = self.detect(df)
        return any(patterns.get(f"chart_{p}", 0) == 1 for p in BULLISH_PATTERNS)

    def detect_bearish(self, df: pd.DataFrame) -> bool:
        """Quick check: any bearish chart pattern present?"""
        patterns = self.detect(df)
        return any(patterns.get(f"chart_{p}", 0) == 1 for p in BEARISH_PATTERNS)

    def _render_chart(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """Render OHLCV data as a 640×640 candlestick image (no axes, clean for CV).

        Returns numpy array (H, W, C) in RGB or None on failure.
        """
        try:
            import mplfinance as mpf  # type: ignore
            import matplotlib
            matplotlib.use("Agg")  # non-interactive backend, no display needed
            import matplotlib.pyplot as plt
            from PIL import Image  # type: ignore

            # mplfinance expects DatetimeIndex
            plot_df = df.copy()
            if not isinstance(plot_df.index, pd.DatetimeIndex):
                plot_df.index = pd.to_datetime(plot_df.index)

            # Ensure standard column names
            plot_df.columns = [c.lower() for c in plot_df.columns]
            for col in ("open", "high", "low", "close", "volume"):
                if col not in plot_df.columns:
                    return None

            buf = io.BytesIO()

            # Minimal style: no axes labels, tight layout — cleaner for YOLOv8
            mpf.plot(
                plot_df,
                type="candle",
                style="charles",
                axisoff=True,           # remove axes
                tight_layout=True,
                savefig=dict(fname=buf, format="png", dpi=80, bbox_inches="tight"),
                figsize=(8, 8),
                volume=False,           # exclude volume subplot — patterns are price-only
            )
            plt.close("all")

            buf.seek(0)
            img = Image.open(buf).convert("RGB").resize(
                (self.IMG_SIZE, self.IMG_SIZE), Image.LANCZOS
            )
            return np.array(img)

        except ImportError as e:
            logger.error("Chart rendering requires mplfinance + Pillow: %s", e)
            return None
        except Exception:
            logger.exception("Chart rendering error")
            return None

    def _get_model(self):
        """Lazy-load the YOLOv8 model. Returns None if file doesn't exist."""
        if self._model is not None:
            return self._model

        if not self._model_path.exists():
            if not self._model_warned:
                logger.warning(
                    "YOLOv8 chart pattern model not found at %s. "
                    "Chart pattern features will be all-zero until model is trained. "
                    "See PLAN.md Phase 2 section for training instructions.",
                    self._model_path,
                )
                self._model_warned = True
            return None

        try:
            from ultralytics import YOLO  # type: ignore

            self._model = YOLO(str(self._model_path))
            logger.info("YOLOv8 chart pattern model loaded from %s", self._model_path)
            return self._model
        except Exception:
            logger.exception("Failed to load YOLOv8 model from %s", self._model_path)
            return None

    def _parse_results(self, results) -> Dict[str, int]:
        """Convert YOLOv8 Results object to binary pattern flags dict."""
        detected = _empty_patterns()

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                class_id = int(box.cls[0])
                if 0 <= class_id < len(PATTERN_CLASSES):
                    pattern_name = PATTERN_CLASSES[class_id]
                    detected[f"chart_{pattern_name}"] = 1

        if any(v == 1 for v in detected.values()):
            found = [k for k, v in detected.items() if v == 1]
            logger.debug("Chart patterns detected: %s", found)

        return detected

    def train(
        self,
        dataset_yaml: str,
        epochs: int = 50,
        imgsz: int = 640,
        batch: int = 16,
    ) -> None:
        """Fine-tune YOLOv8n on an annotated chart pattern dataset.

        Args:
            dataset_yaml: path to Roboflow-exported dataset.yaml
            epochs: training epochs (50 is usually enough for this task)
            imgsz: image size (must match IMG_SIZE = 640)
            batch: batch size (reduce if GPU OOM)

        Example:
            detector = ChartPatternDetector()
            detector.train("datasets/chart_patterns/data.yaml")
        """
        try:
            from ultralytics import YOLO  # type: ignore

            # Start from YOLOv8n pretrained on COCO (best starting point)
            model = YOLO("yolov8n.pt")
            model.train(
                data=dataset_yaml,
                epochs=epochs,
                imgsz=imgsz,
                batch=batch,
                project="models",
                name="chart_patterns_yolov8n",
                exist_ok=True,
            )
            logger.info("Training complete. Model saved to models/chart_patterns_yolov8n/")
        except Exception:
            logger.exception("YOLOv8 training failed")


if __name__ == "__main__":
    import sys

    if "--setup" in sys.argv:
        print("YOLOv8 chart pattern setup:")
        print("1. Install: pip install ultralytics mplfinance Pillow")
        print("2. Download a chart pattern dataset from Roboflow:")
        print("   https://universe.roboflow.com/search?q=stock+chart+pattern")
        print("3. Export dataset in YOLOv8 format, unzip to datasets/chart_patterns/")
        print("4. Run training:")
        print("   from strategy.chart_pattern_detector import ChartPatternDetector")
        print("   ChartPatternDetector().train('datasets/chart_patterns/data.yaml')")
        print("5. Copy best weights to models/chart_patterns_yolov8n.pt")
