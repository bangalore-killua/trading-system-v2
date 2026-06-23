"""
pattern_api.py — YOLOv8 chart inference API wrapper.

Takes a ticker symbol, downloads its price chart (3 months default), runs
YOLOv8 inference on the chart image, and returns a structured list of
detected patterns with trading implications.

The YOLOv8 model is already trained and saved at
models/chart_patterns_yolov8n.pt (16 patterns).

Pattern → implication mapping:
  Head_and_Shoulders_Top       → Bearish reversal — expect ~8% downside
  Inverse_Head_and_Shoulders   → Bullish reversal — expect ~8% upside
  Double_Top                   → Bearish reversal
  Double_Bottom                → Bullish reversal
  Cup_and_Handle               → Bullish continuation — classic breakout
  Ascending_Triangle           → Bullish continuation
  Descending_Triangle          → Bearish continuation
  Symmetrical_Triangle         → Neutral — wait for breakout direction
  Bullish_Flag                 → Bullish continuation
  Bearish_Flag                 → Bearish continuation
  Rising_Wedge                 → Bearish reversal
  Falling_Wedge                → Bullish reversal
  Rounding_Bottom              → Bullish accumulation
  Rounding_Top                 → Bearish distribution
  Channel_Up                   → Bullish trend
  Channel_Down                 → Bearish trend
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = "models/chart_patterns_yolov8n.pt"
CHART_TMP_DIR = Path("logs/charts")
DEFAULT_CONFIDENCE = 0.40

# Pattern → (direction, implication_text)
PATTERN_IMPLICATIONS: Dict[str, tuple] = {
    "Head_and_Shoulders_Top":      ("bearish",  "Bearish reversal — expect ~8% downside"),
    "Inverse_Head_and_Shoulders":  ("bullish",  "Bullish reversal — expect ~8% upside"),
    "Double_Top":                  ("bearish",  "Bearish reversal — resistance rejected twice"),
    "Double_Bottom":               ("bullish",  "Bullish reversal — support held twice"),
    "Cup_and_Handle":              ("bullish",  "Bullish continuation — classic breakout setup"),
    "Ascending_Triangle":          ("bullish",  "Bullish continuation — higher lows building pressure"),
    "Descending_Triangle":         ("bearish",  "Bearish continuation — lower highs, flat support"),
    "Symmetrical_Triangle":        ("neutral",  "Neutral — wait for breakout direction"),
    "Bullish_Flag":                ("bullish",  "Bullish continuation — strong trend consolidating"),
    "Bearish_Flag":                ("bearish",  "Bearish continuation — weak bounce before next leg"),
    "Rising_Wedge":                ("bearish",  "Bearish reversal — diminishing buying pressure"),
    "Falling_Wedge":               ("bullish",  "Bullish reversal — diminishing selling pressure"),
    "Rounding_Bottom":             ("bullish",  "Bullish accumulation — slow base building"),
    "Rounding_Top":                ("bearish",  "Bearish distribution — slow topping process"),
    "Channel_Up":                  ("bullish",  "Bullish trend — buy on pullbacks to lower channel"),
    "Channel_Down":                ("bearish",  "Bearish trend — sell rallies to upper channel"),
}


def detect_patterns(
    symbol: str,
    period: str = "3mo",
    confidence: float = DEFAULT_CONFIDENCE,
    model_path: str = DEFAULT_MODEL_PATH,
) -> List[dict]:
    """Download chart, run YOLOv8 inference, return pattern detections.

    Parameters
    ----------
    symbol     : equity ticker
    period     : chart lookback period (yfinance format: 1mo, 3mo, 6mo, 1y)
    confidence : minimum detection confidence (0.0–1.0)
    model_path : path to .pt model file

    Returns list of dicts:
      {pattern, confidence, direction, implication, bbox: [x1,y1,x2,y2]}
    """
    chart_path = _generate_chart(symbol, period)
    if chart_path is None:
        return []

    model = _load_model(model_path)
    if model is None:
        return []

    return _run_inference(model, chart_path, confidence)


def _generate_chart(symbol: str, period: str) -> Optional[Path]:
    """Generate a candlestick chart image and return its path."""
    CHART_TMP_DIR.mkdir(parents=True, exist_ok=True)
    chart_path = CHART_TMP_DIR / f"{symbol}_{period}.png"

    try:
        import yfinance as yf  # type: ignore
        import mplfinance as mpf  # type: ignore

        hist = yf.Ticker(symbol).history(period=period)
        if hist is None or len(hist) < 20:
            logger.debug("Insufficient price history for %s (%s)", symbol, period)
            return None

        # Style matching dashboard dark theme
        mc = mpf.make_marketcolors(
            up="#00ff88", down="#ff4444",
            wick={"up": "#00ff88", "down": "#ff4444"},
            edge="inherit",
        )
        style = mpf.make_mpf_style(
            base_mpf_style="nightclouds",
            marketcolors=mc,
            figcolor="#0a0a0f",
            facecolor="#0a0a0f",
        )

        mpf.plot(
            hist,
            type="candle",
            style=style,
            savefig=str(chart_path),
            figsize=(10, 6),
            tight_layout=True,
            volume=False,
            show_nontrading=False,
        )
        return chart_path
    except Exception:
        logger.exception("Chart generation failed for %s", symbol)
        return None


_model_cache: Optional[object] = None


def _load_model(model_path: str):
    """Lazy-load the YOLOv8 model (cached module-level)."""
    global _model_cache
    if _model_cache is not None:
        return _model_cache

    if not os.path.exists(model_path):
        logger.warning("YOLOv8 model not found at %s — pattern detection disabled", model_path)
        return None

    try:
        from strategy.chart_pattern_detector import ChartPatternDetector  # type: ignore
        _model_cache = ChartPatternDetector(model_path)
        logger.info("YOLOv8 chart pattern model loaded: %s", model_path)
        return _model_cache
    except ImportError:
        logger.warning("ultralytics not installed — pattern detection disabled")
        return None
    except Exception:
        logger.exception("Failed to load YOLOv8 model from %s", model_path)
        return None


def _run_inference(model, chart_path: Path, confidence: float) -> List[dict]:
    """Run YOLOv8 inference and map results to pattern dicts."""
    try:
        # ChartPatternDetector.detect() returns list of (label, conf, bbox)
        detections = model.detect(str(chart_path), confidence_threshold=confidence)
        results = []
        for det in detections:
            label = det.get("label", "") if isinstance(det, dict) else str(det)
            conf  = det.get("confidence", 0.0) if isinstance(det, dict) else confidence
            bbox  = det.get("bbox", []) if isinstance(det, dict) else []

            direction, implication = PATTERN_IMPLICATIONS.get(
                label, ("neutral", "Pattern detected — no specific implication")
            )
            results.append({
                "pattern":     label,
                "confidence":  round(float(conf), 3),
                "direction":   direction,
                "implication": implication,
                "bbox":        bbox,
            })

        # Sort by confidence descending
        results.sort(key=lambda x: x["confidence"], reverse=True)
        return results
    except Exception:
        logger.exception("YOLOv8 inference failed")
        return []


if __name__ == "__main__":
    import sys
    sym = sys.argv[1] if len(sys.argv) > 1 else "NVDA"
    patterns = detect_patterns(sym)
    if patterns:
        for p in patterns:
            print(f"{p['pattern']} ({p['confidence']:.1%}) — {p['implication']}")
    else:
        print("No patterns detected (model may not be loaded or chart unavailable)")
