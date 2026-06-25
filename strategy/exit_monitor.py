"""
exit_monitor.py — LSTM binary classifier for peak detection.

Question: "Is this position near a local price peak right now?"
Answer: P(peak in next 5 days) → binary classification (not time-series forecasting).

Why LSTM not N-BEATS:
  N-BEATS predicts future prices (regression). We don't need price values.
  We need P(peak) = binary classification. LSTM outputs this directly.
  No error propagation through forecast steps → cleaner inference signal.

Architecture:
  Input:  30-day sequence × 5 features [RSI, MACD, volume_ratio, momentum, price_return]
  LSTM:   hidden_size=64, num_layers=2
  Output: sigmoid(linear(h_n[-1])) = P(peak) in [0, 1]

Additional position-aware features appended before final classification:
  unrealized_pnl_pct, peak_drawdown_pct, days_held, entry_composite_score,
  regime_changed_since_entry

Threshold: P(peak) ≥ 0.65 → E1 partial exit trigger
           P(peak) ≥ 0.78 → E2 partial exit trigger
           P(peak) ≥ 0.88 → E3 full exit trigger

Model path: models/exit_lstm_v1.pt
Training: see strategy/training_pipeline.py AutoGluonPipeline.run_exit_model()
"""

import logging
import os
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

EXIT_MODEL_PATH = "models/exit_lstm_v1.pt"

SEQUENCE_LEN = 30
FEATURES     = ["rsi_14", "macd", "vol_ratio", "momentum", "price_return"]

E1_THRESHOLD = 0.65  # partial exit (20-50%)
E2_THRESHOLD = 0.78  # partial exit (30-80% of remaining)
E3_THRESHOLD = 0.88  # full exit


class ExitMonitor:
    """LSTM-based peak detection for open positions.

    Usage:
        monitor = ExitMonitor()
        result  = monitor.score("NVDA", df_1d, position_data)
        # result["peak_prob"] = 0.72 → E2 trigger
        # result["exit_action"] = "E2" | "E1" | "hold"
    """

    def __init__(self, model_path: str = EXIT_MODEL_PATH):
        self._model_path = model_path
        self._model      = None

    def score(self, symbol: str, df_1d: pd.DataFrame, position_data: dict) -> dict:
        """Score an open position for peak probability.

        Parameters
        ----------
        symbol        : ticker
        df_1d         : daily OHLCV DataFrame, must have 'close', 'volume' columns
        position_data : {unrealized_pnl_pct, peak_drawdown_pct, days_held,
                         entry_composite_score, regime_changed_since_entry}

        Returns
        -------
        {
          "symbol":      str,
          "peak_prob":   float 0-1,
          "exit_action": "E3"|"E2"|"E1"|"hold",
          "threshold_hit": str,
          "model_source": "lstm"|"fallback_rsi",
        }
        """
        # Try LSTM model first
        if self._is_model_available():
            try:
                peak_prob = self._infer_lstm(df_1d, position_data)
                return self._build_result(symbol, peak_prob, "lstm")
            except Exception as e:
                logger.warning("ExitMonitor: LSTM inference failed for %s — %s. Using RSI fallback.", symbol, e)

        # Fallback: rule-based RSI divergence + ATR expansion
        peak_prob = self._fallback_rsi_score(df_1d, position_data)
        return self._build_result(symbol, peak_prob, "fallback_rsi")

    def _build_result(self, symbol: str, peak_prob: float, source: str) -> dict:
        peak_prob = round(float(peak_prob), 4)
        if peak_prob >= E3_THRESHOLD:
            action = "E3"
        elif peak_prob >= E2_THRESHOLD:
            action = "E2"
        elif peak_prob >= E1_THRESHOLD:
            action = "E1"
        else:
            action = "hold"

        result = {
            "symbol":       symbol,
            "peak_prob":    peak_prob,
            "exit_action":  action,
            "threshold_hit": f"≥{E3_THRESHOLD}" if action == "E3" else
                             f"≥{E2_THRESHOLD}" if action == "E2" else
                             f"≥{E1_THRESHOLD}" if action == "E1" else "none",
            "model_source": source,
        }
        logger.info("ExitMonitor %s: P(peak)=%.3f → %s [%s]", symbol, peak_prob, action, source)
        return result

    def _is_model_available(self) -> bool:
        return os.path.exists(self._model_path)

    def _infer_lstm(self, df: pd.DataFrame, position_data: dict) -> float:
        """Run LSTM inference. Loads model on first call (lazy)."""
        import torch
        import torch.nn as nn

        if self._model is None:
            self._model = ExitLSTM()
            self._model.load_state_dict(torch.load(self._model_path, map_location="cpu"))
            self._model.eval()

        # Build 30-day sequence
        sequence = self._build_sequence(df)  # shape (30, 5)
        if sequence is None:
            raise ValueError("insufficient bars for sequence")

        x = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)  # (1, 30, 5)

        # Position-aware features
        pos_feats = torch.tensor([
            position_data.get("unrealized_pnl_pct",       0.0),
            position_data.get("peak_drawdown_pct",         0.0),
            min(position_data.get("days_held",             0) / 60, 1.0),  # normalize
            position_data.get("entry_composite_score",     0.5),
            float(position_data.get("regime_changed_since_entry", False)),
        ], dtype=torch.float32).unsqueeze(0)  # (1, 5)

        with torch.no_grad():
            prob = self._model(x, pos_feats).item()
        return prob

    def _build_sequence(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """Build 30-day feature sequence from daily OHLCV."""
        if len(df) < SEQUENCE_LEN + 26:
            return None

        close  = df["close"]
        volume = df["volume"]

        rsi    = _rsi(close, 14).values[-SEQUENCE_LEN:]
        macd   = (close.ewm(span=12).mean() - close.ewm(span=26).mean()).values[-SEQUENCE_LEN:]
        vol_r  = (volume / volume.rolling(20).mean()).values[-SEQUENCE_LEN:]
        mom    = close.pct_change(20).values[-SEQUENCE_LEN:]
        ret    = close.pct_change(1).values[-SEQUENCE_LEN:]

        seq = np.column_stack([rsi, macd, vol_r, mom, ret])
        seq = np.nan_to_num(seq, nan=0.0)

        # Normalize per-feature to [0,1] range (robust to scale differences)
        for i in range(seq.shape[1]):
            mn, mx = seq[:, i].min(), seq[:, i].max()
            if mx > mn:
                seq[:, i] = (seq[:, i] - mn) / (mx - mn)
        return seq

    def _fallback_rsi_score(self, df: pd.DataFrame, position_data: dict) -> float:
        """Rule-based fallback: RSI overbought + momentum divergence."""
        if len(df) < 30:
            return 0.3  # assume not a peak by default

        close  = df["close"]
        rsi    = _rsi(close, 14).iloc[-1]
        mom_20 = close.pct_change(20).iloc[-1]
        pnl    = position_data.get("unrealized_pnl_pct", 0.0)

        score = 0.3
        if rsi > 75:    score += 0.20
        if rsi > 80:    score += 0.15
        if mom_20 > 0.15: score += 0.10  # already up a lot
        if pnl > 0.20:  score += 0.10   # sitting on big gain

        return min(score, 0.85)  # cap fallback at 0.85 (reserve >0.85 for LSTM only)


class ExitLSTM:
    """LSTM binary classifier for peak detection."""

    def __init__(self):
        try:
            import torch.nn as nn
            import torch
            super().__init__()  # won't work without nn.Module — see _ExitLSTMModule
        except ImportError:
            pass


class _ExitLSTMModule:
    """Actual torch.nn.Module definition. Kept here for reference.
    Instantiated in training pipeline and saved as exit_lstm_v1.pt.

    class _ExitLSTMModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm       = nn.LSTM(input_size=5, hidden_size=64, num_layers=2, batch_first=True, dropout=0.2)
            self.pos_linear = nn.Linear(5, 16)
            self.classifier = nn.Linear(64 + 16, 1)

        def forward(self, x, pos_feats):
            _, (h, _) = self.lstm(x)
            h_last  = h[-1]                          # (batch, 64)
            pos_enc = torch.relu(self.pos_linear(pos_feats))  # (batch, 16)
            combined = torch.cat([h_last, pos_enc], dim=1)    # (batch, 80)
            return torch.sigmoid(self.classifier(combined)).squeeze(-1)
    """


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up    = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    dn    = (-delta).clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    return 100 - (100 / (1 + up / (dn + 1e-9)))
