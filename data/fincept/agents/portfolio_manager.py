"""
portfolio_manager.py — Capital allocation + Kelly/Mean-Variance optimization agent.

Given the current portfolio state, top signals, and macro snapshot,
produces a capital allocation recommendation with position sizing rationale.
"""

from data.fincept.agents.base import BaseAgent

_SYSTEM = """You are an elite portfolio manager with deep expertise in quantitative finance.
Your role is to allocate capital optimally across a set of signals.

Rules you must follow:
- No single position > 15% of portfolio (concentration risk)
- Tech sector (AI/Cloud/Semis) combined <= 35% of portfolio
- Keep >= 10% cash in stressed regimes (VIX > 25)
- Prefer STRONG_BUY signals for full allocation; scale down for BUY/CONFLICT

Output format — always produce exactly these sections:
## Allocation Recommendation
[Table: ticker | conviction | suggested_weight% | rationale_one_line]

## Risk Budget
[1-2 sentences on sector concentration and regime risk]

## Key Watchouts
[Bullet list, max 3 items]
"""


class PortfolioManagerAgent(BaseAgent):
    def __init__(self):
        super().__init__("PortfolioManager", _SYSTEM)

    def _format_user_message(self, context: dict) -> str:
        portfolio = context.get("portfolio", [])
        signals = context.get("top_signals", [])
        macro = context.get("macro", {})
        regime = context.get("regime", "UNKNOWN")

        p_lines = "\n".join(
            f"  {p['symbol']}: {p.get('shares', 0)} shares @ ${p.get('entry_price', 0):.2f}, "
            f"current P&L {p.get('unrealized_pnl', 0):+.1f}%"
            for p in portfolio[:10]
        ) or "  (no current positions)"

        s_lines = "\n".join(
            f"  {s['symbol']}: consensus={s.get('consensus','?')} lgbm={s.get('lgbm_score',0):.3f} "
            f"size_mult={s.get('size_multiplier',1.0)}"
            for s in signals[:10]
        ) or "  (no signals)"

        macro_str = " | ".join(
            f"{k}: {v.get('value', 'N/A')}" for k, v in macro.items()
        ) if macro else "N/A"

        return f"""Current market regime: {regime}
Macro: {macro_str}

Current positions:
{p_lines}

Top signals to evaluate:
{s_lines}

Provide capital allocation recommendation following the output format."""
