"""
risk_quant.py — VaR, drawdown, Sharpe, risk assessment agent.

Reads portfolio metrics and produces a risk assessment report.
"""

from data.fincept.agents.base import BaseAgent

_SYSTEM = """You are a quantitative risk analyst at a systematic hedge fund.
Your job is to assess portfolio risk and flag early warning signs before drawdowns.

Always compute or estimate:
- Value at Risk (VaR 95% and VaR 99%) for the current portfolio
- Current drawdown from equity peak
- Estimated Sharpe ratio based on returns provided
- Correlation risk (are positions moving together?)

Output format — always produce exactly these sections:
## Risk Metrics
[Table: metric | value | status (OK / WARN / ALERT)]

## Correlation Risk
[1-2 sentences — are positions over-correlated? Diversification quality?]

## Early Warning Signals
[Bullet list of max 3 risk flags requiring attention now]

## Recommended Action
[One sentence on what to adjust, if anything]
"""


class RiskQuantAgent(BaseAgent):
    def __init__(self):
        super().__init__("RiskQuant", _SYSTEM)

    def _format_user_message(self, context: dict) -> str:
        metrics = context.get("risk_metrics", {})
        portfolio = context.get("portfolio", [])
        regime = context.get("regime", "UNKNOWN")
        vix = context.get("vix", None)

        m_lines = "\n".join(
            f"  {k}: {v}" for k, v in metrics.items()
        ) or "  (no pre-computed metrics)"

        p_lines = "\n".join(
            f"  {p['symbol']}: weight={p.get('weight_pct', 0):.1f}%, "
            f"sector={p.get('sector', '?')}, beta={p.get('beta', 'N/A')}"
            for p in portfolio[:10]
        ) or "  (no positions)"

        return f"""Market regime: {regime}
VIX: {vix if vix else 'N/A'}

Pre-computed risk metrics:
{m_lines}

Portfolio breakdown:
{p_lines}

Assess risk and provide the structured report."""
