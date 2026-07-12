"""Plotly chart renderer (Phase 9).

Converts a `ChartDecision` and result rows into a serialised Plotly
figure (JSON string). The Streamlit frontend (Phase 12) deserialises
this JSON and renders it with `plotly.io.from_json()`.

Design principles:
  - Pure function: no LLM calls, no I/O, deterministic given same inputs
  - Every renderer returns a JSON string or an empty string on failure
  - Chart configuration is professional but minimal — no distracting
    watermarks, gridlines kept subtle, tooltips enabled
  - Dark-mode compatible colour palette (works on both light and dark UIs)

Each chart type is handled by a dedicated `_render_*` method so they
can be independently tested and extended without touching other renderers.
"""

import json
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from app.core.logging import get_logger
from app.visualization.chart_inference import ChartDecision

log = get_logger("analytics.chart_renderer")

# Consistent colour palette across all chart types
_PALETTE = px.colors.qualitative.Set2
_TEMPLATE = "plotly_dark"  # works well in both Streamlit dark and light themes


class ChartRenderer:
    """Renders a Plotly figure from a ChartDecision and query results.

    Usage:
        renderer = ChartRenderer()
        chart_json = renderer.render(decision, rows, columns)
        # Returns "" if rendering is not applicable (chart_type "none" or "table")
    """

    def render(
        self,
        decision: ChartDecision,
        rows: list[dict],
        columns: list[str],
    ) -> str:
        """Render the chart and return a Plotly JSON string.

        Returns an empty string for `chart_type` values of "none" or
        "table" (no chart needed) and on any rendering error.
        """
        if decision.chart_type in ("none", "table") or not rows:
            return ""

        df = self._to_dataframe(rows, columns)
        if df.empty:
            return ""

        renderer_map = {
            "bar":     self._render_bar,
            "line":    self._render_line,
            "pie":     self._render_pie,
            "scatter": self._render_scatter,
            "area":    self._render_area,
        }

        render_fn = renderer_map.get(decision.chart_type)
        if render_fn is None:
            log.warning(f"No renderer for chart type '{decision.chart_type}'")
            return ""

        try:
            fig = render_fn(df, decision)
            fig = self._apply_layout(fig, decision.title)
            return fig.to_json()
        except Exception as exc:  # noqa: BLE001
            log.warning(f"Chart rendering failed for type '{decision.chart_type}': {exc}")
            return ""

    # ── Renderers ─────────────────────────────────────────────────────────────

    def _render_bar(self, df: pd.DataFrame, d: ChartDecision) -> go.Figure:
        x_col = self._resolve_column(df, d.x_column)
        y_col = self._resolve_column(df, d.y_column)
        fig = px.bar(
            df, x=x_col, y=y_col,
            color_discrete_sequence=_PALETTE,
            template=_TEMPLATE,
        )
        fig.update_traces(marker_line_width=0)
        return fig

    def _render_line(self, df: pd.DataFrame, d: ChartDecision) -> go.Figure:
        x_col = self._resolve_column(df, d.x_column)
        y_col = self._resolve_column(df, d.y_column)
        fig = px.line(
            df, x=x_col, y=y_col,
            markers=True,
            color_discrete_sequence=_PALETTE,
            template=_TEMPLATE,
        )
        return fig

    def _render_pie(self, df: pd.DataFrame, d: ChartDecision) -> go.Figure:
        names_col = self._resolve_column(df, d.x_column)
        values_col = self._resolve_column(df, d.y_column)
        fig = px.pie(
            df, names=names_col, values=values_col,
            color_discrete_sequence=_PALETTE,
            template=_TEMPLATE,
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")
        return fig

    def _render_scatter(self, df: pd.DataFrame, d: ChartDecision) -> go.Figure:
        x_col = self._resolve_column(df, d.x_column)
        y_col = self._resolve_column(df, d.y_column)
        fig = px.scatter(
            df, x=x_col, y=y_col,
            color_discrete_sequence=_PALETTE,
            template=_TEMPLATE,
        )
        return fig

    def _render_area(self, df: pd.DataFrame, d: ChartDecision) -> go.Figure:
        x_col = self._resolve_column(df, d.x_column)
        y_col = self._resolve_column(df, d.y_column)
        fig = px.area(
            df, x=x_col, y=y_col,
            color_discrete_sequence=_PALETTE,
            template=_TEMPLATE,
        )
        return fig

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _to_dataframe(rows: list[dict], columns: list[str]) -> pd.DataFrame:
        """Convert rows + columns into a Pandas DataFrame."""
        try:
            return pd.DataFrame(rows, columns=columns)
        except Exception as exc:
            log.warning(f"DataFrame construction failed: {exc}")
            return pd.DataFrame()

    @staticmethod
    def _resolve_column(df: pd.DataFrame, col_name: str) -> str:
        """Return `col_name` if it exists in df, else the first column."""
        if col_name and col_name in df.columns:
            return col_name
        return df.columns[0] if len(df.columns) > 0 else ""

    @staticmethod
    def _apply_layout(fig: go.Figure, title: str) -> go.Figure:
        """Apply consistent layout settings across all chart types."""
        fig.update_layout(
            title=dict(text=title, font=dict(size=16)),
            margin=dict(l=40, r=20, t=60, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        fig.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,0.2)")
        fig.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.2)")
        return fig
