"""
Tests for ISYS 573 Sales Dashboard
===================================
Run: pytest tests/ -v
"""

import pytest
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from dashboard import load_data, build_region_bar, build_monthly_line, \
                      build_category_pie, build_top_products, build_rep_leaderboard

DATA_PATH = Path(__file__).parent.parent / "data" / "sales.csv"


@pytest.fixture
def df() -> pd.DataFrame:
    return load_data(DATA_PATH)


class TestLoadData:
    def test_loads_without_error(self, df):
        assert len(df) == 500

    def test_required_columns_present(self, df):
        required = {"date", "region", "category", "product",
                    "units_sold", "unit_price", "revenue", "channel"}
        assert required.issubset(set(df.columns))

    def test_date_parsed_as_datetime(self, df):
        assert pd.api.types.is_datetime64_any_dtype(df["date"])

    def test_quarter_column_added(self, df):
        assert "quarter" in df.columns
        assert set(df["quarter"].unique()).issubset({"Q1", "Q2", "Q3", "Q4"})

    def test_month_column_added(self, df):
        assert "month" in df.columns

    def test_revenue_is_positive(self, df):
        assert (df["revenue"] > 0).all()

    def test_four_regions(self, df):
        assert df["region"].nunique() == 4

    def test_raises_on_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_data(Path("/nonexistent/path.csv"))


class TestRegionChart:
    def test_returns_figure(self, df):
        fig = build_region_bar(df)
        assert fig is not None

    def test_has_four_bars(self, df):
        fig = build_region_bar(df)
        # px.bar with color creates one trace per region
        assert len(fig.data) == 4

    def test_filtered_by_quarter(self, df):
        q1 = df[df["quarter"] == "Q1"]
        fig = build_region_bar(q1)
        # Should only show regions present in Q1
        assert len(fig.data[0].x) <= 4


class TestMonthlyChart:
    def test_returns_figure(self, df):
        fig = build_monthly_line(df)
        assert fig is not None

    def test_has_twelve_months_or_fewer(self, df):
        fig = build_monthly_line(df)
        assert len(fig.data[0].x) <= 12

    def test_revenue_values_are_positive(self, df):
        fig = build_monthly_line(df)
        assert all(v > 0 for v in fig.data[0].y)


class TestCategoryChart:
    def test_returns_figure(self, df):
        fig = build_category_pie(df)
        assert fig is not None

    def test_six_categories(self, df):
        fig = build_category_pie(df)
        assert len(fig.data[0].labels) == 6


class TestTopProducts:
    def test_returns_figure(self, df):
        fig = build_top_products(df)
        assert fig is not None

    def test_default_top_10(self, df):
        fig = build_top_products(df)
        assert len(fig.data[0].y) == 10

    def test_custom_n(self, df):
        fig = build_top_products(df, n=5)
        assert len(fig.data[0].y) == 5

    def test_sorted_ascending_for_horizontal_bar(self, df):
        fig = build_top_products(df)
        revenues = list(fig.data[0].x)
        assert revenues == sorted(revenues)


class TestRepLeaderboard:
    def test_returns_figure(self, df) -> None:
        """Test that build_rep_leaderboard returns a valid Plotly Figure."""
        fig = build_rep_leaderboard(df)
        assert fig is not None

    def test_has_bar_chart_trace(self, df) -> None:
        """Test that the leaderboard is a horizontal bar chart."""
        fig = build_rep_leaderboard(df)
        assert len(fig.data) >= 1
        # Verify it's a bar chart with horizontal orientation
        assert fig.data[0].orientation == "h"

    def test_respects_quarter_filter(self, df) -> None:
        """Test that leaderboard updates when filtered by quarter."""
        q1_data = df[df["quarter"] == "Q1"]
        fig_q1 = build_rep_leaderboard(q1_data)
        
        # Q1 should have fewer or equal sales reps represented than full year
        assert len(fig_q1.data[0].y) <= len(build_rep_leaderboard(df).data[0].y)

    def test_sorted_descending_by_revenue(self, df) -> None:
        """Test that sales reps are sorted by revenue (ascending x for horizontal bar)."""
        fig = build_rep_leaderboard(df)
        revenues = list(fig.data[0].x)
        # Should be sorted ascending (for horizontal bar, largest at top)
        assert revenues == sorted(revenues)

    def test_hover_template_contains_revenue(self, df) -> None:
        """Test that hover template includes revenue data."""
        fig = build_rep_leaderboard(df)
        hover = fig.data[0].hovertemplate
        assert "Revenue" in hover or "revenue" in hover or "$" in hover
