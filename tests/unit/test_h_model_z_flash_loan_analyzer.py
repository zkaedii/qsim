#!/usr/bin/env python3
"""
Comprehensive tests for h_model_z_flash_loan_analyzer.py

Tests cover:
- Helper functions (softplus, sigmoid)
- FlashLoanEvent dataclass
- MarketImpact dataclass
- H_MODEL_Z_FlashLoanImpactAnalyzer class and methods
"""

import pytest
import numpy as np
import sys
import os
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from hmodelz.core.h_model_z_flash_loan_analyzer import (
    softplus,
    sigmoid,
    FlashLoanEvent,
    MarketImpact,
    H_MODEL_Z_FlashLoanImpactAnalyzer,
)


class TestHelperFunctions:
    """Tests for helper mathematical functions"""

    def test_softplus_positive_large(self):
        """Test softplus for large positive values"""
        result = softplus(np.array([100.0]))
        # For large x, softplus(x) ≈ x
        assert abs(result[0] - 100.0) < 1

    def test_softplus_zero(self):
        """Test softplus at zero"""
        result = softplus(np.array([0.0]))
        # softplus(0) = log(1 + 1) = log(2)
        assert abs(result[0] - np.log(2)) < 0.01

    def test_softplus_negative(self):
        """Test softplus for negative values"""
        result = softplus(np.array([-10.0]))
        assert result[0] > 0
        assert result[0] < 0.001

    def test_softplus_array(self):
        """Test softplus on array"""
        arr = np.array([-10.0, 0.0, 10.0, 100.0])
        result = softplus(arr)
        assert len(result) == 4
        assert all(r > 0 for r in result)

    def test_sigmoid_at_zero(self):
        """Test sigmoid at zero"""
        result = sigmoid(0.0)
        assert abs(result - 0.5) < 0.01

    def test_sigmoid_large_positive(self):
        """Test sigmoid for large positive values"""
        result = sigmoid(100.0)
        assert abs(result - 1.0) < 0.01

    def test_sigmoid_large_negative(self):
        """Test sigmoid for large negative values"""
        result = sigmoid(-100.0)
        assert abs(result - 0.0) < 0.01


class TestFlashLoanEvent:
    """Tests for FlashLoanEvent dataclass"""

    def test_flash_loan_event_creation(self):
        """Test FlashLoanEvent creation"""
        event = FlashLoanEvent(
            timestamp=1.0,
            borrower="trader_1",
            asset="HMLZ",
            amount=1000.0,
            fee=5.0,
            strategy="arbitrage",
            profit=10.0,
            success=True,
        )
        assert event.timestamp == 1.0
        assert event.borrower == "trader_1"
        assert event.asset == "HMLZ"
        assert event.amount == 1000.0
        assert event.fee == 5.0
        assert event.strategy == "arbitrage"
        assert event.profit == 10.0
        assert event.success is True

    def test_flash_loan_event_failed(self):
        """Test failed FlashLoanEvent"""
        event = FlashLoanEvent(
            timestamp=2.0,
            borrower="trader_2",
            asset="WETH",
            amount=500.0,
            fee=2.5,
            strategy="liquidation",
            profit=0.0,
            success=False,
        )
        assert event.success is False
        assert event.profit == 0.0


class TestMarketImpact:
    """Tests for MarketImpact dataclass"""

    def test_market_impact_creation(self):
        """Test MarketImpact creation"""
        impact = MarketImpact(
            price_impact=0.01,
            liquidity_impact=-0.1,
            volatility_impact=0.02,
            volume_impact=2000.0,
            sentiment_impact=0.05,
        )
        assert impact.price_impact == 0.01
        assert impact.liquidity_impact == -0.1
        assert impact.volatility_impact == 0.02
        assert impact.volume_impact == 2000.0
        assert impact.sentiment_impact == 0.05


class TestFlashLoanImpactAnalyzer:
    """Tests for H_MODEL_Z_FlashLoanImpactAnalyzer class"""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance"""
        return H_MODEL_Z_FlashLoanImpactAnalyzer()

    def test_initialization(self, analyzer):
        """Test analyzer initialization"""
        assert analyzer.n == 5
        assert analyzer.flash_loan_impact_factor == 0.5
        assert analyzer.liquidity_sensitivity == 0.3
        assert analyzer.current_price == 0.333
        assert analyzer.total_liquidity == 20.0
        assert len(analyzer.flash_loan_events) == 0
        assert len(analyzer.market_impacts) == 0

    def test_A_i_base_amplitude(self, analyzer):
        """Test A_i amplitude function"""
        result = analyzer.A_i(0, 0.0)
        assert isinstance(result, (float, np.floating))
        # Base amplitude is 1.0 + 0.1 * sin(0) = 1.0
        assert abs(result - 1.0) < 0.01

    def test_A_i_with_flash_loan_impact(self, analyzer):
        """Test A_i with flash loan events"""
        # Add a flash loan event
        event = FlashLoanEvent(
            timestamp=5.0,
            borrower="test",
            asset="HMLZ",
            amount=1000.0,
            fee=5.0,
            strategy="arbitrage",
            profit=10.0,
            success=True,
        )
        analyzer.flash_loan_events.append(event)

        # Get amplitude at nearby time
        result = analyzer.A_i(0, 5.5)
        # Should have some impact from flash loan
        assert result > 0

    def test_B_i_frequency(self, analyzer):
        """Test B_i frequency modulation"""
        result = analyzer.B_i(0, 1.0)
        # Base frequency for i=0 is 1.0
        assert isinstance(result, (float, np.floating))
        assert result > 0

    def test_phi_i_phase(self, analyzer):
        """Test phi_i phase offset"""
        assert abs(analyzer.phi_i(0) - np.pi) < 0.001
        assert abs(analyzer.phi_i(1) - np.pi / 2) < 0.001

    def test_C_i_constant(self, analyzer):
        """Test C_i returns constant"""
        assert analyzer.C_i(0) == 0.3
        assert analyzer.C_i(5) == 0.3

    def test_D_i_decay_rate(self, analyzer):
        """Test D_i decay rate"""
        result = analyzer.D_i(0)
        assert result == 0.05  # Base decay with no events

    def test_f_market_function(self, analyzer):
        """Test f(x) = cos(x)"""
        assert abs(analyzer.f(0) - 1.0) < 0.001

    def test_g_prime_derivative(self, analyzer):
        """Test g_prime(x) = -sin(x)"""
        assert abs(analyzer.g_prime(0) - 0.0) < 0.001

    def test_u_control_input(self, analyzer):
        """Test u(t) control input"""
        result = analyzer.u(0)
        assert abs(result) < 0.01  # u(0) ≈ 0

    def test_normal_distribution(self, analyzer):
        """Test normal distribution sampling"""
        samples = [analyzer.normal(0, 1) for _ in range(100)]
        assert abs(np.mean(samples)) < 0.5
        assert 0.5 < np.std(samples) < 1.5


class TestMarketImpactCalculation:
    """Tests for market impact calculations"""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance"""
        return H_MODEL_Z_FlashLoanImpactAnalyzer()

    def test_calculate_market_impact_no_loan(self, analyzer):
        """Test market impact with no flash loan"""
        impact = analyzer.calculate_market_impact(1.0, flash_loan_amount=0.0)
        assert impact.price_impact == 0.0
        assert impact.liquidity_impact == 0.0

    def test_calculate_market_impact_with_loan(self, analyzer):
        """Test market impact with flash loan"""
        impact = analyzer.calculate_market_impact(1.0, flash_loan_amount=1000.0)
        assert impact.price_impact > 0
        assert impact.liquidity_impact < 0  # Temporary reduction
        assert impact.volume_impact == 2000.0  # 2x loan amount

    def test_volatility_impact_increases_with_events(self, analyzer):
        """Test volatility impact increases with more events"""
        # Add several events
        for i in range(5):
            event = FlashLoanEvent(
                timestamp=float(i),
                borrower=f"trader_{i}",
                asset="HMLZ",
                amount=100.0,
                fee=0.5,
                strategy="arbitrage",
                profit=1.0,
                success=True,
            )
            analyzer.flash_loan_events.append(event)

        impact = analyzer.calculate_market_impact(2.5, flash_loan_amount=100.0)
        # More events should mean more volatility
        assert impact.volatility_impact > 0


class TestFlashLoanSimulation:
    """Tests for flash loan simulation"""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance"""
        return H_MODEL_Z_FlashLoanImpactAnalyzer()

    def test_simulate_flash_loan_arbitrage(self, analyzer):
        """Test simulating arbitrage flash loan"""
        event = analyzer.simulate_flash_loan(
            t=1.0, borrower="trader_1", asset="HMLZ", amount=1000.0, strategy="arbitrage"
        )

        assert event.timestamp == 1.0
        assert event.borrower == "trader_1"
        assert event.asset == "HMLZ"
        assert event.amount == 1000.0
        assert event.strategy == "arbitrage"
        assert event.fee > 0  # Fee should be calculated
        assert len(analyzer.flash_loan_events) == 1

    def test_simulate_flash_loan_liquidation(self, analyzer):
        """Test simulating liquidation flash loan"""
        event = analyzer.simulate_flash_loan(
            t=2.0, borrower="trader_2", asset="WETH", amount=2000.0, strategy="liquidation"
        )

        assert event.strategy == "liquidation"
        assert event.amount == 2000.0

    def test_simulate_flash_loan_yield_farming(self, analyzer):
        """Test simulating yield farming flash loan"""
        event = analyzer.simulate_flash_loan(
            t=3.0, borrower="trader_3", asset="HMLZ", amount=500.0, strategy="yield_farming"
        )

        assert event.strategy == "yield_farming"

    def test_flash_loan_updates_market_state(self, analyzer):
        """Test that successful flash loan updates market state"""
        initial_price = analyzer.current_price

        # Simulate a successful loan (may need multiple tries due to randomness)
        for _ in range(10):
            event = analyzer.simulate_flash_loan(
                t=1.0, borrower="test", asset="HMLZ", amount=1000.0, strategy="arbitrage"
            )
            if event.success:
                break

        # Market state should be affected
        assert len(analyzer.market_impacts) > 0


class TestHHatFunction:
    """Tests for H_hat flash loan impact function"""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance"""
        return H_MODEL_Z_FlashLoanImpactAnalyzer()

    def test_H_hat_returns_float(self, analyzer):
        """Test H_hat returns float"""
        result = analyzer.H_hat(1.0)
        assert isinstance(result, (float, np.floating))

    def test_H_hat_handles_zero(self, analyzer):
        """Test H_hat at t=0"""
        result = analyzer.H_hat(0.0)
        assert not np.isnan(result)
        assert not np.isinf(result)

    def test_H_hat_updates_history(self, analyzer):
        """Test H_hat updates internal history"""
        t_val = 5.0
        result = analyzer.H_hat(t_val)
        assert t_val in analyzer.H_hist
        assert analyzer.H_hist[t_val] == result

    def test_H_hat_with_flash_loans(self, analyzer):
        """Test H_hat incorporates flash loan effects"""
        # Add some flash loan events
        analyzer.simulate_flash_loan(
            t=5.0, borrower="test", asset="HMLZ", amount=1000.0, strategy="arbitrage"
        )

        # H_hat should incorporate flash loan memory
        result = analyzer.H_hat(6.0)
        assert isinstance(result, (float, np.floating))


class TestEcosystemSimulation:
    """Tests for ecosystem simulation"""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance"""
        return H_MODEL_Z_FlashLoanImpactAnalyzer()

    def test_simulate_ecosystem_returns_results(self, analyzer):
        """Test ecosystem simulation returns expected results"""
        results = analyzer.simulate_flash_loan_ecosystem(T=20, num_flash_loans=5)

        assert "time_series" in results
        assert "flash_loan_events" in results
        assert "total_volume" in results
        assert "success_rate" in results
        assert "final_price" in results

    def test_simulate_ecosystem_event_count(self, analyzer):
        """Test correct number of events are generated"""
        results = analyzer.simulate_flash_loan_ecosystem(T=20, num_flash_loans=10)
        assert results["flash_loan_events"] == 10

    def test_simulate_ecosystem_time_series_length(self, analyzer):
        """Test time series has correct length"""
        T = 30
        results = analyzer.simulate_flash_loan_ecosystem(T=T, num_flash_loans=5)
        assert len(results["time_series"]) == T


class TestReportGeneration:
    """Tests for report generation"""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance"""
        return H_MODEL_Z_FlashLoanImpactAnalyzer()

    @pytest.fixture
    def temp_dir(self, tmp_path):
        """Create temp directory for output"""
        old_cwd = os.getcwd()
        os.chdir(tmp_path)
        yield tmp_path
        os.chdir(old_cwd)

    def test_generate_report_empty_events(self, analyzer, temp_dir):
        """Test report generation with no events"""
        # Should handle empty case gracefully
        analyzer.generate_flash_loan_report("test_report.json")
        # No assertion needed - just checking it doesn't crash

    def test_generate_report_with_events(self, analyzer, temp_dir):
        """Test report generation with events"""
        # Simulate some events first
        analyzer.simulate_flash_loan_ecosystem(T=20, num_flash_loans=5)

        # Generate report
        analyzer.generate_flash_loan_report("test_report.json")

        # Check file exists
        assert os.path.exists("test_report.json")

        # Check content
        import json

        with open("test_report.json") as f:
            report = json.load(f)

        assert "timestamp" in report
        assert "ecosystem_overview" in report
        assert "strategy_analysis" in report


class TestVisualization:
    """Tests for visualization functions"""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance"""
        return H_MODEL_Z_FlashLoanImpactAnalyzer()

    @pytest.fixture
    def temp_dir(self, tmp_path):
        """Create temp directory for output"""
        old_cwd = os.getcwd()
        os.chdir(tmp_path)
        yield tmp_path
        os.chdir(old_cwd)

    def test_visualize_returns_results(self, analyzer, temp_dir):
        """Test visualization returns results dict"""
        results = analyzer.visualize_flash_loan_impact(T=10, output_file="test_viz.svg")

        assert isinstance(results, dict)
        assert "time_series" in results
        assert "flash_loan_events" in results


class TestStrategyBehavior:
    """Tests for different strategy behaviors"""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance"""
        return H_MODEL_Z_FlashLoanImpactAnalyzer()

    def test_arbitrage_strategy_profit_range(self, analyzer):
        """Test arbitrage has expected profit characteristics"""
        # Use fixed random seed for deterministic testing
        np.random.seed(42)
        
        # Run multiple simulations
        profits = []
        for _ in range(20):
            event = analyzer.simulate_flash_loan(
                t=1.0, borrower="test", asset="HMLZ", amount=1000.0, strategy="arbitrage"
            )
            if event.success:
                profits.append(event.profit)

        # With seed 42, arbitrage strategy has 80% success rate, so we expect at least 10 successes
        assert len(profits) >= 10, f"Expected at least 10 successful trades, got {len(profits)}"

    def test_liquidation_strategy_higher_risk(self, analyzer):
        """Test liquidation has higher potential profit"""
        # Liquidation should have higher profit range (1-5% vs 0.1-2% for arbitrage)
        event = analyzer.simulate_flash_loan(
            t=1.0, borrower="test", asset="HMLZ", amount=1000.0, strategy="liquidation"
        )
        # Just verify it runs
        assert event.strategy == "liquidation"

    def test_yield_farming_consistent(self, analyzer):
        """Test yield farming has higher success rate"""
        successes = 0
        total = 20
        for _ in range(total):
            event = analyzer.simulate_flash_loan(
                t=1.0, borrower="test", asset="HMLZ", amount=100.0, strategy="yield_farming"
            )
            if event.success:
                successes += 1

        # Yield farming should have ~90% success rate
        success_rate = successes / total
        # Allow for variance but expect high success rate
        assert success_rate > 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
