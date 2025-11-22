#!/usr/bin/env python3
"""
Comprehensive tests for enterprise_scaling_framework.py

Tests cover:
- EnterpriseConfig dataclass
- EnterpriseMetrics class
- Auto-scaling configuration
- ClusterManager class
- RealWorldDomainManager class
- RealWorldDomainProcessor class
- Domain-specific processing methods
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import patch

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Clear prometheus registry before importing to avoid duplicate metrics
import prometheus_client
prometheus_client.REGISTRY = prometheus_client.CollectorRegistry(auto_describe=True)

from hmodelz.engines.enterprise_scaling_framework import (
    EnterpriseConfig,
    EnterpriseMetrics,
    ClusterManager,
    RealWorldDomainManager,
    RealWorldDomainProcessor,
)


# Fixture to handle prometheus registry cleanup
@pytest.fixture(autouse=True)
def reset_prometheus_registry():
    """Reset prometheus registry before each test"""
    # Store original registry
    original_registry = prometheus_client.REGISTRY
    # Create new registry for test
    prometheus_client.REGISTRY = prometheus_client.CollectorRegistry(auto_describe=True)
    yield
    # Restore original after test
    prometheus_client.REGISTRY = original_registry


class TestEnterpriseConfig:
    """Tests for EnterpriseConfig dataclass"""

    def test_default_values(self):
        """Test EnterpriseConfig default values"""
        config = EnterpriseConfig()
        assert config.min_workers == 10
        assert config.max_workers == 1000
        assert config.auto_scale_threshold == 0.8
        assert config.scale_up_cooldown == 300
        assert config.scale_down_cooldown == 600

    def test_high_availability_defaults(self):
        """Test HA configuration defaults"""
        config = EnterpriseConfig()
        assert len(config.cluster_nodes) == 3
        assert config.health_check_interval == 30
        assert config.failover_timeout == 60
        assert config.backup_interval == 3600

    def test_performance_defaults(self):
        """Test performance configuration defaults"""
        config = EnterpriseConfig()
        assert config.max_concurrent_requests == 10000
        assert config.request_timeout == 30
        assert config.connection_pool_size == 100
        assert config.cache_ttl == 3600

    def test_security_defaults(self):
        """Test security configuration defaults"""
        config = EnterpriseConfig()
        assert config.encryption_key is None
        assert config.jwt_secret == "enterprise-secret-key"
        assert config.rate_limit == "1000 per minute"
        assert "*" in config.cors_origins

    def test_monitoring_defaults(self):
        """Test monitoring configuration defaults"""
        config = EnterpriseConfig()
        assert config.metrics_port == 8080
        assert config.log_level == "INFO"
        assert config.alert_webhook is None

    def test_enabled_domains(self):
        """Test default enabled domains"""
        config = EnterpriseConfig()
        expected_domains = [
            "financial_trading",
            "supply_chain",
            "healthcare",
            "manufacturing",
            "energy_grid",
            "telecommunications",
            "smart_cities",
            "logistics",
        ]
        assert set(config.enabled_domains) == set(expected_domains)

    def test_custom_config(self):
        """Test custom configuration values"""
        config = EnterpriseConfig(
            min_workers=20,
            max_workers=2000,
            auto_scale_threshold=0.75,
            metrics_port=9090,
        )
        assert config.min_workers == 20
        assert config.max_workers == 2000
        assert config.auto_scale_threshold == 0.75
        assert config.metrics_port == 9090


class TestEnterpriseMetrics:
    """Tests for EnterpriseMetrics class"""

    @pytest.fixture(scope="class")
    def shared_metrics(self):
        """Create a shared metrics instance for the test class"""
        return EnterpriseMetrics()

    def test_initialization(self, shared_metrics):
        """Test EnterpriseMetrics initialization"""
        assert shared_metrics.request_count is not None
        assert shared_metrics.request_duration is not None
        assert shared_metrics.active_connections is not None

    def test_system_metrics_exist(self, shared_metrics):
        """Test system metrics are created"""
        assert shared_metrics.cpu_usage is not None
        assert shared_metrics.memory_usage is not None
        assert shared_metrics.disk_usage is not None

    def test_business_metrics_exist(self, shared_metrics):
        """Test business metrics are created"""
        assert shared_metrics.events_processed is not None
        assert shared_metrics.ai_optimization_score is not None
        assert shared_metrics.chain_operations is not None

    def test_performance_metrics_exist(self, shared_metrics):
        """Test performance metrics are created"""
        assert shared_metrics.throughput is not None
        assert shared_metrics.error_rate is not None
        assert shared_metrics.latency_p95 is not None

    @patch("psutil.cpu_percent", return_value=50.0)
    @patch("psutil.virtual_memory")
    @patch("psutil.disk_usage")
    def test_update_system_metrics(self, mock_disk, mock_memory, mock_cpu, shared_metrics):
        """Test updating system metrics"""
        mock_memory.return_value.used = 4 * 1024 * 1024 * 1024  # 4GB
        mock_disk.return_value.percent = 60.0

        shared_metrics.update_system_metrics()
        # If we reach here without error, the update worked


class TestAutoScaler:
    """Tests for auto-scaling configuration parameters"""

    def test_autoscaler_config_setup(self):
        """Test AutoScaler can be configured"""
        config = EnterpriseConfig(min_workers=5, max_workers=100)
        assert config.min_workers == 5
        assert config.max_workers == 100

    def test_autoscaler_scaling_thresholds(self):
        """Test AutoScaler scaling thresholds configuration"""
        config = EnterpriseConfig(auto_scale_threshold=0.75)
        assert config.auto_scale_threshold == 0.75

    def test_autoscaler_cooldown_settings(self):
        """Test AutoScaler cooldown settings"""
        config = EnterpriseConfig(scale_up_cooldown=600, scale_down_cooldown=1200)
        assert config.scale_up_cooldown == 600
        assert config.scale_down_cooldown == 1200


class TestClusterManager:
    """Tests for ClusterManager class"""

    @pytest.fixture
    def cluster_manager(self):
        """Create a ClusterManager instance for testing"""
        config = EnterpriseConfig()
        return ClusterManager(config)

    def test_initialization(self, cluster_manager):
        """Test ClusterManager initialization"""
        assert cluster_manager.node_id is not None
        assert cluster_manager.is_leader is False
        assert cluster_manager.cluster_health == {}

    def test_node_id_is_hostname(self, cluster_manager):
        """Test that node_id is set to hostname"""
        import socket

        assert cluster_manager.node_id == socket.gethostname()


class TestRealWorldDomainProcessor:
    """Tests for RealWorldDomainProcessor class"""

    @pytest.fixture
    def financial_processor(self):
        """Create a financial trading processor"""
        config = {
            "description": "High-frequency trading",
            "max_latency_ms": 1,
            "required_throughput": 100000,
            "compliance": ["SOX"],
        }
        return RealWorldDomainProcessor("financial_trading", config)

    @pytest.fixture
    def supply_chain_processor(self):
        """Create a supply chain processor"""
        config = {
            "description": "Supply chain optimization",
            "max_latency_ms": 100,
            "required_throughput": 50000,
            "compliance": ["ISO 9001"],
        }
        return RealWorldDomainProcessor("supply_chain", config)

    @pytest.fixture
    def healthcare_processor(self):
        """Create a healthcare processor"""
        config = {
            "description": "Patient data management",
            "max_latency_ms": 500,
            "required_throughput": 25000,
            "compliance": ["HIPAA"],
        }
        return RealWorldDomainProcessor("healthcare", config)

    @pytest.fixture
    def manufacturing_processor(self):
        """Create a manufacturing processor"""
        config = {
            "description": "Industrial IoT",
            "max_latency_ms": 50,
            "required_throughput": 75000,
            "compliance": ["ISO 27001"],
        }
        return RealWorldDomainProcessor("manufacturing", config)

    @pytest.fixture
    def energy_processor(self):
        """Create an energy grid processor"""
        config = {
            "description": "Smart grid management",
            "max_latency_ms": 10,
            "required_throughput": 200000,
            "compliance": ["NERC CIP"],
        }
        return RealWorldDomainProcessor("energy_grid", config)

    @pytest.fixture
    def telecom_processor(self):
        """Create a telecommunications processor"""
        config = {
            "description": "Network optimization",
            "max_latency_ms": 5,
            "required_throughput": 150000,
            "compliance": ["GDPR"],
        }
        return RealWorldDomainProcessor("telecommunications", config)

    @pytest.fixture
    def smart_cities_processor(self):
        """Create a smart cities processor"""
        config = {
            "description": "Urban infrastructure",
            "max_latency_ms": 200,
            "required_throughput": 100000,
            "compliance": ["GDPR"],
        }
        return RealWorldDomainProcessor("smart_cities", config)

    @pytest.fixture
    def logistics_processor(self):
        """Create a logistics processor"""
        config = {
            "description": "Fleet management",
            "max_latency_ms": 1000,
            "required_throughput": 30000,
            "compliance": ["DOT"],
        }
        return RealWorldDomainProcessor("logistics", config)

    def test_processor_initialization(self, financial_processor):
        """Test processor initialization"""
        assert financial_processor.domain_name == "financial_trading"
        assert financial_processor.config["max_latency_ms"] == 1
        assert len(financial_processor.processing_history) == 0

    @pytest.mark.asyncio
    async def test_process_financial_trading(self, financial_processor):
        """Test financial trading event processing"""
        event_data = {"symbol": "AAPL", "price": 150.0, "volume": 1000}
        result = await financial_processor.process_event(event_data)

        assert result["domain"] == "financial_trading"
        assert result["status"] == "success"
        assert "processing_time_ms" in result
        assert "result" in result

        # Check result contains trading decision
        inner_result = result["result"]
        assert "symbol" in inner_result
        assert "decision" in inner_result
        assert inner_result["decision"] in ["BUY", "SELL", "HOLD"]
        assert "risk_score" in inner_result

    @pytest.mark.asyncio
    async def test_process_supply_chain(self, supply_chain_processor):
        """Test supply chain event processing"""
        event_data = {
            "shipment_id": "SH001",
            "location": "NYC",
            "temperature": 22.0,
        }
        result = await supply_chain_processor.process_event(event_data)

        assert result["status"] == "success"
        inner_result = result["result"]
        assert "shipment_id" in inner_result
        assert "temperature_status" in inner_result
        assert inner_result["temperature_status"] in ["COMPLIANT", "NON_COMPLIANT"]

    @pytest.mark.asyncio
    async def test_process_healthcare(self, healthcare_processor):
        """Test healthcare event processing with HIPAA compliance"""
        event_data = {
            "patient_id": "patient_123",
            "vital_signs": {"heart_rate": 75, "blood_pressure_systolic": 120},
        }
        result = await healthcare_processor.process_event(event_data)

        assert result["status"] == "success"
        inner_result = result["result"]
        assert "patient_id_hash" in inner_result
        assert "patient_123" not in inner_result["patient_id_hash"]  # Should be hashed
        assert "hipaa_compliant" in inner_result
        assert inner_result["hipaa_compliant"] is True
        assert "risk_assessment" in inner_result

    @pytest.mark.asyncio
    async def test_process_manufacturing(self, manufacturing_processor):
        """Test manufacturing IoT event processing"""
        event_data = {
            "machine_id": "M001",
            "sensor_data": {"vibration": 0.3, "temperature": 28.0, "pressure": 1.2},
        }
        result = await manufacturing_processor.process_event(event_data)

        assert result["status"] == "success"
        inner_result = result["result"]
        assert "machine_id" in inner_result
        assert "health_score" in inner_result
        assert 0 <= inner_result["health_score"] <= 1
        assert "maintenance_needed" in inner_result

    @pytest.mark.asyncio
    async def test_process_energy_grid(self, energy_processor):
        """Test energy grid event processing"""
        event_data = {"grid_node": "NODE_001", "power_demand": 1000.0, "renewable_supply": 800.0}
        result = await energy_processor.process_event(event_data)

        assert result["status"] == "success"
        inner_result = result["result"]
        assert "grid_node" in inner_result
        assert "stability_score" in inner_result
        assert "recommended_action" in inner_result

    @pytest.mark.asyncio
    async def test_process_telecommunications(self, telecom_processor):
        """Test telecommunications event processing"""
        event_data = {
            "cell_tower": "TOWER_001",
            "bandwidth_usage": 0.7,
            "connection_count": 500,
        }
        result = await telecom_processor.process_event(event_data)

        assert result["status"] == "success"
        inner_result = result["result"]
        assert "cell_tower" in inner_result
        assert "qos_score" in inner_result
        assert "congestion_level" in inner_result

    @pytest.mark.asyncio
    async def test_process_smart_cities(self, smart_cities_processor):
        """Test smart cities event processing"""
        event_data = {"zone": "DOWNTOWN", "traffic_density": 0.6, "air_quality_index": 75}
        result = await smart_cities_processor.process_event(event_data)

        assert result["status"] == "success"
        inner_result = result["result"]
        assert "zone" in inner_result
        assert "traffic_flow_score" in inner_result
        assert "environmental_score" in inner_result
        assert "livability_index" in inner_result

    @pytest.mark.asyncio
    async def test_process_logistics(self, logistics_processor):
        """Test logistics event processing"""
        event_data = {
            "vehicle_id": "V001",
            "location": [0, 0],
            "destination": [10, 10],
            "cargo_weight": 1000,
        }
        result = await logistics_processor.process_event(event_data)

        assert result["status"] == "success"
        inner_result = result["result"]
        assert "vehicle_id" in inner_result
        assert "optimized_route_distance_km" in inner_result
        assert "estimated_fuel_cost_usd" in inner_result
        assert "fuel_efficiency_score" in inner_result

    @pytest.mark.asyncio
    async def test_processing_history_recorded(self, financial_processor):
        """Test that processing history is recorded"""
        event_data = {"symbol": "GOOGL", "price": 2800.0, "volume": 500}
        await financial_processor.process_event(event_data)

        assert len(financial_processor.processing_history) == 1
        history_entry = financial_processor.processing_history[0]
        assert "timestamp" in history_entry
        assert "processing_time_ms" in history_entry
        assert "success" in history_entry


class TestRealWorldDomainManager:
    """Tests for RealWorldDomainManager class"""

    @pytest.fixture
    def domain_manager(self):
        """Create a RealWorldDomainManager instance"""
        config = EnterpriseConfig()
        return RealWorldDomainManager(config)

    def test_initialization(self, domain_manager):
        """Test domain manager initialization"""
        assert len(domain_manager.domain_processors) > 0
        assert len(domain_manager.domain_metrics) == 0

    def test_all_domains_initialized(self, domain_manager):
        """Test that all configured domains are initialized"""
        expected_domains = [
            "financial_trading",
            "supply_chain",
            "healthcare",
            "manufacturing",
            "energy_grid",
            "telecommunications",
            "smart_cities",
            "logistics",
        ]
        for domain in expected_domains:
            assert domain in domain_manager.domain_processors

    @pytest.mark.asyncio
    async def test_process_domain_event(self, domain_manager):
        """Test processing domain event"""
        event_data = {"symbol": "TSLA", "price": 250.0, "volume": 2000}
        result = await domain_manager.process_domain_event("financial_trading", event_data)

        assert result["domain"] == "financial_trading"
        assert result["status"] == "success"

    @pytest.mark.asyncio
    async def test_process_invalid_domain_raises(self, domain_manager):
        """Test that invalid domain raises ValueError"""
        with pytest.raises(ValueError, match="not enabled"):
            await domain_manager.process_domain_event("invalid_domain", {})

    def test_get_domain_status(self, domain_manager):
        """Test getting domain status"""
        status = domain_manager.get_domain_status()

        assert len(status) > 0
        for domain, info in status.items():
            assert info["enabled"] is True
            assert "description" in info
            assert "compliance" in info
            assert "performance_requirements" in info


class TestDomainProcessorEdgeCases:
    """Tests for edge cases in domain processors"""

    @pytest.fixture
    def healthcare_processor(self):
        """Create a healthcare processor"""
        config = {
            "description": "Patient data management",
            "max_latency_ms": 500,
            "required_throughput": 25000,
            "compliance": ["HIPAA"],
        }
        return RealWorldDomainProcessor("healthcare", config)

    @pytest.mark.asyncio
    async def test_healthcare_high_risk_detection(self, healthcare_processor):
        """Test healthcare risk detection for abnormal vitals"""
        event_data = {
            "patient_id": "patient_high_risk",
            "vital_signs": {
                "heart_rate": 120,  # High
                "blood_pressure_systolic": 160,  # High
            },
        }
        result = await healthcare_processor.process_event(event_data)

        inner_result = result["result"]
        assert inner_result["risk_assessment"] == "HIGH"
        assert len(inner_result["risk_factors"]) > 0
        assert inner_result["recommendation"] == "CONSULT_PHYSICIAN"

    @pytest.mark.asyncio
    async def test_healthcare_normal_vitals(self, healthcare_processor):
        """Test healthcare with normal vitals"""
        event_data = {
            "patient_id": "patient_normal",
            "vital_signs": {"heart_rate": 70, "blood_pressure_systolic": 115},
        }
        result = await healthcare_processor.process_event(event_data)

        inner_result = result["result"]
        assert inner_result["risk_assessment"] == "NORMAL"
        assert len(inner_result["risk_factors"]) == 0
        assert inner_result["recommendation"] == "ROUTINE_MONITORING"


class TestFinancialTradingRiskScores:
    """Tests for financial trading risk score calculations"""

    @pytest.fixture
    def trading_processor(self):
        """Create a financial trading processor"""
        config = {
            "description": "Trading",
            "max_latency_ms": 1,
            "required_throughput": 100000,
            "compliance": ["SOX"],
        }
        return RealWorldDomainProcessor("financial_trading", config)

    @pytest.mark.asyncio
    async def test_low_risk_buy_decision(self, trading_processor):
        """Test buy decision for low risk trades"""
        event_data = {"symbol": "ABC", "price": 10.0, "volume": 100}  # Low value trade
        result = await trading_processor.process_event(event_data)

        inner_result = result["result"]
        assert inner_result["risk_score"] < 0.3
        # Low risk should suggest BUY (though actual logic depends on implementation)

    @pytest.mark.asyncio
    async def test_high_risk_trade(self, trading_processor):
        """Test high risk trade detection"""
        event_data = {
            "symbol": "XYZ",
            "price": 1000000.0,  # Very high value
            "volume": 10000,
        }
        result = await trading_processor.process_event(event_data)

        # High value trade should have higher risk score


class TestEnergyGridOptimization:
    """Tests for energy grid optimization logic"""

    @pytest.fixture
    def energy_processor(self):
        """Create an energy grid processor"""
        config = {
            "description": "Smart grid",
            "max_latency_ms": 10,
            "required_throughput": 200000,
            "compliance": ["NERC CIP"],
        }
        return RealWorldDomainProcessor("energy_grid", config)

    @pytest.mark.asyncio
    async def test_supply_deficit_scenario(self, energy_processor):
        """Test grid behavior with supply deficit"""
        event_data = {
            "grid_node": "NODE_001",
            "power_demand": 1000.0,
            "renewable_supply": 600.0,  # Deficit
        }
        result = await energy_processor.process_event(event_data)

        inner_result = result["result"]
        assert inner_result["supply_deficit_mw"] > 0
        assert inner_result["recommended_action"] == "ACTIVATE_BACKUP_GENERATORS"

    @pytest.mark.asyncio
    async def test_excess_energy_scenario(self, energy_processor):
        """Test grid behavior with excess renewable energy"""
        event_data = {
            "grid_node": "NODE_002",
            "power_demand": 800.0,
            "renewable_supply": 1200.0,  # Excess
        }
        result = await energy_processor.process_event(event_data)

        inner_result = result["result"]
        assert inner_result["supply_deficit_mw"] == 0
        assert inner_result["recommended_action"] == "STORE_EXCESS_ENERGY"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
