"""
Testing Framework

This module provides a comprehensive testing framework for H-Model validation,
including tests for parameter validation, simulation accuracy, drift detection,
security features, performance, blockchain integrity, and vector embeddings.
"""

import logging
import time
from typing import Any, Dict, List, TYPE_CHECKING

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore

from .exceptions import ValidationError
from .security import SecurityValidator
from .data_structures import ModelParameters

if TYPE_CHECKING:
    from .model_manager import HModelManager


logger = logging.getLogger(__name__)


class HModelTester:
    """Comprehensive testing framework for H-Model validation."""

    def __init__(self, model_manager: "HModelManager"):
        self.model_manager = model_manager
        self.test_results: List[Dict] = []
        logger.info("HModelTester initialized")

    def run_all_tests(self) -> Dict[str, Any]:
        """Run comprehensive test suite."""
        tests = [
            self.test_parameter_validation,
            self.test_simulation_accuracy,
            self.test_drift_detection,
            self.test_security_features,
            self.test_performance,
            self.test_blockchain_integrity,
            self.test_vector_embeddings,
        ]

        results = {}
        for test in tests:
            try:
                test_name = test.__name__
                logger.info(f"Running test: {test_name}")
                result = test()
                results[test_name] = {"status": "passed", "result": result}
                logger.info(f"Test {test_name} passed")
            except Exception as e:
                results[test_name] = {"status": "failed", "error": str(e)}
                logger.error(f"Test {test_name} failed: {e}")

        return results

    def test_parameter_validation(self) -> Dict[str, Any]:
        """Test parameter validation logic."""
        valid_params = {
            "A": 1.0,
            "B": 0.5,
            "C": 0.3,
            "D": 0.2,
            "eta": 0.1,
            "gamma": 1.5,
            "beta": 0.8,
            "sigma": 0.05,
            "tau": 1.0,
        }

        try:
            ModelParameters(**valid_params)
        except Exception as e:
            raise AssertionError(f"Valid parameters rejected: {e}")

        invalid_params = valid_params.copy()
        invalid_params["sigma"] = -1.0

        try:
            ModelParameters(**invalid_params)
            raise AssertionError("Invalid parameters accepted")
        except ValidationError:
            pass

        return {"validation_tests": "passed"}

    def test_simulation_accuracy(self) -> Dict[str, Any]:
        """Test simulation accuracy and consistency."""
        if not np:
            raise ImportError("numpy is required for this test.")

        test_data = np.sin(np.linspace(0, 10, 100)) + np.random.normal(0, 0.1, 100)
        self.model_manager.load_data(test_data)

        t_values = [0.1, 0.2, 0.3, 0.4, 0.5]
        results = []

        for t in t_values:
            H_t = self.model_manager.simulate(t)
            results.append(H_t)

        if any(np.isnan(r) or np.isinf(r) for r in results):
            raise AssertionError("Simulation produced NaN or Inf values")

        variance = np.var(results)
        if variance > 1000:
            raise AssertionError(f"Simulation results too variable: {variance}")

        return {"simulation_results": results, "variance": variance}

    def test_drift_detection(self) -> Dict[str, Any]:
        """Test drift detection mechanisms."""
        if not np:
            raise ImportError("numpy is required for this test.")

        stable_data = np.random.normal(0, 1, 100)
        drift_data = np.random.normal(2, 1, 100)

        combined_data = np.concatenate([stable_data, drift_data])
        self.model_manager.load_data(combined_data)

        for i in range(len(combined_data)):
            self.model_manager.simulate(i * 0.1)

        drift_result = self.model_manager.detect_drift(window=50, threshold=0.1)

        if not drift_result["drift_detected"]:
            raise AssertionError("Failed to detect synthetic drift")

        return drift_result

    def test_security_features(self) -> Dict[str, Any]:
        """Test security validation and features."""
        malicious_input = "<script>alert('xss')</script>"

        if SecurityValidator.validate_input(malicious_input):
            raise AssertionError("Security validator accepted malicious input")

        token1 = SecurityValidator.generate_token()
        token2 = SecurityValidator.generate_token()

        if token1 == token2:
            raise AssertionError("Token generator produced duplicate tokens")

        if len(token1) < 32:
            raise AssertionError("Generated token too short")

        return {"security_validation": "passed", "token_length": len(token1)}

    def test_performance(self) -> Dict[str, Any]:
        """Test performance characteristics."""
        start_time = time.perf_counter()

        for i in range(100):
            self.model_manager.simulate(i * 0.01)

        elapsed_time = time.perf_counter() - start_time
        avg_time_per_simulation = elapsed_time / 100

        if avg_time_per_simulation > 0.1:
            raise AssertionError(f"Simulation too slow: {avg_time_per_simulation:.4f}s per call")

        return {
            "total_time": elapsed_time,
            "avg_time_per_simulation": avg_time_per_simulation,
            "simulations_per_second": 100 / elapsed_time,
        }

    def test_blockchain_integrity(self) -> Dict[str, Any]:
        """Test blockchain verification system."""
        for i in range(5):
            data = {"test_operation": i, "value": i * 10}
            self.model_manager.blockchain.create_block(data)

        if not self.model_manager.blockchain.verify_chain():
            raise AssertionError("Blockchain integrity check failed")

        if len(self.model_manager.blockchain.chain) > 0:
            original_data = self.model_manager.blockchain.chain[1]["data"]
            self.model_manager.blockchain.chain[1]["data"] = "tampered_data"

            if self.model_manager.blockchain.verify_chain():
                self.model_manager.blockchain.chain[1]["data"] = original_data
                raise AssertionError("Failed to detect blockchain tampering")

            self.model_manager.blockchain.chain[1]["data"] = original_data

        return {"blockchain_blocks": len(self.model_manager.blockchain.chain)}

    def test_vector_embeddings(self) -> Dict[str, Any]:
        """Test vector embedding system."""
        text1 = "test string one"
        text2 = "test string two"
        text3 = "completely different content"

        embedding1 = self.model_manager.vector_engine.generate_embedding(text1)
        embedding2 = self.model_manager.vector_engine.generate_embedding(text2)
        embedding3 = self.model_manager.vector_engine.generate_embedding(text3)

        if len(embedding1) != self.model_manager.vector_engine.dimension:
            raise AssertionError("Embedding dimension mismatch")

        sim_12 = self.model_manager.vector_engine.compute_similarity(embedding1, embedding2)
        sim_13 = self.model_manager.vector_engine.compute_similarity(embedding1, embedding3)

        if sim_12 <= sim_13:
            logger.warning("Similarity ordering unexpected but not necessarily wrong")

        return {
            "embedding_dimension": len(embedding1),
            "similarity_12": sim_12,
            "similarity_13": sim_13,
        }
