import pytest
import sys
from unittest.mock import patch
from agentic_reliability_framework.cli.diagnose import run_diagnose, get_risk_level, get_suggested_action

class TestDiagnoseFunctions:
    def test_get_risk_level(self):
        assert get_risk_level(0.9) == "LOW"
        assert get_risk_level(0.7) == "MEDIUM"
        assert get_risk_level(0.4) == "HIGH"
        assert get_risk_level(0.2) == "CRITICAL"
    
    def test_get_suggested_action(self):
        assert get_suggested_action(0.9) == "Monitor"
        assert get_suggested_action(0.7) == "Investigate"
        assert get_suggested_action(0.4) == "Intervene"
        assert get_suggested_action(0.2) == "IMMEDIATE ACTION REQUIRED"
    
    def test_run_diagnose_with_anomaly(self):
        result = run_diagnose(anomaly_score=0.3)
        assert result["reliability"] == pytest.approx(0.7)
        assert result["risk_level"] == "MEDIUM"
        assert result["source"] == "direct anomaly score"
    
    def test_run_diagnose_with_latency(self):
        result = run_diagnose(latency=200)
        assert 0 <= result["reliability"] <= 1
        assert result["source"] == "latency (200ms)"
    
    def test_run_diagnose_with_error_rate(self):
        result = run_diagnose(error_rate=0.05)
        assert 0 <= result["reliability"] <= 1
        assert result["source"] == "error rate (5.0%)"
    
    def test_run_diagnose_no_args(self):
        result = run_diagnose()
        assert result["reliability"] == 0.77
        assert result["source"] == "demo (no input provided)"
    
    @patch("sys.argv", ["arf", "--anomaly", "0.2"])
    def test_main_with_anomaly(self):
        from agentic_reliability_framework.cli.diagnose import main
        exit_code = main()
        assert exit_code == 0
    
    @patch("sys.argv", ["arf", "--latency", "500"])
    def test_main_with_latency(self):
        from agentic_reliability_framework.cli.diagnose import main
        exit_code = main()
        assert exit_code in (0,1,2)
