import numpy as np
import pytest
from src.conformal import ConformalPredictor

def test_conformal_flow():
    """
    [EN] Unit test verifying the end-to-end mathematical flow of ConformalPredictor.
    [TR] ConformalPredictor sınıfının uçtan uca matematiksel akışını doğrulayan birim testi.
    """
    # 1. [EN] Generate Mock Calibration Data (100 Samples, 3 Classes)
    # 1. [TR] Sahte Kalibrasyon Verisi Üret (100 Örnek, 3 Sınıf)
    np.random.seed(42)
    n_samples = 100
    mock_probs = np.random.dirichlet(np.ones(3), size=n_samples)
    mock_targets = np.random.randint(0, 3, size=n_samples)

    # 2. [EN] Initialize ConformalPredictor (95% Coverage Guarantee -> alpha = 0.05)
    # 2. [TR] ConformalPredictor Başlat (%95 Kapsama Garantisi -> alpha = 0.05)
    cp = ConformalPredictor(alpha=0.05)

    # 3. [EN] Compute Non-conformity Scores / [TR] Uyumsuzluk Skorlarını Hesapla
    scores = cp.compute_nonconformity_scores(mock_probs, mock_targets)
    assert len(scores) == 100
    assert np.all(scores >= 0) and np.all(scores <= 1)

    # 4. [EN] Calibrate and Find Threshold (q_hat) / [TR] Kalibre Et ve Eşik (q_hat) Bul
    q_hat = cp.calibrate(scores)
    assert 0 <= q_hat <= 1

    # 5. [EN] Predict Sets for New Test Data / [TR] Yeni Test Verisi İçin Küme Üret
    # Test sample 1: Easy/Confident [0.9, 0.05, 0.05]
    # Test sample 2: Hard/Uncertain [0.4, 0.35, 0.25]
    test_probs = np.array([[0.9, 0.05, 0.05], [0.4, 0.35, 0.25]])
    pred_sets = cp.predict_set(test_probs)

    assert len(pred_sets) == 2
    # [EN] Ensure prediction set for sample 1 contains at least 1 candidate class
    # [TR] 1. örnek için tahmin kümesinin en az 1 aday sınıf içerdiğini doğrula
    assert len(pred_sets[0]) >= 1