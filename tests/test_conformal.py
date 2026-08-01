import numpy as np
import pytest
from src.conformal import ConformalPredictor

def test_conformal_flow():
    # 1. Mock Kalibrasyon Verisi Oluştur (100 Örnek, 3 Sınıf)
    np.random.seed(42)
    n_samples = 100
    mock_probs = np.random.dirichlet(np.ones(3), size=n_samples)
    mock_targets = np.random.randint(0, 3, size=n_samples)

    # 2. ConformalPredictor Başlat (%95 Garanti -> alpha = 0.05)
    cp = ConformalPredictor(alpha=0.05)

    # 3. Skorları Hesapla
    scores = cp.compute_nonconformity_scores(mock_probs, mock_targets)
    assert len(scores) == 100
    assert np.all(scores >= 0) and np.all(scores <= 1)

    # 4. Kalibre Et ve Eşik (q_hat) Bul
    q_hat = cp.calibrate(scores)
    assert 0 <= q_hat <= 1

    # 5. Yeni Test Verisi İçin Küme Üret
    test_probs = np.array([[0.9, 0.05, 0.05], [0.4, 0.35, 0.25]])
    pred_sets = cp.predict_set(test_probs)

    assert len(pred_sets) == 2
    # Kolay örnek (0.9) tek elemanlı veya dar küme olmalı
    assert len(pred_sets[0]) >= 1