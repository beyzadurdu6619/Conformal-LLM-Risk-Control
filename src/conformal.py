import numpy as np
from typing import List, Union

class ConformalPredictor:
    """
    [EN] Distribution-Free Conformal Prediction Engine for Uncertainty Quantification.
         Based on Angelopoulos & Bates (2022).
    [TR] Dağılımdan Bağımsız Belirsizlik Ölçümü için Uyumlu Tahmin Motoru.
         Angelopoulos & Bates (2022) makalesine dayanmaktadır.
    """
    def __init__(self, alpha: float = 0.05):
        """
        :param alpha: [EN] Desired error rate tolerance (e.g., 0.05 for 95% coverage guarantee).
                      [TR] İstenen hata oranı toleransı (Örn: %95 kapsama garantisi için 0.05).
        """
        # [EN] Check if alpha is within valid probability bounds (0, 1)
        # [TR] Alpha değerinin geçerli olasılık sınırlarında (0, 1) olup olmadığını kontrol et
        if not (0 < alpha < 1):
            raise ValueError("Alpha (error rate) must be strictly between 0 and 1. / Alpha 0 ile 1 arasında olmalıdır.")
        
        self.alpha = alpha
        self.q_hat = None  # [EN] Calibrated quantile threshold / [TR] Kalibre edilmiş quantile eşiği

    def compute_nonconformity_scores(self, probabilities: np.ndarray, target_indices: np.ndarray) -> np.ndarray:
        """
        [EN] Calculates Non-conformity Score: s_i = 1 - P(y_true | x_i)
             Measures how 'surprised' the model is by the ground truth label.
        [TR] Uyumsuzluk Skoru Hesaplar: s_i = 1 - P(y_doğru | x_i)
             Modelin gerçek doğru cevaba ne kadar 'şaşırdığını' veya bocaladığını ölçer.
        
        :param probabilities: [EN] Predicted probabilities matrix (N, Num_Classes)
                              [TR] Tahmin edilen olasılıklar matrisi (N, Sınıf_Sayısı)
        :param target_indices: [EN] True class indices for calibration set (N,)
                               [TR] Kalibration seti için gerçek sınıf indeksleri (N,)
        :return: [EN] Array of non-conformity scores / [TR] Uyumsuzluk skorları dizisi
        """
        n = len(target_indices)
        if len(probabilities) != n:
            raise ValueError("Length of probabilities and target_indices must match. / Olasılıklar ve hedef boyutları eşleşmelidir.")
            
        # [EN] Extract probabilities assigned by the model to the ground-truth classes
        # [TR] Modelin gerçek doğru sınıflara atadığı olasılık değerlerini çek
        true_class_probs = probabilities[np.arange(n), target_indices]
        
        # [EN] Calculate Non-conformity score: s_i = 1 - P(y_true | x_i)
        # [TR] Uyumsuzluk skorunu hesapla: s_i = 1 - P(y_doğru | x_i)
        scores = 1.0 - true_class_probs
        return scores

    def calibrate(self, scores: np.ndarray) -> float:
        """
        [EN] Calculates the conformal empirical quantile threshold (q_hat).
             Formula: q_hat = ceil((n + 1) * (1 - alpha)) / n -th smallest score.
        [TR] Uyumlu ampirik quantile eşiğini (q_hat) hesaplar.
             Formül: q_hat = ceil((n + 1) * (1 - alpha)) / n -inci en küçük skor.
        
        :param scores: [EN] Array of non-conformity scores from calibration set.
                       [TR] Kalibrasyon setinden elde edilen uyumsuzluk skorları dizisi.
        :return: [EN/TR] q_hat (Quantile threshold / Quantile eşiği)
        """
        n = len(scores)
        if n == 0:
            raise ValueError("Calibration set cannot be empty. / Kalibrasyon seti boş olamaz.")
            
        # [EN] Calculate finite-sample corrected quantile index 'k'
        # [TR] Sonlu örneklem düzeltmeli (n+1) quantile indeksi 'k' değerini hesapla
        k = int(np.ceil((n + 1) * (1.0 - self.alpha)))
        
        # [EN] Clip 'k' index to ensure it stays within valid array bounds [1, n]
        # [TR] 'k' indeksinin dizi sınırlarında [1, n] kalmasını garanti et
        k = min(max(k, 1), n)
        
        # [EN] Sort scores in ascending order and select the k-th smallest score (1-indexed)
        # [TR] Skorları küçükten büyüğe sırala ve k-ıncı en küçük skoru seç (1-indeksli)
        sorted_scores = np.sort(scores)
        self.q_hat = sorted_scores[k - 1]
        
        return self.q_hat

    def predict_set(self, probabilities: np.ndarray) -> List[List[int]]:
        """
        [EN] Generates prediction sets for new test samples based on calibrated q_hat.
             Includes candidate classes where nonconformity <= q_hat (i.e., probability >= 1 - q_hat).
        [TR] Kalibre edilmiş q_hat eşiğine göre yeni test örnekleri için tahmin kümeleri üretir.
             Uyumsuzluğu <= q_hat (yani olasılığı >= 1 - q_hat) olan aday sınıfları kümeye ekler.
        
        :param probabilities: [EN] Test sample probabilities (N_test, Num_Classes)
                              [TR] Test örneği olasılıkları (N_test, Sınıf_Sayısı)
        :return: [EN] List of prediction sets (candidate indices for each test point)
                 [TR] Tahmin kümeleri listesi (her test noktası için aday sınıf indeksleri)
        """
        if self.q_hat is None:
            raise RuntimeError("Model must be calibrated using .calibrate() before predict_set(). / Model kalibre edilmeden tahmin kümesi üretilemez.")
            
        # [EN] Calculate minimum required probability threshold: P_threshold = 1 - q_hat
        # [TR] Gerekli minimum olasılık barajını hesapla: P_baraj = 1 - q_hat
        prob_threshold = 1.0 - self.q_hat
        prediction_sets = []
        
        for probs in probabilities:
            # [EN] Include classes with probability >= 1 - q_hat
            # [TR] Olasılığı >= 1 - q_hat şartını sağlayan sınıfları kümeye al
            included_indices = np.where(probs >= prob_threshold)[0].tolist()
            prediction_sets.append(included_indices)
            
        return prediction_sets