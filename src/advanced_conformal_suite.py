import os
import numpy as np
import matplotlib.pyplot as plt

# Tekrarlanabilirlik için sabit tohum
np.random.seed(42)

def run_advanced_conformal_suite():
    print("=" * 80)
    print(" 🚀 ADVANCED CONFORMAL PREDICTION SUITE (BÖLÜM 4.1 - 4.5)")
    print("=" * 80)

    os.makedirs("assets", exist_ok=True)
    alpha = 0.10  # %90 Hedef Güvenlik / Kapsama (1 - alpha)

    # -------------------------------------------------------------------------
    # 1. Group-Balanced & Class-Conditional Conformal (Bölüm 4.1 & 4.2)
    # -------------------------------------------------------------------------
    print("\n[1/3] Group-Balanced (4.1) & Class-Conditional (4.2) Simüle Ediliyor...")
    
    n_samples = 1200
    # Sentetik Dengesiz Veri: Sınıf 0 (%70), Sınıf 1 (%20), Sınıf 2 (%10)
    y_true = np.random.choice([0, 1, 2], size=n_samples, p=[0.70, 0.20, 0.10])
    
    # Modelin ürettiği olasılıklar (Gürültülü / Dengesiz Başarı)
    probs = np.zeros((n_samples, 3))
    for i, label in enumerate(y_true):
        base = np.random.dirichlet([1, 1, 1])
        base[label] += np.random.uniform(1.5, 3.0)  # Gerçek sınıfa yüksek olasılık
        probs[i] = base / np.sum(base)

    # Ayrı q_hat eşikleri hesaplama (Class-Conditional / Group-Balanced)
    q_hats = {}
    for k in range(3):
        mask_k = (y_true == k)
        scores_k = 1.0 - probs[mask_k, k]
        n_k = len(scores_k)
        q_level = np.ceil((n_k + 1) * (1 - alpha)) / n_k
        q_hats[k] = np.quantile(scores_k, np.clip(q_level, 0, 1), method='higher')

    # Standart vs Class-Conditional Kapsamaları
    all_scores = 1.0 - probs[np.arange(n_samples), y_true]
    q_hat_standard = np.quantile(all_scores, (1 - alpha), method='higher')

    std_coverages = [np.mean((1.0 - probs[y_true == k, k]) <= q_hat_standard) * 100 for k in range(3)]
    cc_coverages = [np.mean((1.0 - probs[y_true == k, k]) <= q_hats[k]) * 100 for k in range(3)]

    # -------------------------------------------------------------------------
    # 2. Conformal Risk Control & Outlier Detection (Bölüm 4.3 & 4.4)
    # -------------------------------------------------------------------------
    print("[2/3] Risk Control (4.3) & Outlier Detection (4.4) Simüle Ediliyor...")
    
    # Risk Control (Segmentasyon / Multilabel Loss)
    lambdas = np.linspace(0.1, 0.9, 50)
    # Lambda büyüdükçe kayıp (Loss) düşer (monotonik kayıp)
    empirical_risks = 1.0 - (1.0 / (1.0 + np.exp(-5 * (lambdas - 0.4))))
    
    # B = 1, n = 1000 için Risk Eşiği
    risk_threshold = alpha - (1.0 - alpha) / 1000.0
    idx_selected = np.where(empirical_risks <= risk_threshold)[0][0]
    selected_lambda = lambdas[idx_selected]

    # Outlier Detection (Anomali Tespiti)
    clean_scores = np.random.gamma(shape=2.0, scale=1.0, size=1000)
    q_hat_outlier = np.quantile(clean_scores, 1 - alpha)
    
    test_inliers = np.random.gamma(shape=2.0, scale=1.0, size=300)
    test_outliers = np.random.gamma(shape=6.0, scale=1.5, size=100)

    # -------------------------------------------------------------------------
    # 3. Covariate Shift Under Conformal (Bölüm 4.5)
    # -------------------------------------------------------------------------
    print("[3/3] Covariate Shift / Ağırlıklı Conformal (4.5) Simüle Ediliyor...")
    
    # Kalibrasyon (Dağılım P) vs Test (Dağılım P_test)
    calib_x = np.random.normal(loc=0.0, scale=1.0, size=1000)
    test_x = np.random.normal(loc=1.5, scale=1.0, size=500) # Kayma var!

    # Likelihood Ratio w(x) = P_test(x) / P(x)
    weights_calib = np.exp(-0.5 * ((calib_x - 1.5)**2 - calib_x**2))
    weights_calib /= np.sum(weights_calib)

    scores_shift = np.abs(calib_x + np.random.normal(0, 0.5, 1000))
    sorted_idx = np.argsort(scores_shift)
    sorted_scores = scores_shift[sorted_idx]
    sorted_weights = weights_calib[sorted_idx]

    cum_weights = np.cumsum(sorted_weights)
    q_hat_weighted = sorted_scores[np.searchsorted(cum_weights, 1 - alpha)]
    q_hat_unweighted = np.quantile(scores_shift, 1 - alpha)

    # =========================================================================
    # GÖRSEL 1: Class-Conditional & Risk Control Analizi
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Grafik 1.1: Standart vs Class-Conditional Coverage
    x_cls = np.arange(3)
    width = 0.35
    axes[0].bar(x_cls - width/2, std_coverages, width, label='Standart Conformal (Adaletsiz)', color='crimson', alpha=0.8)
    axes[0].bar(x_cls + width/2, cc_coverages, width, label='Class-Conditional (Adil)', color='seagreen', alpha=0.8)
    axes[0].axhline(90, color='black', linestyle='--', label='Hedef %90 Kapsama')
    axes[0].set_title("Bölüm 4.1/4.2: Sınıf Dengesizliğinde Kapsama Garantisi", fontsize=11, fontweight='bold')
    axes[0].set_xlabel("Sınıf Etiketi (Sınıf 2 Nadir Vaka)")
    axes[0].set_ylabel("Ampirik Kapsama Oranı (%)")
    axes[0].set_xticks(x_cls)
    axes[0].set_xticklabels(['Sınıf 0 (%70)', 'Sınıf 1 (%20)', 'Sınıf 2 (%10)'])
    axes[0].set_ylim(0, 110)
    axes[0].legend(loc='lower right')
    axes[0].grid(True, alpha=0.3)

    # Grafik 1.2: Conformal Risk Control
    axes[1].plot(lambdas, empirical_risks, color='darkblue', linewidth=2.5, label='Ampirik Risk R(λ)')
    axes[1].axhline(alpha, color='red', linestyle='--', label=f'Risk Sınırı α = {alpha}')
    axes[1].axvline(selected_lambda, color='green', linestyle=':', linewidth=2, label=f'Seçilen λ̂ = {selected_lambda:.2f}')
    axes[1].set_title("Bölüm 4.3: Conformal Risk Control (Monotonik Kayıp)", fontsize=11, fontweight='bold')
    axes[1].set_xlabel("Sıkılık Parametresi (λ)")
    axes[1].set_ylabel("Beklenen Ortalama Kayıp E[Loss]")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    chart1_path = "assets/advanced_section4_part1.png"
    plt.savefig(chart1_path, dpi=300, bbox_inches='tight')
    plt.close()

    # =========================================================================
    # GÖRSEL 2: Outlier Detection & Covariate Shift
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Grafik 2.1: Outlier Detection
    axes[0].hist(test_inliers, bins=20, alpha=0.6, color='royalblue', label='Normal Veri (Inliers)')
    axes[0].hist(test_outliers, bins=20, alpha=0.6, color='crimson', label='Aykırı Veri (Outliers)')
    axes[0].axvline(q_hat_outlier, color='black', linestyle='--', linewidth=2, label=f'Eşik q̂ = {q_hat_outlier:.2f}')
    axes[0].set_title("Bölüm 4.4: Etiketsiz Anomali Tespiti (Outlier Detection)", fontsize=11, fontweight='bold')
    axes[0].set_xlabel("Anomali Skoru s(x)")
    axes[0].set_ylabel("Örnek Sayısı")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Grafik 2.2: Covariate Shift
    axes[1].hist(calib_x, bins=25, alpha=0.5, color='gray', density=True, label='Eski Kalibrasyon P(X)')
    axes[1].hist(test_x, bins=25, alpha=0.5, color='orange', density=True, label='Yeni Test P_test(X)')
    axes[1].axvline(q_hat_unweighted, color='crimson', linestyle='--', label=f'Ağırlıksız Eşik = {q_hat_unweighted:.2f}')
    axes[1].axvline(q_hat_weighted, color='seagreen', linestyle='-', linewidth=2, label=f'Ağırlıklı Eşik (Shift-Adjusted) = {q_hat_weighted:.2f}')
    axes[1].set_title("Bölüm 4.5: Dağılım Kayması Altında Conformal (Covariate Shift)", fontsize=11, fontweight='bold')
    axes[1].set_xlabel("Girdi Özelliği X")
    axes[1].set_ylabel("Yoğunluk")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    chart2_path = "assets/advanced_section4_part2.png"
    plt.savefig(chart2_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n ✅ TÜM İLERİ SEVİYE GÖRSELLER BAŞARIYLA ÜRETİLDİ:")
    print(f"    - Görsel 1: {chart1_path}")
    print(f"    - Görsel 2: {chart2_path}")

if __name__ == "__main__":
    run_advanced_conformal_suite()