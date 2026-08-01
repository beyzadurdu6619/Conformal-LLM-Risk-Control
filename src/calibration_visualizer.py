import os
import numpy as np
import matplotlib.pyplot as plt

def run_and_visualize_day2_pipeline():
    print("=" * 80)
    print("🚀 DAY 2: AUTOMATIC QUANTILE CALIBRATION & MODEL COMPARISON")
    print("🚀 2. GÜN: OTOMATİK KUANTİL KALİBRASYONU VE MODEL KARŞILAŞTIRMASI")
    print("=" * 80)

    np.random.seed(42)
    n_calib = 1000  # Calibration set size / Kalibrasyon boyutu
    n_test = 1000   # Test set size / Test boyutu
    n_classes = 10  # 10 Classes / 10 Sınıf
    alpha = 0.10    # Target Error Rate (Coverage Target = 90%) / Hedef Hata Oranı (%90)

    # -------------------------------------------------------------------------
    # 1. MODEL SIMULATIONS / MODEL SİMÜLASYONLARI
    # -------------------------------------------------------------------------
    # Model A (Eğitilmiş / Yüksek Güvenli Model): Doğru sınıfa yüksek olasılık verir
    # Model B (Zayıf / Kararsız Model): Olasılıkları düzensiz ve yayvan dağıtır
    def generate_model_outputs(model_type, n_samples):
        probs_list, labels_list = [], []
        for _ in range(n_samples):
            label = np.random.randint(0, n_classes)
            probs = np.random.dirichlet(np.ones(n_classes) * 0.5)
            if model_type == "good":
                # Doğru sınıfa baskın olasılık ata
                probs[label] += np.random.uniform(2.0, 5.0)
            probs = probs / np.sum(probs)
            probs_list.append(probs)
            labels_list.append(label)
        return np.array(probs_list), np.array(labels_list)

    # Verileri Üret
    calib_probs_good, calib_labels_good = generate_model_outputs("good", n_calib)
    test_probs_good, test_labels_good = generate_model_outputs("good", n_test)

    calib_probs_bad, calib_labels_bad = generate_model_outputs("bad", n_calib)
    test_probs_bad, test_labels_bad = generate_model_outputs("bad", n_test)

    # -------------------------------------------------------------------------
    # 2. CALIBRATION FUNCTION / KALİBRASYON FONKSİYONU
    # -------------------------------------------------------------------------
    def calibrate_and_evaluate(calib_probs, calib_labels, test_probs, test_labels):
        # Skor Hesabı
        calib_scores = []
        for probs, label in zip(calib_probs, calib_labels):
            sort_idx = np.argsort(probs)[::-1]
            sorted_probs = probs[sort_idx]
            label_rank = np.where(sort_idx == label)[0][0]
            cum_score = np.sum(sorted_probs[:label_rank + 1])
            calib_scores.append(cum_score)
        calib_scores = np.array(calib_scores)

        # qhat Hesabı
        quantile_level = np.ceil((n_calib + 1) * (1 - alpha)) / n_calib
        qhat = np.quantile(calib_scores, quantile_level, method='higher')

        # Test Değerlendirmesi
        covered_flags, set_sizes = [], []
        for probs, label in zip(test_probs, test_labels):
            sort_idx = np.argsort(probs)[::-1]
            sorted_probs = probs[sort_idx]
            cumsum_probs = np.cumsum(sorted_probs)
            
            cutoff = np.searchsorted(cumsum_probs, qhat) + 1
            pred_set = sort_idx[:cutoff]
            
            set_sizes.append(len(pred_set))
            covered_flags.append(1 if label in pred_set else 0)

        return calib_scores, qhat, np.mean(covered_flags), set_sizes

    # Her iki modeli de çalıştır
    scores_good, qhat_good, cov_good, sizes_good = calibrate_and_evaluate(
        calib_probs_good, calib_labels_good, test_probs_good, test_labels_good
    )
    scores_bad, qhat_bad, cov_bad, sizes_bad = calibrate_and_evaluate(
        calib_probs_bad, calib_labels_bad, test_probs_bad, test_labels_bad
    )

    # -------------------------------------------------------------------------
    # 3. VISUALIZATION DASHBOARD (3 PANELS WITH COMPARISON)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle('Day 2: Calibration Pipeline - Good vs. Weak Model Comparison\n'
                 '2. Gün: Kalibrasyon Süreci - İyi ve Zayıf Model Karşılaştırması', 
                 fontsize=12, fontweight='bold')

    # --- PANEL 1: Calibration Score Distributions (qhat Comparison) ---
    ax1 = axes[0]
    ax1.hist(scores_good, bins=25, alpha=0.65, color='#2ecc71', edgecolor='black', label='Good Model / İyi Model')
    ax1.hist(scores_bad, bins=25, alpha=0.45, color='#e74c3c', edgecolor='black', label='Weak Model / Zayıf Model')
    
    ax1.axvline(x=qhat_good, color='#27ae60', linestyle='--', linewidth=2.5, label=f'q̂ (Good) = {qhat_good:.3f}')
    ax1.axvline(x=qhat_bad, color='#c0392b', linestyle='--', linewidth=2.5, label=f'q̂ (Weak) = {qhat_bad:.3f}')

    ax1.set_title('1. Calibration Score Distributions & q̂ Thresholds\n1. Kalibrasyon Skor Dağılımları ve q̂ Eşikleri', fontsize=10, fontweight='bold')
    ax1.set_xlabel('APS Non-conformity Score (CumSum) / APS Skoru')
    ax1.set_ylabel('Sample Frequency / Örnek Frekansı')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(axis='y', linestyle=':', alpha=0.6)

    # --- PANEL 2: Coverage Comparison ---
    ax2 = axes[1]
    target_coverage = 1 - alpha
    bars = ax2.bar(['Target / Hedef\n(90%)', 'Good Model\n(İyi)', 'Weak Model\n(Zayıf)'], 
                   [target_coverage, cov_good, cov_bad], 
                   color=['#95a5a6', '#2ecc71', '#3498db'],
                   edgecolor='black', width=0.5, alpha=0.85)
    
    ax2.axhline(y=target_coverage, color='#e74c3c', linestyle='--', linewidth=2, label='Target Boundary (%90)')
    ax2.set_title('2. Empirical Coverage Verification\n2. Kapsama Oranı Doğrulaması', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Coverage Rate / Kapsama Oranı')
    ax2.set_ylim(0, 1.15)
    ax2.grid(axis='y', linestyle=':', alpha=0.6)
    ax2.legend(loc='upper right', fontsize=8)

    for bar in bars:
        height = bar.get_height()
        ax2.annotate(f'%{height*100:.1f}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points",
                     ha='center', va='bottom', fontsize=10, fontweight='bold')

    # --- PANEL 3: Set Size Comparison (Efficiency) ---
    ax3 = axes[2]
    ax3.hist(sizes_good, bins=np.arange(1, n_classes + 2) - 0.5, alpha=0.65, color='#2ecc71', 
             edgecolor='black', label=f'Good Model (Mean = {np.mean(sizes_good):.2f})')
    ax3.hist(sizes_bad, bins=np.arange(1, n_classes + 2) - 0.5, alpha=0.45, color='#e74c3c', 
             edgecolor='black', label=f'Weak Model (Mean = {np.mean(sizes_bad):.2f})')
    
    ax3.set_title('3. Prediction Set Size Distribution (Efficiency)\n3. Tahmin Kümesi Boyut Dağılımı (Verimlilik)', fontsize=10, fontweight='bold')
    ax3.set_xlabel('Set Size / Küme Boyutu (Sınıf Sayısı)')
    ax3.set_ylabel('Test Sample Count / Test Örnek Sayısı')
    ax3.set_xticks(range(1, n_classes + 1))
    ax3.grid(axis='y', linestyle=':', alpha=0.6)
    ax3.legend(loc='upper right', fontsize=8)

    plt.tight_layout()

    output_dir = "assets"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "day2_calibration_pipeline.png")
    plt.savefig(save_path, dpi=300)
    print(f"\n🎨 Graph updated successfully / Görsel başarıyla güncellendi: {save_path}")
    
    plt.show()

if __name__ == "__main__":
    run_and_visualize_day2_pipeline()