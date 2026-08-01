import numpy as np
import matplotlib.pyplot as plt

def run_full_conformal_suite():
    print("=" * 85)
    print("🎯 CONFORMAL PREDICTION: FULL ANALYSIS & GRAPHICAL VISUALIZATION")
    print("🎯 UYUMLU TAHMİN: DETAYLI ANALİZ VE GRAFİKSEL GÖRSELLEŞTİRME")
    print("=" * 85)

    # -------------------------------------------------------------------------
    # 1. SETUP & DATASETS / KURULUM VE VERİ KÜMELERİ
    # -------------------------------------------------------------------------
    classes = np.array(["Fox Squirrel", "Gray Fox", "Bucket", "Rain Barrel", "Marmot"])
    
    # 3 Senaryo Olasılık Dağılımları (Kolay, Belirsiz, Zor)
    scenarios = {
        "Scenario 1: Easy Input / Kolay Girdi": np.array([0.85, 0.10, 0.03, 0.01, 0.01]),
        "Scenario 2: Ambiguous Input / Belirsiz Girdi": np.array([0.40, 0.35, 0.15, 0.08, 0.02]),
        "Scenario 3: Hard-Uniform Input / Zor Girdi": np.array([0.22, 0.21, 0.20, 0.19, 0.18])
    }

    # Eşik Değerleri (Calibration Quantiles)
    # Target Alpha = 0.10 (%90 Coverage Guarantee / %90 Kapsama Garantisi)
    qhat_basic = 0.70       # Basic CP Quantile (Prob Threshold = 1 - 0.70 = 0.30)
    prob_threshold = 0.30   # Standart CP için baraj olasılık değeri
    qhat_aps = 0.85         # APS Kümülatif Eşiği (%85 kümülatif kütle hedefi)

    # Matplotlib Grafik Penceresi Hazırlığı (1 Satır, 3 Sütun)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle('Adaptive Prediction Sets (APS) vs Basic Conformal Prediction\n'
                 '(Yeşil / Green: Kümeye Alınanlar | Gri / Grey: Dışarıda Kalanlar)', 
                 fontsize=13, fontweight='bold')

    # -------------------------------------------------------------------------
    # 2. EXECUTION, TEXT ANALYSIS & PLOTTING / HESAPLAMA, ANALİZ VE ÇİZİM
    # -------------------------------------------------------------------------
    for ax_idx, (title, probs) in enumerate(scenarios.items()):
        ax = axes[ax_idx]
        print(f"\n📌 {title}")
        print("   Model Output Probabilities / Model Olasılık Çıktıları:")
        for c, p in zip(classes, probs):
            print(f"      - {c:13s}: {p:.2f}")

        # --- YÖNTEM 1: BASIC CONFORMAL PREDICTION ---
        basic_set = classes[probs >= prob_threshold]

        # --- YÖNTEM 2: ADAPTIVE PREDICTION SETS (APS) ---
        # 1. Büyükten küçüğe sırala
        sort_idx = np.argsort(probs)[::-1]
        sorted_probs = probs[sort_idx]
        sorted_classes = classes[sort_idx]

        # 2. Kümülatif toplamı al (Cumulative Sum)
        cumsum_probs = np.cumsum(sorted_probs)

        # 3. Kümeye dahil edilecek eleman sayısını bul (Cutoff)
        cutoff = np.searchsorted(cumsum_probs, qhat_aps) + 1
        aps_set = sorted_classes[:cutoff]

        # --- TERMINAL ANALİZ ÇIKTILARI ---
        print("\n   RESULTS / SONUÇLAR:")
        print(f"   [1] Basic CP Set  (Prob >= {prob_threshold:.2f}) : {list(basic_set)} (Boyut/Size: {len(basic_set)})")
        print(f"   [2] APS (Cumulative) Set (CumSum <= {qhat_aps:.2f}) : {list(aps_set)} (Boyut/Size: {len(aps_set)})")
        
        print("   💡 INTERPRETATION / YORUM:")
        if len(basic_set) == 0:
            print("      ⚠️ Standart CP HİÇBİR sınıfı seçemedi (Boş Küme)! Sabit eşik çöktü.")
            print("      🚀 APS ise kümülatif kütle ile 5 sınıfın hepsini alarak %90 garantiyi KORUDU.")
        elif len(basic_set) == len(aps_set):
            print("      ✅ Her iki yöntem de yüksek emniyetle tekli/benzer tahmin kümesi üretti.")
        else:
            print(f"      🚀 APS modelin belirsizliğine göre küme boyutunu dinamik olarak {len(aps_set)} elemana çıkardı.")
        print("-" * 85)

        # --- GRAFİK ÇİZİMİ (VISUALIZATION) ---
        # Kümeye girenler YEŞİL (#2ecc71), dışarıda kalanlar GRİ (#bdc3c7)
        bar_colors = ['#2ecc71' if i < cutoff else '#bdc3c7' for i in range(len(sorted_classes))]
        
        # Olasılık Barları
        bars = ax.bar(sorted_classes, sorted_probs, color=bar_colors, edgecolor='black', alpha=0.85)
        
        # Kümülatif Çizgi ve Eşik Çizgisi
        ax.plot(sorted_classes, cumsum_probs, color='#e74c3c', marker='o', linewidth=2, label='Cumulative Sum')
        ax.axhline(y=qhat_aps, color='#34495e', linestyle='--', linewidth=1.5, label=f'Threshold (q̂={qhat_aps})')

        # Grafik Detayları
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_ylabel('Probability / Cumulative Mass')
        ax.set_ylim(0, 1.15)
        ax.tick_params(axis='x', rotation=30)
        ax.grid(axis='y', linestyle=':', alpha=0.6)
        ax.legend(loc='upper right', fontsize=8)

        # Bar üzerindeki değer etiketleri
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    print("\n🎨 Grafik oluşturuldu! Görsel penceresi açılıyor...")
    plt.show()

if __name__ == "__main__":
    run_full_conformal_suite()