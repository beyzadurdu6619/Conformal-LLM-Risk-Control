import os
import numpy as np
import matplotlib.pyplot as plt

def run_full_suite_and_generate_assets():
    print("=" * 85)
    print("🎯 CONFORMAL PREDICTION: FULL ANALYSIS & README ASSET GENERATOR")
    print("🎯 UYUMLU TAHMİN: DETAYLI ANALİZ VE README GÖRSEL OLUŞTURUCU")
    print("=" * 85)

    # 1. Görsellerin Kaydedileceği Klasörü Oluştur
    output_dir = "assets"
    os.makedirs(output_dir, exist_ok=True)

    # 2. Sınıflar ve Senaryolar
    classes = np.array(["Fox Squirrel", "Gray Fox", "Bucket", "Rain Barrel", "Marmot"])
    
    scenarios = [
        {
            "id": "scenario_1_easy",
            "title_tr": "Senaryo 1: Kolay Girdi (Yüksek Güven)",
            "title_en": "Scenario 1: Easy Input (High Confidence)",
            "probs": np.array([0.85, 0.10, 0.03, 0.01, 0.01])
        },
        {
            "id": "scenario_2_ambiguous",
            "title_tr": "Senaryo 2: Belirsiz Girdi (Kararsızlık)",
            "title_en": "Scenario 2: Ambiguous Input (Uncertain Model)",
            "probs": np.array([0.40, 0.35, 0.15, 0.08, 0.02])
        },
        {
            "id": "scenario_3_hard",
            "title_tr": "Senaryo 3: Zor Girdi (Homojen Belirsizlik)",
            "title_en": "Scenario 3: Hard-Uniform Input (High Uncertainty)",
            "probs": np.array([0.22, 0.21, 0.20, 0.19, 0.18])
        }
    ]

    # Eşik Değerleri (Calibration Quantiles)
    prob_threshold = 0.30   # Basic CP için baraj olasılık değeri (1 - qhat_basic)
    qhat_aps = 0.85         # APS Kümülatif Eşiği (%85 kümülatif kütle hedefi)

    # -------------------------------------------------------------------------
    # TERMINAL ÇIKTISI VE ANALİZ DÖNGÜSÜ
    # -------------------------------------------------------------------------
    for item in scenarios:
        probs = item["probs"]
        print(f"\n📌 {item['title_en']} / {item['title_tr']}")
        print("   Model Output Probabilities / Model Olasılık Çıktıları:")
        for c, p in zip(classes, probs):
            print(f"      - {c:13s}: {p:.2f}")

        # --- YÖNTEM 1: BASIC CONFORMAL PREDICTION ---
        basic_set = classes[probs >= prob_threshold]

        # --- YÖNTEM 2: ADAPTIVE PREDICTION SETS (APS) ---
        sort_idx = np.argsort(probs)[::-1]
        sorted_probs = probs[sort_idx]
        sorted_classes = classes[sort_idx]
        cumsum_probs = np.cumsum(sorted_probs)
        cutoff = np.searchsorted(cumsum_probs, qhat_aps) + 1
        aps_set = sorted_classes[:cutoff]

        # --- TERMINAL ANALİZ VE KARŞILAŞTIRMA ---
        print("\n   RESULTS / SONUÇLAR:")
        print(f"   [1] Basic CP Set  (Prob >= {prob_threshold:.2f}) : {list(basic_set)} | Size/Boyut: {len(basic_set)}")
        print(f"   [2] APS (Cumulative) Set (CumSum <= {qhat_aps:.2f}) : {list(aps_set)} | Size/Boyut: {len(aps_set)}")
        
        print("   💡 COMPARATIVE INTERPRETATION / KARŞILAŞTIRMALI YORUM:")
        if len(basic_set) == 0:
            print("      ⚠️ Standart CP HİÇBİR sınıfı seçemedi (Boş Küme)! Sabit eşik çöktü.")
            print("      🚀 APS ise kümülatif kütle yöntemiyle 5 sınıfın hepsini alarak %90 garantiyi KORUDU.")
        elif len(basic_set) == len(aps_set):
            print("      ✅ Her iki yöntem de model net olduğu için tek elemanlı güvenli küme üretti.")
        else:
            print(f"      🚀 Model kararsızlaştıkça APS küme boyutunu esneterek {len(aps_set)} elemana çıkardı.")
            print(f"         Standart CP ise katı baraj uygulayarak bazı olası sınıfları kaçırma riski oluşturdu.")
        print("-" * 85)

        # ---------------------------------------------------------------------
        # TEKİL GRAFİK OLUŞTURMA VE KAYDETME
        # ---------------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(7, 4.8))
        bar_colors = ['#2ecc71' if i < cutoff else '#bdc3c7' for i in range(len(sorted_classes))]
        
        bars = ax.bar(sorted_classes, sorted_probs, color=bar_colors, edgecolor='black', alpha=0.85)
        ax.plot(sorted_classes, cumsum_probs, color='#e74c3c', marker='o', linewidth=2, label='Cumulative Sum')
        ax.axhline(y=qhat_aps, color='#34495e', linestyle='--', linewidth=1.5, label=f'Threshold (q̂={qhat_aps})')

        ax.set_title(f"{item['title_en']}\n({item['title_tr']})", fontsize=10, fontweight='bold')
        ax.set_ylabel('Probability / Cumulative Mass')
        ax.set_ylim(0, 1.15)
        ax.tick_params(axis='x', rotation=25)
        ax.grid(axis='y', linestyle=':', alpha=0.6)
        ax.legend(loc='upper right', fontsize=8)

        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        single_file_path = os.path.join(output_dir, f"{item['id']}.png")
        plt.savefig(single_file_path, dpi=300)
        plt.close()

    # -------------------------------------------------------------------------
    # 3 BİRLEŞİK PANELDEN OLUŞAN GENİŞ GRAFİĞİ KAYDETME
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Adaptive Prediction Sets (APS) vs Basic Conformal Prediction\n'
                 '(Green/Yeşil: Included in Set | Grey/Gri: Excluded)', fontsize=13, fontweight='bold')

    for idx, item in enumerate(scenarios):
        ax = axes[idx]
        probs = item["probs"]
        sort_idx = np.argsort(probs)[::-1]
        sorted_probs = probs[sort_idx]
        sorted_classes = classes[sort_idx]
        cumsum_probs = np.cumsum(sorted_probs)
        cutoff = np.searchsorted(cumsum_probs, qhat_aps) + 1

        bar_colors = ['#2ecc71' if i < cutoff else '#bdc3c7' for i in range(len(sorted_classes))]
        bars = ax.bar(sorted_classes, sorted_probs, color=bar_colors, edgecolor='black', alpha=0.85)
        ax.plot(sorted_classes, cumsum_probs, color='#e74c3c', marker='o', linewidth=2, label='Cumulative Sum')
        ax.axhline(y=qhat_aps, color='#34495e', linestyle='--', linewidth=1.5, label=f'Threshold (q̂={qhat_aps})')

        ax.set_title(item["title_en"], fontsize=10, fontweight='bold')
        ax.set_ylabel('Probability / Cumulative Mass')
        ax.set_ylim(0, 1.15)
        ax.tick_params(axis='x', rotation=25)
        ax.grid(axis='y', linestyle=':', alpha=0.6)
        ax.legend(loc='upper right', fontsize=8)

        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    combined_path = os.path.join(output_dir, "all_scenarios_combined.png")
    plt.savefig(combined_path, dpi=300)
    plt.close()

    print("\n🎉 İŞLEM TAMAMLANDI!")
    print(f"📁 Kaydedilen Görseller ('{output_dir}/' Klasöründe):")
    print("   1. scenario_1_easy.png")
    print("   2. scenario_2_ambiguous.png")
    print("   3. scenario_3_hard.png")
    print("   4. all_scenarios_combined.png")

if __name__ == "__main__":
    run_full_suite_and_generate_assets()