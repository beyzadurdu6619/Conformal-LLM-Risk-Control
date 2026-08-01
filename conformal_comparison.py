import os
import numpy as np
import matplotlib.pyplot as plt

def generate_and_save_readme_assets():
    print("=" * 85)
    print("🎯 CONFORMAL PREDICTION: GENERATING README ASSETS")
    print("🎯 UYUMLU TAHMİN: README GÖRSELLERİ OLUŞTURULUYOR")
    print("=" * 85)

    # Görsellerin kaydedileceği klasörü oluştur
    output_dir = "assets"
    os.makedirs(output_dir, exist_ok=True)

    classes = np.array(["Fox Squirrel", "Gray Fox", "Bucket", "Rain Barrel", "Marmot"])
    
    scenarios = [
        {
            "id": "scenario_1_easy",
            "title_tr": "Senaryo 1: Kolay Girdi (High Confidence)",
            "title_en": "Scenario 1: Easy Input",
            "probs": np.array([0.85, 0.10, 0.03, 0.01, 0.01])
        },
        {
            "id": "scenario_2_ambiguous",
            "title_tr": "Senaryo 2: Belirsiz Girdi (Ambiguous)",
            "title_en": "Scenario 2: Ambiguous Input",
            "probs": np.array([0.40, 0.35, 0.15, 0.08, 0.02])
        },
        {
            "id": "scenario_3_hard",
            "title_tr": "Senaryo 3: Zor Girdi (Uniform Uncertainty)",
            "title_en": "Scenario 3: Hard-Uniform Input",
            "probs": np.array([0.22, 0.21, 0.20, 0.19, 0.18])
        }
    ]

    qhat_aps = 0.85  # APS Kümülatif Eşiği

    # -------------------------------------------------------------------------
    # 1. HER SENARYOYU AYRI AYRI KAYDET (FOR README SECTIONS)
    # -------------------------------------------------------------------------
    for item in scenarios:
        probs = item["probs"]
        
        # Sıralama & Kümülatif Hesaplama
        sort_idx = np.argsort(probs)[::-1]
        sorted_probs = probs[sort_idx]
        sorted_classes = classes[sort_idx]
        cumsum_probs = np.cumsum(sorted_probs)
        cutoff = np.searchsorted(cumsum_probs, qhat_aps) + 1

        # Grafik Çizimi
        fig, ax = plt.subplots(figsize=(7, 5))
        bar_colors = ['#2ecc71' if i < cutoff else '#bdc3c7' for i in range(len(sorted_classes))]
        
        bars = ax.bar(sorted_classes, sorted_probs, color=bar_colors, edgecolor='black', alpha=0.85)
        ax.plot(sorted_classes, cumsum_probs, color='#e74c3c', marker='o', linewidth=2, label='Cumulative Sum')
        ax.axhline(y=qhat_aps, color='#34495e', linestyle='--', linewidth=1.5, label=f'Threshold (q̂={qhat_aps})')

        ax.set_title(f"{item['title_en']}\n({item['title_tr']})", fontsize=11, fontweight='bold')
        ax.set_ylabel('Probability / Cumulative Mass')
        ax.set_ylim(0, 1.15)
        ax.tick_params(axis='x', rotation=25)
        ax.grid(axis='y', linestyle=':', alpha=0.6)
        ax.legend(loc='upper right', fontsize=9)

        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        
        # Dosyayı kaydet
        file_path = os.path.join(output_dir, f"{item['id']}.png")
        plt.savefig(file_path, dpi=300)
        plt.close()
        print(f"✅ Kaydedildi: {file_path}")

    # -------------------------------------------------------------------------
    # 2. HEPSİNİ TEK BİR GENİŞ PANEL OLARAK KAYDET
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Adaptive Prediction Sets (APS) - Scenario Comparison', fontsize=14, fontweight='bold')

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

        ax.set_title(item["title_en"], fontsize=11, fontweight='bold')
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
    print(f"✅ Birleşik Grafik Kaydedildi: {combined_path}")

    print("\n🎉 Tüm görseller 'assets/' klasörüne başarıyla kaydedildi!")

if __name__ == "__main__":
    generate_and_save_readme_assets()