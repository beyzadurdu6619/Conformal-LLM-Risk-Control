import numpy as np
import matplotlib.pyplot as plt
import os
# Tekrarlanabilirlik için rastgelelik tohumu
np.random.seed(42)

def run_proof_experiments():
    print("=" * 80)
    print(" 🧪 CONFORMAL PREDICTION: TEORİK KANIT VE KARŞILAŞTIRMA SİMÜLASYONU")
    print("=" * 80)
    
    # -------------------------------------------------------------------------
    # 1. KANIT: Skaler Belirsizlik vs. Standart Sabit Aralık (Bant Esnemesi)
    # -------------------------------------------------------------------------
    print("\n[1] SKALER BELİRSİZLİK İLE DİNAMİK BANT ESNEMESİ TEST EDİLİYOR...")
    
    n_calib = 1000
    n_test = 200
    alpha = 0.10  # %90 Kapsama Hedefi
    
    # Girdi (X): 1 ile 10 arasında değerler
    X_calib = np.random.uniform(1, 10, n_calib)
    # Gerçek Hata (Heteroskedastic): X büyüdükçe gürültü/belirsizlik de büyüyor!
    noise_calib = np.random.normal(0, scale=0.5 * X_calib)
    y_calib = 2 * X_calib + noise_calib
    
    # Model Tahminleri
    f_hat_calib = 2 * X_calib  # Nokta Tahmini
    u_x_calib = 0.5 * X_calib  # Modelin Kararsızlık Hissi (X büyüdükçe belirsizlik artıyor)
    
    # Uyumsuzluk Skoru: s(x, y) = |y - f_hat(x)| / u(x)
    scores = np.abs(y_calib - f_hat_calib) / u_x_calib
    q_hat = np.quantile(scores, np.ceil((n_calib + 1) * (1 - alpha)) / n_calib)
    
    print(f" -> Hesaplanan Otomatik Eşik Çarpanı (q_hat): {q_hat:.4f}")
    
    # Test Aşaması
    X_test = np.linspace(1, 10, n_test)
    f_hat_test = 2 * X_test
    u_x_test = 0.5 * X_test
    
    # Dinamik Conformal Aralığı: [f_hat - u(x)*q_hat, f_hat + u(x)*q_hat]
    lower_bound = f_hat_test - u_x_test * q_hat
    upper_bound = f_hat_test + u_x_test * q_hat
    
    # Sabit (Esnemeyen) Standart Sapma Aralığı (Klasik İstatistik)
    fixed_std = np.std(y_calib - f_hat_calib)
    fixed_lower = f_hat_test - 1.645 * fixed_std
    fixed_upper = f_hat_test + 1.645 * fixed_std

    # -------------------------------------------------------------------------
    # 2. KANIT: Marjinal Kapsama vs. Koşullu Kapsama (Adalet Testi - FSC)
    # -------------------------------------------------------------------------
    print("\n[2] ADALET TESTİ: MARJİNAL VS. KOŞULLU KAPSAMA (FSC METRİĞİ)...")
    
    # Nüfus: %90 Grup A (Kolay/Düşük Gürültü), %10 Grup B (Zor/Yüksek Gürültü)
    n_group_A = 900
    n_group_B = 100
    
    # Grup A (Kolay): Hata varyansı çok küçük
    errors_A = np.random.normal(0, 0.5, n_group_A)
    # Grup B (Zor): Hata varyansı çok büyük
    errors_B = np.random.normal(0, 4.0, n_group_B)
    
    # Kötü / Hatalı Bir Sabit Aralık Yöntemi (Sadece ortalamada %90 sağlıyor)
    # Sabit Eşik: 1.2 birim
    threshold_bad = 1.2
    
    cov_A_bad = np.mean(np.abs(errors_A) <= threshold_bad)
    cov_B_bad = np.mean(np.abs(errors_B) <= threshold_bad)
    marginal_cov_bad = (n_group_A * cov_A_bad + n_group_B * cov_B_bad) / (n_group_A + n_group_B)
    
    # FSC Metriği (En kötü performans gösteren alt grubun kapsaması)
    fsc_metric_bad = min(cov_A_bad, cov_B_bad)
    
    print(f" ❌ KÖTÜ ALGORİTMA (Marjinal var, Koşullu Kapsama YOK):")
    print(f"    - Genel (Marjinal) Kapsama : %{marginal_cov_bad * 100:.1f} (Hedef %90'a yakın görünüyor!)")
    print(f"    - Grup A (Kolay) Kapsaması : %{cov_A_bad * 100:.1f}")
    print(f"    - Grup B (Zor) Kapsaması   : %{cov_B_bad * 100:.1f}  <-- ADALETSİZLİK/FELAKET!")
    print(f"    - FSC Metriği (En Zayıf Group): %{fsc_metric_bad * 100:.1f}")

    # Doğru Adaptif Conformal Yöntem (Grupların zorluğunu dikkate alan)
    # Her gruba kendi u(x) belirsizliğiyle yaklaşıyoruz
    u_A = np.full(n_group_A, 0.5)
    u_B = np.full(n_group_B, 4.0)
    
    all_scores = np.concatenate([np.abs(errors_A) / u_A, np.abs(errors_B) / u_B])
    q_hat_fair = np.quantile(all_scores, 0.90)
    
    cov_A_good = np.mean(np.abs(errors_A) <= u_A * q_hat_fair)
    cov_B_good = np.mean(np.abs(errors_B) <= u_B * q_hat_fair)
    fsc_metric_good = min(cov_A_good, cov_B_good)
    
    print(f"\n ✅ İDEAL ADAPTİF CONFORMAL ALGORİTMA (Koşullu Kapsama Var):")
    print(f"    - Grup A (Kolay) Kapsaması : %{cov_A_good * 100:.1f}")
    print(f"    - Grup B (Zor) Kapsaması   : %{cov_B_good * 100:.1f}  <-- HER İKİ GRUP DA KORUNDU!")
    print(f"    - FSC Metriği (En Zayıf Group): %{fsc_metric_good * 100:.1f}")

    # -------------------------------------------------------------------------
    # 3. GÖRSELLEŞTİRME: Tüm Kanıtların Grafiğe Dökülmesi
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Grafik 1: Skaler Belirsizlikle Esneyen Tahmin Bandı
    axes[0].plot(X_test, f_hat_test, color='black', label=r'Nokta Tahmini $\hat{f}(x)$')
    axes[0].fill_between(X_test, lower_bound, upper_bound, color='deepskyblue', alpha=0.4, 
                         label=r'Conformal Esnek Bant ($\hat{f}(x) \pm q_{\hat{q}}u(x)$)')
    axes[0].plot(X_test, fixed_lower, color='red', linestyle='--', label='Sabit Enli Klasik Bant (Esnemeyen)')
    axes[0].plot(X_test, fixed_upper, color='red', linestyle='--')
    axes[0].set_title("Kanıt 1: Belirsizliğe Göre Esneyen Tahmin Bandı", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Girdi (X) - Zorluk Artıyor ➔")
    axes[0].set_ylabel("Sayısal Çıktı (Y)")
    axes[0].legend(loc='upper left')
    axes[0].grid(True, alpha=0.3)
    
    # Grafik 2: Adalet ve Koşullu Kapsama (FSC Metriği) Kıyaslaması
    groups = ['Grup A (Kolay)', 'Grup B (Zor)']
    bad_coverages = [cov_A_bad * 100, cov_B_bad * 100]
    good_coverages = [cov_A_good * 100, cov_B_good * 100]
    
    x = np.arange(len(groups))
    width = 0.35
    
    axes[1].bar(x - width/2, bad_coverages, width, label='Adaletsiz / Sabit Yöntem', color='crimson', alpha=0.8)
    axes[1].bar(x + width/2, good_coverages, width, label='Adil Conformal Yöntem', color='seagreen', alpha=0.8)
    axes[1].axhline(90, color='black', linestyle=':', label='Hedef %90 Kapsama')
    axes[1].set_title("Kanıt 2: Gruplar Arası Adalet (FSC Metriği)", fontsize=12, fontweight='bold')
    axes[1].set_ylabel("Ampirik Kapsama (%)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(groups)
    axes[1].set_ylim(0, 110)
    axes[1].legend(loc='lower right')
    axes[1].grid(True, alpha=0.3)
      
    os.makedirs("assets", exist_ok=True)
    plt.savefig("assets/proof_of_concepts.png", dpi=300, bbox_inches='tight')
    print("\n 🖼️ Grafik başarıyla 'assets/proof_of_concepts.png' konumuna kaydedildi!")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_proof_experiments()