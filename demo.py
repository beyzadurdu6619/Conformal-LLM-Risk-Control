import numpy as np

def run_conformal_demo():
    print("=" * 60)
    print("🎯 CONFORMAL PREDICTION (RISK CONTROL) DEMO RUNNER")
    print("=" * 60)

    # 1. Simüle Edilmiş LLM / Sınıflandırıcı Olasılık Çıktıları (5 Örnek, 4 Sınıf)
    # Sınıflar: [0: 'Yanlış/Riskli', 1: 'Kısmen Doğru', 2: 'Doğru', 3: 'Mükemmel']
    classes = ["Kritik Hata", "Zayıf Yanıt", "Kabul Edilebilir", "Tam Doğru"]
    
    # Modelin ürettiği olasılık tahminleri (Probabilities)
    test_probs = np.array([
        [0.05, 0.10, 0.70, 0.15],  # Örnek 1: Model 2. sınıftan oldukça emin (%70)
        [0.40, 0.35, 0.15, 0.10],  # Örnek 2: Model kararsız (0 ve 1 arasında kalmış)
        [0.01, 0.04, 0.05, 0.90],  # Örnek 3: Model 3. sınıftan çok emin (%90)
        [0.25, 0.25, 0.25, 0.25],  # Örnek 4: Tam belirsizlik (Model hiçbir şey bilmiyor)
    ])

    # 2. Risk Toleransı (Alpha) Belirleme
    # alpha = 0.10 -> %90 Güven Düzeyi (Hatalı sınıfı kaçırma riski en fazla %10)
    alpha = 0.10
    coverage_target = (1 - alpha) * 100

    print(f"\n📊 Hata Toleransı (Alpha): {alpha} | Hedef Güven Düzeyi: %{coverage_target:.0f}\n")
    print("-" * 60)

    # 3. Conformal Set (Tahmin Kümesi) Oluşturma
    # Basit bir 1-p(y) uyumsuzluk skoru eşiği simülasyonu
    threshold = 1 - alpha # Eşik değeri

    for i, probs in enumerate(test_probs):
        # Eşiği geçen sınıfları küme olarak seç
        prediction_set = [classes[j] for j, p in enumerate(probs) if p >= (1 - threshold)]
        
        print(f"🔹 Örnek {i+1}:")
        print(f"   Model Olasılıkları : {dict(zip(classes, np.round(probs, 2)))}")
        print(f"   Tahmin Kümesi (Set) : {prediction_set}")
        print(f"   Küme Genişliği     : {len(prediction_set)} alternatif yanıt")
        
        # Yorumlama
        if len(prediction_set) == 1:
            print("   💡 YORUM: Model bu yanıttan YÜKSEK GÜVENLE emin. İnsan onayına gerek yok.")
        elif len(prediction_set) > 1:
            print("   ⚠️ YORUM: Model KARARSIZ. Risk kontrolü gereği birden fazla güvenli seçenek sunuldu.")
        else:
            print("   🚨 YORUM: Hiçbir yanıt güvenlik eşiğini geçemedi! İstisna/Fallback mekanizması tetiklenmeli.")
        print("-" * 60)

if __name__ == "__main__":
    run_conformal_demo()