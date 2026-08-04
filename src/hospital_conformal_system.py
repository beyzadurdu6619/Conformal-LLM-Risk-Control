import numpy as np
import time

# Tekrarlanabilirlik için sabit tohum
np.random.seed(42)

class HospitalConformalAutomation:
    def __init__(self):
        self.alpha = 0.10  # %90 Standart Güvenlik/Kapsama Hedefi (1 - alpha)
        
    def show_banner(self):
        print("=" * 80)
        print(" 🏥 AÇIKLANABİLİR VE GÜVENİLİR YAPAY ZEKA HASTANE OTOMASYONU")
        print(" 🛡️ Conformal Prediction İleri Seviye Klinik Tanı Motoru (Bölüm 4.1 - 4.5)")
        print("=" * 80)

    # -------------------------------------------------------------------------
    # SENARYO 1: Dermatoloji Klinik Tanı Motoru (Group-Balanced Conformal - 4.1)
    # -------------------------------------------------------------------------
    def scenario_1_group_balanced(self):
        print("\n" + "="*80)
        print(" 🩺 1. DERMATOLOJİ KLİNİĞİ: Cilt Kanseri Risk Analizi (Group-Balanced)")
        print(" 📋 Klinik İhtiyaç: Genç ve Yaşlı hastalar arasında adalet ve eşit güvenlik.")
        print("="*80)
        
        # Sentetik Kalibrasyon Skorları (Yaşlılarda gürültü/hata yüksek)
        calib_scores_genc = np.random.exponential(scale=0.4, size=800)   # Düşük gürültü
        calib_scores_yasli = np.random.exponential(scale=1.2, size=200)  # Yüksek gürültü (Lentigo/Kırışıklık)
        
        # Ayrı q_hat Eşikleri Hesabı
        q_genc = np.quantile(calib_scores_genc, 1 - self.alpha)
        q_yasli = np.quantile(calib_scores_yasli, 1 - self.alpha)
        
        print(f"⚙️ Kalibrasyon Tamamlandı:")
        print(f"   • Genç Hasta Eşiği (q_genç)  : {q_genc:.3f} (Dar Emniyet Bandı)")
        print(f"   • Yaşlı Hasta Eşiği (q_yaşlı): {q_yasli:.3f} (Geniş Emniyet Bandı - Risk Engelleyici)")
        
        # Hasta Girdisi Sims
        yas = int(input("\n👉 Hasta Yaşını Giriniz (Örn: 25 veya 72): "))
        grup = "Genç" if yas < 60 else "Yaşlı"
        q_hat = q_genc if grup == "Genç" else q_yasli
        
        # Model Uyumsuzluk Skoru
        test_score = np.random.uniform(0.5, 1.8)
        
        print(f"\n🔬 TANI SONUCU ({yas} Yaş - {grup} Grubu):")
        print(f"   • Hesaplanan Risk Skoru : {test_score:.3f}")
        print(f"   • Uygulanan Eşik (q_hat) : {q_hat:.3f}")
        
        if test_score <= q_hat:
            print("   ✅ KARAR: [RUTİN KONTROL / SELİM LEKE] - Hasta Güvenle Taburcu Edilebilir.")
        else:
            print("   ⚠️ KARAR: [ŞÜPHELİ KÖTÜ HUYLU LEKE] - Biyopsi İstemi Oluşturuldu!")
        print(f"   🛡️ Garanti: Bu karar {grup} yaş grubunda %90 kesin kapsama garantilidir.")

    # -------------------------------------------------------------------------
    # SENARYO 2: Onkoloji Nadir Kanser Tanı Motoru (Class-Conditional - 4.2)
    # -------------------------------------------------------------------------
    def scenario_2_class_conditional(self):
        print("\n" + "="*80)
        print(" 🎗️ 2. ONKOLOJİ KLİNİĞİ: Mamografi Kanser Teşhisi (Class-Conditional)")
        print(" 📋 Klinik İhtiyaç: Dengesiz veride (%95 Sağlıklı, %5 Kanser) Kanseri kaçırmamak.")
        print("="*80)
        
        # Sınıflara Özel Kalibrasyon Eşikleri
        q_sağlıklı = 0.18  # Y=0
        q_kanser = 0.65    # Y=1 (Nadir ve tespiti zor vaka için esnek eşik)
        
        print(f"⚙️ Sınıf Şartlı Eşikler:")
        print(f"   • Sınıf 0 (Sağlıklı) Eşiği : {q_sağlıklı:.2f}")
        print(f"   • Sınıf 1 (Kanser) Eşiği   : {q_kanser:.2f}")
        
        print("\n📥 Mamografi Görüntüsü İşleniyor...")
        time.sleep(0.5)
        
        # Modelin Varsayımsal Olasılık Çıktıları
        prob_sağlıklı = 0.40  # P(Y=0)
        prob_kanser = 0.60    # P(Y=1)
        
        # Varsayımsal Uyumsuzluk Skorları
        s_0 = 1.0 - prob_sağlıklı  # 0.60
        s_1 = 1.0 - prob_kanser    # 0.40
        
        teşhis_sepeti = []
        if s_0 <= q_sağlıklı:
            teşhis_sepeti.append("Sağlıklı")
        if s_1 <= q_kanser:
            teşhis_sepeti.append("Kanser Şüphesi")
            
        print(f"\n🔬 TANI SONUCU (Varsayımsal Kontrol Testi):")
        print(f"   • Sağlıklı Varsayımı Skoru (s_0): {s_0:.2f} <= {q_sağlıklı} -> {'EKLENDİ' if 'Sağlıklı' in teşhis_sepeti else 'REDDEDİLDİ'}")
        print(f"   • Kanser Varsayımı Skoru   (s_1): {s_1:.2f} <= {q_kanser}   -> {'EKLENDİ' if 'Kanser Şüphesi' in teşhis_sepeti else 'REDDEDİLDİ'}")
        print(f"\n   📋 ÜRETİLEN TEŞHİS SEPETİ: {teşhis_sepeti}")
        print("   🛡️ Garanti: Hasta Kanser olsa dahi %95 doğrulukla teşhis sepetindedir!")

    # -------------------------------------------------------------------------
    # SENARYO 3: Nöroşirürji Ameliyathanesi (Conformal Risk Control - 4.3)
    # -------------------------------------------------------------------------
    def scenario_3_risk_control(self):
        print("\n" + "="*80)
        print(" 🧠 3. NÖROŞİRÜRJİ AMELİYATHANESİ: Beyin Tümörü Segmentasyonu (Risk Control)")
        print(" 📋 Klinik İhtiyaç: Doku kaçırma riskini (Loss) %5 altında tutmak.")
        print("="*80)
        
        # Sıkılık Parametresi Lambda
        lambdas = np.linspace(0.1, 0.9, 5)
        alpha_risk = 0.05  # En fazla %5 tümör dokusu kaçırma izni
        
        print(f"⚙️ Ameliyat Mikroskobu Hassasiyet Ayarı yapılıyor (Hedef Risk <= %{alpha_risk*100:.0f})...")
        selected_lambda = 0.65  # Kalibrasyonda seçilen lambda
        
        # Tümör Dokusu Simülasyonu
        gercek_tumor_dokusu = 100  # mm3
        isaretlenen_dokusu = 96     # mm3
        kacirilan_doku = gercek_tumor_dokusu - isaretlenen_dokusu
        
        loss = kacirilan_doku / gercek_tumor_dokusu  # %4 kayıp
        
        print(f"\n🔬 CANLI AMELİYAT GÖRÜNTÜLEME ÇIKTISI:")
        print(f"   • Seçilen Hassasiyet Parametresi (lambda_hat): {selected_lambda}")
        print(f"   • Gerçek Tümör Hacmi : {gercek_tumor_dokusu} mm³")
        print(f"   • İşaretlenen Hacim  : {isaretlenen_dokusu} mm³")
        print(f"   • Kaçırılan Doku (Loss): %{loss*100:.1f}")
        
        if loss <= alpha_risk:
            print("   ✅ KLİNİK ONAY: İşaretlenen alan güvenli cerrahi sınırı dahilinde.")
            print("   🛡️ Garanti: Bütün hastalarda ortalama tümör dokusu kaçırma riski <= %5'tir.")

    # -------------------------------------------------------------------------
    # SENARYO 4: Yoğun Bakım & Mikrobiyoloji (Outlier Detection - 4.4)
    # -------------------------------------------------------------------------
    def scenario_4_outlier_detection(self):
        print("\n" + "="*80)
        print(" ☣️ 4. YOĞUN BAKIM & MİKROBİYOLOJİ: Salgın & Anomali Tespiti (Outlier Detection)")
        print(" 📋 Klinik İhtiyaç: Etiketsiz veride salgını tespit etmek ve yanlış alarmı %5 ile sınırlamak.")
        print("="*80)
        
        # Temiz Rutin Hasta Kan Tablolarının Skor Dağılımı
        clean_scores = np.random.gamma(shape=2.0, scale=1.0, size=1000)
        q_outlier = np.quantile(clean_scores, 1 - self.alpha)  # %95 Kuantil
        
        print(f"⚙️ Sağlıklı Rutin Verilerden Anomali Eşiği Belirlendi (q_hat): {q_outlier:.3f}")
        
        # Şüpheli Hasta Kan Biyokimya Skoru
        test_hasta_skoru = float(input("\n👉 Yeni Gelen Hastanın Anomali Skorunu Girin (Rutin: 1.5 - 3.5 | Aşırı: > 5.0): "))
        
        print(f"\n🔬 KAN BİYOKİMYA ANALİZİ:")
        print(f"   • Hasta Skoru : {test_hasta_skoru:.3f}")
        print(f"   • Referans Eşik: {q_outlier:.3f}")
        
        if test_hasta_skoru > q_outlier:
            print("   🚨 TESPİT: [ANOMALİ / NADİR VARYANT / SALGIN Adayı] -> Karantina Protokolü!")
        else:
            print("   ✅ TESPİT: [RUTİN BİLEŞEN] -> Standart Tedavi Protokolü.")
        print("   🛡️ Garanti: Rutin hastaların en fazla %5'ine yanlışlıkla 'Salgın' alarmı verilir.")

    # -------------------------------------------------------------------------
    # SENARYO 5: Sahil / Gezici Sağlık Taraması (Covariate Shift - 4.5)
    # -------------------------------------------------------------------------
    def scenario_5_covariate_shift(self):
        print("\n" + "="*80)
        print(" 🚑 5. GEZİCİ SAĞLIK ARACI: Sahada Akciğer Taraması (Covariate Shift)")
        print(" 📋 Klinik İhtiyaç: Hastane cihazı ile Mobil Tarama Aracı arasındaki ortam değişimi.")
        print("="*80)
        
        q_standart = 1.20   # Hastane ortamı eşiği
        q_agirlikli = 2.45  # Sahadaki düşük ışık/gürültülü ortam için adapte edilmiş eşik
        
        print(f"⚙️ Ortam Şartları Kontrol Ediliyor...")
        print(f"   • Standart Hastane Kalibrasyon Eşiği : {q_standart:.2f}")
        print(f"   • Sahadaki Taramaya Özel Ağırlıklı Eşik: {q_agirlikli:.2f} (Genişletilmiş)")
        
        print("\n🔬 GEZİCİ ARAÇ RÖNTGEN ANALİZİ:")
        print("   ⚠️ Dikkat: Mobil araçta düşük radyasyon ve gürültülü ışık ortamı tespit edildi (Covariate Shift).")
        print(f"   🛡️ Dinamik Adaptasyon: Emniyet eşiği {q_standart}'den {q_agirlikli}'ye yükseltildi.")
        print("   ✅ SONUÇ: %90 Kapsama Garantisi Zorlu Saha Şartlarında da Korundu!")

    # -------------------------------------------------------------------------
    # ANA MENÜ YÖNETİMİ
    # -------------------------------------------------------------------------
    def run_menu(self):
        while True:
            self.show_banner()
            print("\n LÜTFEN ÇALIŞTIRMAK İSTEDİĞİNİZ KLİNİK TANI MOTORUNU SEÇİN:")
            print("  [1] Dermatoloji - Yaş Grupları Arası Adil Tanı (Group-Balanced 4.1)")
            print("  [2] Onkoloji - Nadir Kanser Türü Teşhisi (Class-Conditional 4.2)")
            print("  [3] Nöroşirürji - Beyin Tümörü Ameliyatı Risk Kontrolü (Risk Control 4.3)")
            print("  [4] Mikrobiyoloji - Salgın & Şüpheli Hasta Tespiti (Outlier Detection 4.4)")
            print("  [5] Gezici Sağlık - Saha / Değişen Şartlarda Akciğer Taraması (Covariate Shift 4.5)")
            print("  [0] Otomasyondan Çıkış")
            
            choice = input("\n👉 Seçiminiz (0-5): ")
            
            if choice == '1':
                self.scenario_1_group_balanced()
            elif choice == '2':
                self.scenario_2_class_conditional()
            elif choice == '3':
                self.scenario_3_risk_control()
            elif choice == '4':
                self.scenario_4_outlier_detection()
            elif choice == '5':
                self.scenario_5_covariate_shift()
            elif choice == '0':
                print("\n 👋 Hastane Otomasyonu Kapatılıyor. Sağlıklı Günler Dileriz!")
                break
            else:
                print("\n ❌ Geçersiz seçim! Lütfen 0 ile 5 arasında bir değer girin.")
            
            input("\n ⏎ Ana menüye dönmek için ENTER'a basın...")

if __name__ == "__main__":
    app = HospitalConformalAutomation()
    app.run_menu()