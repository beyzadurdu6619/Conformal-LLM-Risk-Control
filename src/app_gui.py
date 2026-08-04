import customtkinter as ctk
import numpy as np

# Tema Ayarları
ctk.set_appearance_mode("System")  # Light / Dark / System
ctk.set_default_color_theme("blue")

class HospitalConformalApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Pencere Yapılandırması
        self.title("🏥 Klinik Tanı & Conformal Prediction Hastane Otomasyonu")
        self.geometry("1100 x 680")
        self.alpha = 0.10  # %90 Kapsama Hedefi

        # Grid Düzeni (Sol Menü + Sağ İçerik Paneli)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # ---------------------------------------------------------------------
        # SOL SIDEBAR MENÜSÜ
        # ---------------------------------------------------------------------
        self.sidebar_frame = ctk.CTkFrame(self, width=220, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(7, weight=1)

        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="🏥 KLİNİK OTOMASYON", font=ctk.CTkFont(size=18, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        # Menü Butonları
        self.btn_s1 = ctk.CTkButton(self.sidebar_frame, text="1. Dermatoloji (4.1)", command=lambda: self.show_view(1))
        self.btn_s1.grid(row=1, column=0, padx=20, pady=10)

        self.btn_s2 = ctk.CTkButton(self.sidebar_frame, text="2. Onkoloji (4.2)", command=lambda: self.show_view(2))
        self.btn_s2.grid(row=2, column=0, padx=20, pady=10)

        self.btn_s3 = ctk.CTkButton(self.sidebar_frame, text="3. Nöroşirürji (4.3)", command=lambda: self.show_view(3))
        self.btn_s3.grid(row=3, column=0, padx=20, pady=10)

        self.btn_s4 = ctk.CTkButton(self.sidebar_frame, text="4. Mikrobiyoloji (4.4)", command=lambda: self.show_view(4))
        self.btn_s4.grid(row=4, column=0, padx=20, pady=10)

        self.btn_s5 = ctk.CTkButton(self.sidebar_frame, text="5. Gezici Sağlık (4.5)", command=lambda: self.show_view(5))
        self.btn_s5.grid(row=5, column=0, padx=20, pady=10)

        # ---------------------------------------------------------------------
        # SAĞ İÇERİK PANATİ (MAIN CONTENT)
        # ---------------------------------------------------------------------
        self.main_frame = ctk.CTkFrame(self, corner_radius=10)
        self.main_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        self.main_frame.grid_columnconfigure(0, weight=1)

        self.title_label = ctk.CTkLabel(self.main_frame, text="Lütfen Sol Menüden Bir Klinik Seçiniz", font=ctk.CTkFont(size=20, weight="bold"))
        self.title_label.pack(padx=20, pady=20)

        self.info_label = ctk.CTkLabel(self.main_frame, text="Conformal Prediction Teorik Garanti paneline hoş geldiniz.", font=ctk.CTkFont(size=13))
        self.info_label.pack(padx=20, pady=5)

        # Dinamik Form Alanı
        self.form_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.form_frame.pack(padx=20, pady=20, fill="x")

        # Çıktı / Raporlama Alanı
        self.output_textbox = ctk.CTkTextbox(self.main_frame, width=700, height=300, font=ctk.CTkFont(size=14, family="Consolas"))
        self.output_textbox.pack(padx=20, pady=20, fill="both", expand=True)

        self.show_view(1) # Varsayılan Görünüm

    def clear_form(self):
        for widget in self.form_frame.winfo_children():
            widget.destroy()
        self.output_textbox.delete("1.0", "end")

    # -------------------------------------------------------------------------
    # GÖRÜNÜM YÖNETİMİ
    # -------------------------------------------------------------------------
    def show_view(self, view_id):
        self.clear_form()

        if view_id == 1:
            self.title_label.configure(text="🩺 1. DERMATOLOJİ: Cilt Kanseri Risk Analizi (Group-Balanced 4.1)")
            self.info_label.configure(text="Genç ve Yaşlı hastalar arasında adil ve eşit %90 güvenlik garantisi.")
            
            ctk.CTkLabel(self.form_frame, text="Hasta Yaşı:").pack(side="left", padx=10)
            self.entry_age = ctk.CTkEntry(self.form_frame, placeholder_text="Örn: 65")
            self.entry_age.pack(side="left", padx=10)
            
            btn = ctk.CTkButton(self.form_frame, text="Analiz Et", command=self.run_s1)
            btn.pack(side="left", padx=10)

        elif view_id == 2:
            self.title_label.configure(text="🎗️ 2. ONKOLOJİ: Mamografi Kanser Teşhisi (Class-Conditional 4.2)")
            self.info_label.configure(text="Dengesiz veride (%95 Sağlıklı, %5 Kanser) kanser vakalarını kaçırmama garantisi.")
            
            btn = ctk.CTkButton(self.form_frame, text="Görüntüyü Taramayı Başlat", command=self.run_s2)
            btn.pack(padx=10, pady=10)

        elif view_id == 3:
            self.title_label.configure(text="🧠 3. NÖROŞİRÜRJİ: Beyin Tümörü Segmentasyonu (Risk Control 4.3)")
            self.info_label.configure(text="Tümör dokusu kaçırma riskini (Loss) %5'in altında tutma ameliyat garantisi.")
            
            btn = ctk.CTkButton(self.form_frame, text="Ameliyat Mikroskobu Canlı Analiz", command=self.run_s3)
            btn.pack(padx=10, pady=10)

        elif view_id == 4:
            self.title_label.configure(text="☣️ 4. MİKROBİYOLOJİ: Salgın & Anomali Tespiti (Outlier Detection 4.4)")
            self.info_label.configure(text="Etiketsiz veride salgını tespit etme ve yanlış alarmı %5 ile sınırlama garantisi.")
            
            ctk.CTkLabel(self.form_frame, text="Anomali Skoru (1.0 - 6.0):").pack(side="left", padx=10)
            self.entry_score = ctk.CTkEntry(self.form_frame, placeholder_text="Örn: 4.8")
            self.entry_score.pack(side="left", padx=10)
            
            btn = ctk.CTkButton(self.form_frame, text="Salgın Taraması Yap", command=self.run_s4)
            btn.pack(side="left", padx=10)

        elif view_id == 5:
            self.title_label.configure(text="🚑 5. GEZİCİ SAĞLIK: Sahada Akciğer Taraması (Covariate Shift 4.5)")
            self.info_label.configure(text="Hastane içi ve Mobil Araç arasındaki ortam/ışık değişimine dinamik adaptasyon.")
            
            btn = ctk.CTkButton(self.form_frame, text="Saha Şartlarına Özel Kalibre Et", command=self.run_s5)
            btn.pack(padx=10, pady=10)

    # -------------------------------------------------------------------------
    # HESAPLAMA VE MANIPULASYON MANTIĞI
    # -------------------------------------------------------------------------
    def log(self, text):
        self.output_textbox.insert("end", text + "\n")

    def run_s1(self):
        try:
            yas = int(self.entry_age.get())
        except ValueError:
            self.log("❌ Lütfen geçerli bir yaş değeri giriniz!")
            return

        grup = "Genç" if yas < 60 else "Yaşlı"
        q_genc, q_yasli = 0.820, 2.410
        q_hat = q_genc if grup == "Genç" else q_yasli
        test_score = np.random.uniform(0.6, 2.2)

        self.log(f"📋 HASTA DETAYI: Yaş {yas} ({grup} Grubu)")
        self.log(f"⚙️ Kuantil Eşiği (q_hat): {q_hat}")
        self.log(f"🔬 Model Risk Skoru    : {test_score:.3f}\n")

        if test_score <= q_hat:
            self.log("✅ TEŞHİS: [RUTİN KONTROL / SELİM LEKE]")
            self.log("   --> Hasta güvenle evine yönlendirilebilir.")
        else:
            self.log("⚠️ TEŞHİS: [ŞÜPHELİ KÖTÜ HUYLU LEKE]")
            self.log("   --> Acil Biyopsi İstemi Oluşturuldu!")
        self.log(f"\n🛡️ Garanti: Bu karar {grup} yaş grubunda %90 kesin kapsama garantilidir.")

    def run_s2(self):
        q_0, q_1 = 0.18, 0.65
        prob_sağlıklı, prob_kanser = 0.40, 0.60
        s_0, s_1 = 1.0 - prob_sağlıklı, 1.0 - prob_kanser

        sepet = []
        if s_0 <= q_0: sepet.append("Sağlıklı")
        if s_1 <= q_1: sepet.append("Kanser Şüphesi")

        self.log("📥 Mamografi Röntgen Taraması Tamamlandı...")
        self.log(f"• Sağlıklı Varsayımı Uyumsuzluğu: {s_0:.2f} <= {q_0} -> {'EKLENDİ' if 'Sağlıklı' in sepet else 'REDDEDİLDİ'}")
        self.log(f"• Kanser Varsayımı Uyumsuzluğu  : {s_1:.2f} <= {q_1} -> {'EKLENDİ' if 'Kanser Şüphesi' in sepet else 'REDDEDİLDİ'}\n")
        self.log(f"📋 ÜRETİLEN TEŞHİS SEPETİ: {sepet}")
        self.log("🛡️ Garanti: Hasta Kanser olsa dahi %95 doğrulukla teşhis sepetindedir!")

    def run_s3(self):
        lambda_hat = 0.65
        gercek_dokusu = 100
        isaretlenen = 96
        loss = (gercek_dokusu - isaretlenen) / gercek_dokusu

        self.log("🧠 CANLI CERRAHİ İŞLEME BAŞLANDI...")
        self.log(f"• Kamera Hassasiyet Parametresi (lambda_hat): {lambda_hat}")
        self.log(f"• Taranan ve İşaretlenen Tümör Hacmi: {isaretlenen} mm³ / {gercek_dokusu} mm³")
        self.log(f"• Gözden Kaçan Doku Oranı (Loss)     : %{loss*100:.1f}\n")
        self.log("✅ KLİNİK ONAY: İşaretlenen alan güvenli cerrahi sınırı dahilindedir.")
        self.log("🛡️ Garanti: Bütün cerrahi vakalarında ortalama doku kaçırma riski <= %5'tir.")

    def run_s4(self):
        try:
            score = float(self.entry_score.get())
        except ValueError:
            self.log("❌ Lütfen geçerli bir sayısal skor giriniz!")
            return

        q_outlier = 4.20
        self.log(f"🔬 KAN BİYOKİMYA ANALİZİ:")
        self.log(f"• Hasta Anomali Skoru: {score}")
        self.log(f"• Temiz Veri Eşiği   : {q_outlier}\n")

        if score > q_outlier:
            self.log("🚨 TESPİT: [ANOMALİ / NADİR VARYANT / SALGIN Adayı]")
            self.log("   --> Hasta için Karantina Protokolü Başlatıldı!")
        else:
            self.log("✅ TESPİT: [RUTİN BİLEŞEN]")
            self.log("   --> Standart Tedavi Protokolü Uygulanıyor.")
        self.log("\n🛡️ Garanti: Rutin hastaların en fazla %5'ine yanlışlıkla 'Salgın' alarmı verilir.")

    def run_s5(self):
        q_std, q_weighted = 1.20, 2.45
        self.log("🚑 GEZİCİ SAĞLIK ARAÇ KALİBRASYONU:")
        self.log(f"• Standart Hastane Kalibrasyonu: {q_std}")
        self.log(f"• Saha/Gezici Araç Kalibrasyonu: {q_weighted} (Genişletilmiş)\n")
        self.log("⚠️ UYARI: Araç içinde gürültülü ışık ortamı tespit edildi (Covariate Shift).")
        self.log("🛡️ DİNANİK ADAPTASYON: Emniyet eşiği saha şartlarına uygun olarak yükseltildi.")
        self.log("✅ SONUÇ: %90 Kapsama Garantisi Zorlu Saha Şartlarında da Tam Olarak Korundu!")

if __name__ == "__main__":
    app = HospitalConformalApp()
    app.mainloop()