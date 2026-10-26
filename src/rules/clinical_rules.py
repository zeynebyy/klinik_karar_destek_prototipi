class ClinicalRuleEngine:
    @staticmethod
    def calculate_news2(hr, rr, temp, sbp, spo2, o2_flow):
        score = 0
        
        # Respiration Rate
        if rr <= 8: score += 3
        elif 9 <= rr <= 11: score += 1
        elif 12 <= rr <= 20: score += 0
        elif 21 <= rr <= 24: score += 2
        elif rr >= 25: score += 3
            
        # SpO2 (Scale 1)
        if spo2 <= 91: score += 3
        elif 92 <= spo2 <= 93: score += 2
        elif 94 <= spo2 <= 95: score += 1
        elif spo2 >= 96: score += 0
            
        # O2 Supplement
        if o2_flow > 0: score += 2
            
        # Systolic BP
        if sbp <= 90: score += 3
        elif 91 <= sbp <= 100: score += 2
        elif 101 <= sbp <= 110: score += 1
        elif 111 <= sbp <= 219: score += 0
        elif sbp >= 220: score += 3
            
        # Heart Rate
        if hr <= 40: score += 3
        elif 41 <= hr <= 50: score += 1
        elif 51 <= hr <= 90: score += 0
        elif 91 <= hr <= 110: score += 1
        elif 111 <= hr <= 130: score += 2
        elif hr >= 131: score += 3
            
        # Temp
        if temp <= 35.0: score += 3
        elif 35.1 <= temp <= 36.0: score += 1
        elif 36.1 <= temp <= 38.0: score += 0
        elif 38.1 <= temp <= 39.0: score += 1
        elif temp >= 39.1: score += 2
            
        return score

    @staticmethod
    def calculate_qsofa(rr, sbp):
        # We don't have Glasgow Coma Scale, so we approximate with mobility/age as a proxy, 
        # but traditionally it's just RR >= 22, SBP <= 100, altered mentation.
        score = 0
        if rr >= 22: score += 1
        if sbp <= 100: score += 1
        # Mentation is missing, max score here will be 2
        return score

    @staticmethod
    def calculate_sirs(temp, hr, rr):
        score = 0
        if temp > 38.0 or temp < 36.0: score += 1
        if hr > 90: score += 1
        if rr > 20: score += 1
        # WBC count is missing, max score here is 3
        return score

    @staticmethod
    def analyze_patterns(hr, rr, temp, sbp, spo2):
        patterns = []
        # Erken Sepsis Pattern: Temp > 38, HR > 100, RR > 22
        if temp > 38.0 and hr > 100 and rr > 22:
            patterns.append("Erken Sepsis Şüphesi")
            
        # Hipoksi Pattern: SpO2 < 92, RR > 24
        if spo2 < 92 and rr > 24:
            patterns.append("Akut Hipoksi Belirtisi")
            
        # Şok Pattern: SBP < 90, HR > 120
        if sbp < 90 and hr > 120:
            patterns.append("Şok Belirtisi (Kardiyojenik/Septik/Hipovolemik)")
            
        return patterns

    @staticmethod
    def evaluate_severity(prob, hr, rr, temp, sbp, spo2, threshold=0.5):
        """
        0: Stabil (Yeşil)
        1: Hafif Risk (Sarı)
        2: Yüksek Risk / Yakın Takip (Turuncu)
        3: Kritik Alarm (Kırmızı)
        """
        # Hard Clinical Rules for Critical (3)
        if sbp < 90 or spo2 < 90 or rr > 30 or hr > 130 or temp > 39.5:
            return 3, "Kritik", "Kritik Vital Eşik İhlali (Hard Rule)"
            
        # False Positive Reduction / Mild Anomalies -> max 1 (Hafif Risk) or 2
        # If probability is very high but no hard rule is broken, cap at 2
        
        # Calculate base severity from ML prob
        if prob < threshold * 0.5:
            base_sev = 0 # Stabil
        elif prob < threshold:
            base_sev = 1 # Hafif Risk
        elif prob < threshold * 1.5:
            base_sev = 2 # Yüksek Risk
        else:
            base_sev = 3 # Kritik (but capped later if no hard rules)

        # Cap at 2 if no hard rules are met
        if base_sev == 3:
            base_sev = 2
            reason = "Yüksek ML Skoru (Yakın Takip)"
        else:
            reason = "Stabil" if base_sev == 0 else ("Hafif Riskli Vital" if base_sev == 1 else "Yakın Takip Önerilir")
            
        # Extra check for specific mild anomalies to force to 1
        if base_sev > 1:
            if (100 <= hr <= 105) and (38.0 <= temp <= 38.3) and (20 <= rr <= 22):
                base_sev = 1
                reason = "Hafif Anomaliler Filtrelendi (Sıradan Enfeksiyon Bulgusu)"
                
        labels = {0: "Stabil", 1: "Hafif Risk", 2: "Yüksek Risk", 3: "Kritik"}
        return base_sev, labels[base_sev], reason
