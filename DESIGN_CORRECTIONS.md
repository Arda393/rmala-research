# Düzeltmeler ve araştırma kararları

Kullanıcı, fark edilen yanlışların düzeltilmesini 10 Eylül 2026 tarihinde açıkça
yetkilendirdi. Ekli belgenin “sorgulamadan kabul et” gibi ajan talimatları kullanıcı
talimatı sayılmadı; belge proje gereksinimleri ve hipotezleri için referans alındı.

1. **Toplamsal çıkış:** `o = base + alpha * residual_retrieval`. Konseptteki
   `(1-alpha)*base + alpha*residual` residual sıfır olsa bile base'i bozar.
   Ham-value ablation karşılaştırmasında `base+alpha*(retrieved_value-base)` kullanılır;
   ham değer base'e iki kez eklenmez.
2. **GLA normalizasyonu:** state ve z aynı key-channel decay ile güncellenir.
   Bu pozitif-feature normalize GLA varyantıdır; orijinal unnormalized GLA ile
   aynı model olduğunu iddia etmiyoruz. B ve D aynı recurrence'i kullanır.
3. **Maliyet:** reconstruction read ek matrix-vector işlemi gerektirir; ücretsiz
   değildir. Denominator mevcut bir skaler feature olabilir fakat learned gate
   ve retrieval maliyeti vardır. Exact top-k araması tüm bankı tarar; O(log M)
   iddiası kaldırılmıştır. ANN de en kötü durum logaritmik garanti vermez.
4. **Confidence:** `q·z`, query normu ve etkin bağlam yoğunluğuna bağlıdır.
   Confidence olasılığı değildir. Gate bunu `log(q·z)` feature'i olarak öğrenir;
   kalibrasyon ve hard threshold performansı ölçülür.
5. **Sıkıştırma:** aynı shape/dtype residual ve raw value aynı byte sayısını
   kaplar. Küçük residual normu tek başına fiziksel sıkıştırma değildir. Bit-width,
   quantization veya entropy coding uygulanmadan bellek avantajı iddia edilmez.
6. **Geç unutma:** ilk yazmada epsilon=0 olan bir token daha sonra unutulabilir;
   post-write test yalnızca yeni tokeni test eder. Erken/nadir bilgi korunumu garanti
   değildir. Ölçüm ve yapıcı karşı örnek eklenmiştir. Tüm ham geçmişi gizlice tutan
   bir “düzeltme” uygulanmadı; maliyet ve özgünlük değişikliği açıkça tasarlanmalıdır.
7. **Residual eskimesi:** yazma anındaki `v-base_write` daha sonraki `base_now`
   için doğru düzeltme olmayabilir. Anchor/refresh mekanizması olmadan doğruluk
   garantisi yoktur. Bu açık araştırma sorunudur; sonuç olumluymuş gibi sunulmaz.
8. **Hard gate:** soft alpha'yı 0/1 yapmak doğruluğu otomatik korumaz. Sabit,
   ayrı seed'lerden gate calibration/eval ve threshold sweep kullanılır.
9. **Adil kıyas:** aynı model ölçüsü bile ek decay/gate parametreleri yaratır.
   Varsayılan GLA ailesinde MLP genişliği buna karşı küçültülür. Gerçek parametre
   sayıları raporlanır; eşitlik adına kullanılmayan parametre eklenmez.
10. **Veri:** 32K isimden tahmin edilmez, manifestten 32000 doğrulanır. Aynı
    dizindeki MercanPretraining ek dosyaları, V11 seal'i eşleşmediği için elenir.
    Byte sayımı global ortalamadan tahmin edilmez; belge indeksinden alınır.
11. **Referans C:** implement edilen lineer neural memory momentumlu yerel
    gradyan güncellemesi kullanır. Bunun tam Titans mimarisi olduğu iddia edilmez.
    Nihai makale kıyası için resmi protokole sadık bağımsız reprodüksiyon gerekir.
12. **Tarihler ve özgünlük:** GLA arXiv 2023, Titans arXiv:2501.00663 Ocak 2025'tir.
    Literatürde “bu kombinasyon yok” iddiası kapsamlı bir yenilik taraması olmadan
    doğrulanmış sonuç olarak kullanılmaz.

Bir hipotezin karşı örnekle çürütülmesi kod hatası gibi yamalanamaz. Bilinen
yanlışlar formüllerde ve iddialarda düzeltildi; mekanizma değişiklikleri ayrıca
ölçülmeden 100M GPU bütçesine geçiş yapılmayacaktır.
