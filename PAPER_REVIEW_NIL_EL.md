# NIL-EL makalesinin RMALA'ya uygunluğu

Güncelleme: kullanıcı ilk uyarlamayı onayladı. V20 aynı özelliklerle
eşleşme/fayda hedefi pilotu olarak 1566598 numarasıyla çalışıyor.
İnceleme anındaki karar notları aşağıda korunmuştur; tam tür destekli
CLINK uygulaması yapıldığı anlamına gelmez.

İnceleme kararı: sorun ailesi doğrudan ilgili; yöntem kontrollü bir uyarlama
deneyine değer. Bizim sıfır grup zararı koşulumuzu çözdüğüne dair kanıt yok.
Kullanıcının istediği inceleme ve birlikte karar aşaması için bu öneri
hazırlandı; bu belge yeni bir deney sonucu değildir.

## Makaleden doğrulananlar

[Zhu ve diğerleri, ACL Findings 2023](https://aclanthology.org/2023.findings-acl.690.pdf)
bilgi tabanında uygun karşılığı olmayan ifadeleri NIL olarak reddetmeyi
inceliyor. Missing Entity ve Non-Entity Phrase ayrılıyor. Doğru adayın
maskelenmesiyle ek NIL örnekleri oluşturuluyor. Aday çiftlerinde ikili
eşleşme kaybına yardımcı tür tahmini ve focal loss ekleniyor. Anlamsal
skor ile tahmin edilen türlerin cosine benzerliği birleştirilip eşikleniyor.
Bu, zorunlu olarak öğrenilmiş ayrı bir NIL vektörü eklemek değildir.
NEL üzerinde CLINK-cross genel doğruluğu %87.71, NIL doğruluğu %89.19;
hata tamamen kaldırılmıyor. Tür sistemi ve aktarılabilirlik sınırlamaları
da belirtiliyor. İlgili yerler: §3.2, §4, Tablo 3, §5.2, sınırlamalar.
[Resmî depo](https://github.com/solitaryzero/NIL_EL) tip tahminli ve tipsiz
bi-/cross-encoder karşılaştırmalarını ayrı modeller olarak sunuyor.

## Mevcut kodla karşılaştırma

V16'da incelenen 10 korunan-grup zararlı kabulün tamamı yanlış kimliğe,
bankada olmayan sorguya ait. Bu, uygun karşılık yokken en benzer kaydın
seçilmesiyle oluşan hata ailesidir. Ancak bizim sentetik yeni kimliklerimiz
otomatik olarak doğal dildeki Non-Entity Phrase kategorisi sayılamaz.

Bizde reddetme zaten var. `rmala/utility_gate.py:35` içindeki hedef,
adayın MSE'yi azaltıp azaltmadığıdır; zarar/fayda büyüklüğü ile ağırlıklanır.
Doğru kimliğin gerçekten aday bankada olup olmadığı ayrı bir öğrenme
hedefi değildir. Sekiz özet özellik tam sorgu ve aday bağlamının yerini
tutmuyor. Bu yüzden yalnız kapıya NIL adını vermek yeni çözüm olmaz.

`rmala/experiments_v5.py:15` tür ontolojisi yerine her bağlamda rastgele
üretilen 32 ortak namespace kullanır. `projected()` geçerli sorguya kaynak
namespace'ini, yeni kimlikli sorguya rastgele namespace verir. Aynı kategori
çakışabilir; bu kodda kategoriler film/kişi gibi anlamlı türler değildir.

`rmala/experiments_v3.py:23` yeni kimliğin anahtarını mevcut anahtara .015
gürültü ekleyerek oluşturur; geçerli gürültülü sorguda katsayı .03'tür.
Yeni kimliğin değeri bağımsız ya da sıradan ortak değere yakındır. Dolayısıyla
anahtarın çok yakın olması aynı kimliği kanıtlamaz. Bazı örneklerde mevcut
gözlenebilir girdilerin ayırt ediciliği belirsizdir; bu bir imkânsızlık
ispatı değildir ve yeni bir deneyle incelenmelidir.

`stored`, `rare`, gerçek hedef veya gerçek sorgu kimliğini çıkarımda tür
özelliği olarak vermek cevabı sızdırır. Eğitim etiketi olarak kullanılabilir;
çıkarımda tahmin edilmesi veya gerçek girdiden elde edilmesi gerekir.
Veriyi sonradan kolaylaştırıp eski görev çözülmüş gibi raporlamak da doğru olmaz.

Bizim ölçütümüz her test bağlamında sıradan/eksik sorguların temiz/gürültülü
alt gruplarında GLA'ya göre zarar olmamasıdır (`experiments_v4.py:33`).
Eşleştirme doğruluğu ile bu koşul aynı metrik değildir. Eşleşme doğru olsa
bile bellek düzeltmesi GLA'dan kötü olabilir; mevcut fayda kapısı korunmalıdır.

## Önerilen somut uyarlama

1. Ayrı aday eşleşme skoru: eğitim etiketi, seçilen adayın doğru kaynak
   kimliği olmasıdır. Bankada karşılık yoksa bütün adaylar negatiftir.
   NIL kararı GLA çıktısını korur. Mevcut fayda kapısı bu aşamadan sonra kalır.
2. Eğitimde doğru adayın kontrollü çıkarılmasıyla eksik-kayıt örnekleri
   oluşturulur. Adaylar, sıralama, bankada-bulunma etiketleri ve maliyetler
   maskelenmiş banka için yeniden hesaplanır; gerçek test verisi değiştirilmez.
3. Önce aynı gözlenebilir özelliklerle mevcut fayda hedefi ve açık eşleşme
   hedefi karşılaştırılır. Sonra daha zengin sorgu–aday özelliklerinin etkisi
   ayrı kol olarak ölçülür. Bu ilk pilot, tam CLINK yeniden üretimi değildir.
4. Tür yardımcı görevi ancak girdiden anlamlı biçimde tahmin edilebilen
   bağımsız tür etiketleri tanımlandığında eklenir. Mevcut rastgele namespace
   sonuçlarını anlamsal tür başarısı diye adlandırmayacağız. Gerçek metin
   veya açıkça ayrı adlandırılmış tür kontrollü sentetik görev gerekebilir.
5. Yeni training/validation/holdout ayrımı, üç başlangıç, eski kontrol,
   eksik-kayıt yanlış kabul oranı, doğru kaydı kabul oranı, gürültülü MSE
   ve katı grup koruması birlikte raporlanır. Tür bilgisi olmadan ve tür
   bilgisiyle sonuçlar ayrılır. Banka ve ek parametre/hesap maliyeti sayılır.

Birinci karar: açık eşleşme/NIL hedefini mevcut maliyet sınırları içinde
küçük pilotla sınamak. Tam tip destekli çözümü aynen uyguladığımızı veya
100M geçiş koşulunu sağladığımızı bu pilot başlamadan söyleyemeyiz.
Makale temelli yeni eğitim bu inceleme sırasında başlatılmadı.
