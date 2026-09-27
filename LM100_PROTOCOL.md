# 100M / ortak 3 milyar token karşılaştırması

Kullanıcı beş yaklaşık 100M modelin eğitilmesini ve model başına **birebir aynı
3.000.000.000 hedef token** kullanılmasını istedi. Önceki sentetik deneylerin
kalite geçiş şartı bu araştırma eğitimi için kullanıcı tarafından aşılmıştır;
V14'ün genel zararsızlık şartını geçtiği veya üretime hazır olduğu iddia edilmez.
Listede iki kez geçen belleksiz model tek referans olarak ele alınır.

## Kollar

1. Normalize pozitif özellikli GLA, belleksiz.
2. Aynı GLA + **V14-LM**, kabul edilen belleğe tam katkı.
3. Aynı GLA + **V14-LM**, kabul edilen belleğe sabit 0.5 katkı.
4. RoPE ve causal SDPA kullanan full attention.
5. HoLA: resmî HOLA deposunun `7135e2291f18de17c10b59206641bd093efaac78`
   sürümündeki GatedDeltaNet + betae cache, RMS cache normalizasyonu,
   64 kayıt, 256 chunk, başlangıç gate=-4, sabit tau=1.

HoLA'nın Gated DeltaNet omurgası GLA ile aynı değildir. Sonuç bu beş bütün
mimarinin karşılaştırmasıdır; HoLA farkının yalnız cache'den kaynaklandığını
gösteren bir ablasyon değildir. HoLA 100M ayarı makalenin yayımlanmış bir
ölçek ayarı değil, resmî yöntemin yaklaşık 100M boyutuna ölçeklenmesidir.

Ortak boyut 640, 16 katman, 32K bağlı embedding/output, RMSNorm, SwiGLU,
dropout=0. GLA ve full 10x64 attention head; HoLA 5x128. GLA/HoLA MLP=1536,
full MLP=1728 ile gerçek parametre sayıları yakın tutulur ve ayrıca yazılır.
Üç GLA kolunun başlangıç omurga ağırlıkları bit düzeyinde aynı olmalıdır.

## V14'ten gerçek dil modeline yapılan açık uyarlama

Eski V14 16 boyutlu sentetik, önce doldurulup dondurulan banka üzerinde
çalışıyordu. Sentetik namespace kodlarını gerçek metinde kullanmak mümkün
değildir. Bu nedenle dil modeli kolu açıkça **V14-LM** olarak adlandırılır:

- Kayıt anahtarı bağlamsal öğrenilmiş q/k'nin ilk 16 kanalı; değer head'in
  64 kanalıdır. Sentetik 32→16 namespace projeksiyonu taşınmaz.
- Önceki 64 gözlemin eşik istatistiği, taban tau=0.3 ve her prefix'te %5
  yazma/denenen okuma sınırı korunur. Gelecekten quantile hesaplanmaz.
- Ham int8 değer, fp32 ölçek, fp32 anahtar, 15 bit sketch kullanılır.
  Kayıt başına 158 bayt hesabıyla head başına 12 kayıt/1896 bayt;
  2048 bayt tavanı korunur. GLA state'i bu banka tavanına dahil değildir.
- Her token önce yazma fırsatını sonra okuma fırsatını kullanır. Güncel
  girdinin yazılması sonraki-token tahmininde nedenseldir. UtilityRouter
  okuma sırasında LRU zamanını güncellemediğinden eski kayıt önce çıkar.
- Hamming<=1 ön kontrolü, cosine top-1, aynı 8 fayda özelliği, 8→16→1 kapı,
  5 ondalıklı olasılık ve 0.99 kabul eşiği korunur.
- Kapı V14 raw_int8/41001/1000 ağırlığı ve normalizasyonuyla başlatılır.
  Dil modeli kaybıyla öğrenmesi için eğitimde sigmoid straight-through
  gradyanı kullanılır; ileri hesap her zaman sert kabul/ret yapar. Bu,
  eski donmuş V14 veya eski MSE fayda eğitimiyle birebir aynı eğitim değildir.
- Gerçek int8 değerlerin ve ayrık retrieval'ın gradyanı yoktur. Gizli durum
  q/k ve temel attention yolu üzerinden; kapı vekil gradyan üzerinden öğrenir.
- Vectorize uygulama sketch ön kontrolünü tüm tokenlarda hesaplayabilir.
  Tam cosine retrieval yalnız bütçedeki sorgularda yapılır. Ek sketch hesabı
  ayrıca sayılır; %5 retrieval bütün model FLOP'unun %95 azaldığı anlamına gelmez.
- Training sırasında yazılan değerlerin paralel hesap için tutulduğu geçici
  cache ve aktivasyonlar ayrıca raporlanır; 1896 bayt toplam GPU belleği değildir.

## Ortak veri ve bütçe

Kök: `/arf/scratch/YOURUSER/mercanset_v11_nedo32k_pretokenized_20260902`.
Burada iki ayrı mühürlü koleksiyon birlikte bulunur: MercanSet V11
284.886.070.187 ve MercanPretraining 15.304.461.445 token; toplam
300.190.531.632 token, 5307 shard. Sayısal part kimlikleri tekil değildir;
tam dosya adları, koleksiyon manifestleri ve tokenizer hash'i kullanılır.

`lm100/data_3b_v2.lock.json` bütün kollarda ortaktır. Sabit seed=20260926 ile
shard sırası karıştırılır; ilk 32 validation, sonraki 32 test için ayrılır.
Kalan shard'lardan tek sıralı 3B hedef + 1 başlangıç bağlam tokenı okunur.
Eğitimde belge sonlarında EOS korunur, blok içinde belgeler paketlenir.
2048'lik her örnekte bağlam/banka sıfırlanır; kayıp bütün hedefleri bir kez
görür. Son adımda fazla konumlar -100 maskesi alır. Veri tekrarlanmaz.
Modellere bağlı örnekleme veya farklı karıştırma yoktur. Her güncellemede
ortak hedef byte dizisinin hash zinciri ilerler; resume aynı cursor'u korur.

Validation/test tam belgeler içerir; her shard'da seed ile seçilen belgeler
hedef kotayı doldurur. Uzun belgeler tamamlandığından yaklaşık 10M bütçe
validation'da 15.507.280, testte 11.352.596 token olmuştur. Split'ler fiziksel
shard ve belge düzeyinde ayrıdır; koleksiyonlar arası metin kopyaları için
ayrıca dedup yapılmadığından mutlak içerik sızıntısızlığı iddia edilmez.

Başlangıç eğitim ayarı: AdamW (0.9,0.95), weight decay=0.1, gradient clip=1,
BF16, ortak learning rate=3e-4, ilk %1 token warmup, ardından cosine ile
3e-5'e iniş. Güncelleme başına 131072 hedef token; microbatch beş modelde de
8x2048 token, 8 gradyan biriktirme adımıdır. Aynı optimizasyon bütçesi compute eşitliği
değildir; hız ve toplam hesap farkları ayrıca ölçülecek.

## Ölçüm ve başlatma durumu

PPL = exp(toplam token NLL / hedef token sayısı), EOS dahil. BPB = tam
belgelerin EOS hariç içerik NLL toplamı / (ln(2) × orijinal UTF-8 baytları).
Eğitimde BOS bulunmadığından belgenin ilk tokenı EOS sınır girdisi üzerinden
tahmin edilir. Bu düzeltme v2 veri kilidinde kayıtlıdır; eğitim token dizisi
ilk kilitle birebir aynıdır. EOS dahil BPB de ayrıca raporlanır.
Uzun belgelerde 2048 hedeflik
bloklar ve bir önceki token girdisi kullanılır; hiçbir hedef düşürülmez.
Test sonuçları eşik/hyperparameter seçimine geri beslenmez.

Eğitim sırasında token/s, güncelleme süresi, GPU bellek zirvesi, kayıp,
öğrenme oranı, bellek yazma/okuma/kabul ve FLOP kapsamı kaydedilecek.
Profiler'ın görebildiği matmul FLOP'ları ile özel Triton kernel'lerinin
algoritmik hesap tahminleri ayrı tutulmalı; eksik profiler toplamı tam
donanım FLOP ölçümü olarak adlandırılmamalıdır. %5 bellek bütçesinden toplam
FLOP tasarrufu sonucu çıkarılmaz. Nihai raporda PPL/BPB ve compute birlikte
karşılaştırılır.


Model weights and scientific results are linked from README.md.
