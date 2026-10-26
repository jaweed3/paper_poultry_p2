# ICAIDES 2026 — Video Script + Tanya Jawab

Paper 1571359798. Maks 10 menit, MP4 720p 16:9.
Deck: `deck-icaides/slides.pdf` (11 frame + 4 divider = 15 halaman).

## Naskah video (~8 menit, sisakan buffer)

**0:00 — Judul (frame 1)**
"Halo, saya Fatih dari Universitas Darussalam Gontor. Paper kami: Localizing
INT8 Quantization Collapse in EfficientNet-B0 — deteksi penyakit ayam dari
citra feses, jalan di Raspberry Pi 5."

**0:30 — Gap (frame 2)**
"Penyakit kotoran ayam — Coccidiosis, Salmonella, Newcastle — bikin rugi besar,
tapi lab nggak terjangkau peternak kecil. Riset sebelumnya 99 persen akurat,
tapi jalan di cloud. Pertanyaan kami: model ringan mana yang selamat di-INT8-kan,
dan berapa tradeoff latency-nya di Pi 5?"

**1:20 — Dataset audit (frame 3)**
"Benchmark publiknya klaim 8.770 gambar leak-free. Audit pHash kami nemu 617
pasangan near-duplikat, 94 gambar test byte-identik dengan train. Jadi semua
angka di paper pakai split steril 8.153 gambar, test 1.728."

**2:20 — Collapse (frame 4-5)**
"Dynamic INT8 nyaris nggak nyentuh MobileNetV2 dan ShuffleNetV2 — turun 0,64
dan 0,06 poin. Tapi EfficientNet-B0 ambruk: 98,55 ke 10,42 persen. Uji McNemar
berpasangan konfirmasi. Confusion matrix-nya nunjukin model jadi penebak satu
kelas: 1.676 dari 1.728 prediksi jatuh ke kolom Newcastle."

**3:40 — Recovery (frame 6)**
"Static per-tensor juga ambruk, 9,32 persen. Tapi static per-channel pulih ke
94,56 persen — McNemar lawan dynamic: b 14, c 1468, p kurang dari 1e-15.
Replikasi di model retrain dengan kalibrasi seimbang 25 per kelas: per-tensor
27,14, per-channel 93,40. Jadi ini recovery, bukan parity — lawan FP32 masih
beda signifikan."

**4:50 — Mekanisme (frame 7)**
"Ablasi selektif nunjukin lokasinya: depthwise per-channel sendirian pulih ke
93,46 persen; semua per-channel KECUALI depthwise ambruk ke 8,33. Depthwise
dibiarkan FP32 tembus 97,16. Mekanismenya: satu skala per-tensor diatur channel
terbesar, channel kecil kehilangan resolusi — rentang depthwise melebar sampai
112 kali, channel terburuk terkuantisasi jadi konstanta."

**6:00 — Pi latency (frame 8-9)**
"Di Pi 5, dynamic INT8 malah LEBIH LAMBAT dari FP32: 1,93 sampai 2,8 kali.
FP32 scaling 2,2x dari 1 ke 4 thread; INT8 cuma 1,1x. Bukti kernel: tiap Conv
mengembang jadi DynamicQuantizeLinear + ConvInteger + Mul + Cast — node naik
1,37 sampai 3,05 kali. Tapi static per-channel membalik: 13,50 ms lawan 31,53 ms
FP32 — 2,34 kali lebih cepat."

**7:20 — Rekomendasi + tutup (frame 10-11)**
"Jadi mau ship yang mana? SNv2 FP32 tercepat 6,19 ms; MNv2 FP32 seimbang;
EB0 static per-channel paling akurat di 94,56. Selisih FP32 terbaik-terburuk
cuma 0,86 poin — biar hardware dan skema yang mutusin. Limitasi kami jujur:
satu seed, satu set kalibrasi single-class, energi belum diukur. Artifact —
kode, manifest, log benchmark — di GitHub dan Zenodo. Terima kasih."

## Tanya jawab — persiapan

### Sudah pasti ditanya

**Q: Kenapa EB0 ambruk tapi MNv2/SNv2 selamat?**
A: Depthwise separable di EB0 punya rentang channel ekstrem (112x, median
39,94x). Skala per-tensor tunggal diatur channel terbesar → channel kecil
hancur. MNv2 juga depthwise tapi rentangnya lebih jinak + SE block EB0
(32 layer reduce/expand) nambah sensitivitas. Ablasi C vs D yang membuktikan:
93,46 vs 8,33 — cuma granularity depthwise yang dibalik.

**Q: Kalibrasi 100 gambar single-class — bukannya itu yang bikin ambruk?**
A: Sebagian. Replikasi balanced 25/kelas di model retrain: per-tensor naik
17,82 poin (9,32 → 27,14) tapi TETAP collapse; per-channel stabil 93,40 vs
90,74 single-class. Jadi skew kalibrasi menjelaskan sebagian, bukan seluruh —
granularity tetap penentu.

**Q: Klaim "INT8 lebih lambat" — bukannya INT8 harusnya lebih cepat?**
A: Betul untuk static. Yang lambat itu DYNAMIC di runtime ini: quantize/dequantize
di runtime tiap Conv lebih mahal dari hematnya matmul INT8. Buktinya static
per-channel di board dan runtime yang sama 2,34x LEBIH CEPAT. Jadi klaimnya
scheme-specific, bukan "INT8 lambat" generik.

**Q: Satu board, satu seed — generalisasinya apa?**
A: Nggak ada klaim generalisasi. Paper-nya case study: tiga arsitektur × satu
pipeline × satu runtime. Yang digeneralisasi: METODENYA (audit dedup + ablasi
selektif + atribusi kernel), bukan angkanya.

**Q: Kenapa nggak QAT / kalibrasi lebih besar / multi-seed?**
A: Scope PTQ diagnostic. QAT dan sweep kalibrasi = future work, tertulis di
conclusion. Kalau reviewer nekan: setuju itu langkah berikut, tapi temuan
granularity tetap berdiri karena efeknya 85 poin, bukan noise seed.

**Q: NCD cuma 141 test — class imbalance nggak ditangani?**
A: Betul, diakui sebagai open work di conclusion. Macro-F1 dan balanced acc
dilaporkan per model. EB0 per-channel per-class: cocci 96,03 / healthy 94,84 /
NCD 92,91 / salmo 93,32 — NCD terendah, konsisten dengan underrepresentasi.

**Q: Energi?**
A: Nggak diukur — telemetry current null di board ini. Sudah stated sebagai
limitation, bukan disembunyikan. Latency model-only batch 1, bukan end-to-end.

### Kalau diserang

**"9,32% kebetulan sama antara model utama dan retrain — data dicolok?"**
→ Justru didokumentasikan sebagai coincidence di caption Tabel eb0static +
Poin 1/2 box. Dua model beda (FP32 98,50 vs 98,15), dua set kalibrasi beda,
hasil per-tensor sama persis = kebetulan yang kami disclose, bukan disamarkan.
Raw JSON ada di `results/balanced_replication/poin1_poin2_box.json`.

**"Dedup 617 tapi klaim publisher leak-free diserang — buktinya?"**
→ `results/phash_dedup.json` (393 byte-exact groups, 303 cross-split) +
`results/sterile_split_manifest.json` (513 train + 78 val + 26 test dibuang =
617). Zero cross-split survivors terverifikasi.

## Checklist rekaman

- [ ] Ganti `figures/logo-trim.png` placeholder (kotak "YOUR LOGO HERE") dengan logo UNIDA/Gontor
- [ ] Ukur ulang `titlegraphic height` + `date makebox` kalau logo diganti (lihat README template)
- [ ] Rekam 720p, MP4, maks 150MB, maks 10 menit — naskah di atas ~8 menit
- [ ] Suara bersih, ruangan sepi, format horizontal
- [ ] Tetap hadir di sesi — video tidak menggantikan kehadiran (no-show = gagal Xplore)
