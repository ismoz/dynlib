# Analiz rehberi

Dynlib temel analiz araçlarını sunar. Zamanla bu analiz araçları daha da genişletilecektir. Mevcut analiz araçları için dökümanlara aşağıdaki bağlantılardan ulaşabilirsiniz.

## Konular

- [Havzalar (Basins)](basin.md) — ızgara yapılandırması, algılama eşikleri ve çizim ipuçları ile otomatik veya bilinen çeker havzası hesaplayıcıları (`basin_auto`, `basin_known`).
- [Sabit noktalar (Fixed points)](fixed-points.md) —  `find_fixed_points` ve `FullModel.fixed_points` aracılığıyla sunulan Newton çözücüleri.
- [Çalışma zamanı gözlemcileri](observers.md) — gözlemci çerçevesi, adım öncesi/sonrası kancaları (hooks), iz (trace) tamponları ve doğrudan `Sim.run` içine takılan `lyapunov_*` gözlemci (observer) fabrikaları.
- [Lyapunov analizi](lyapunov.md) — Maksimum Lyapunov üsteli (MLE) ve Lyapunov spektrum hesabı.
- [Tarama araçları (Sweep utilities)](sweep.md) — değişen parametre değerleri için pratikleştirilmiş simülasyonlar ve analizler.
- [Çatallanma diyagramları](bifurcation.md) — post-analiz ile bifürkasyon (çatallanma) verisi elde etme.
- [Son analiz (Post-analysis)](post-analysis.md) — elde edilen simulasyon sonuçları üzerinde kullanabileceiniz çeşitli post-analiz araçları.
- [Manifold analizi](manifold.md) — 1B manifold izleme, heteroklinik/homoklinik yörünge arama ve izleme. 