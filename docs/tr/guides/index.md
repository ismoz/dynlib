# Rehberler

Dynlib’in rehberleri, hızlı başlangıç materyalinin daha derinine iner: her bölüm temel bir alt sistemi “açıp” netleştirir. Böylece modelleri, steppers’ı, analiz yardımcılarını veya çizim (plot) katmanını; dağınık notlar arasında kaybolmadan özelleştirebilirsiniz.

## Rehberlere hızlı bakış

- [Komut satırı rehberi](cli/cli.md) — model doğrulama, stepper inceleme ve cache yönetimi için `dynlib` (ve `python -m dynlib.cli`) komutlarını açıklar.
- [Modelleme rehberi](modeling/index.md) — TOML DSL’i, aux yardımcılarını, mods, presets ve spesifikasyonları okunur/yeniden kullanılabilir tutan yazım kolaylıklarını kapsar.
- [Simülasyon rehberi](simulation/index.md) — steppers, wrappers, snapshots, results ve solver davranışını kontrol eden yapılandırma ayarları gibi runtime kavramlarını tarar.
- [Çizim rehberi](plotting/index.md) — dinamik sistemler için ayarlanmış Matplotlib tabanlı yardımcıları (`plot.series`, `plot.phase`, `plot.manifold`, decorations, exports, themes) özetler.
- [Analiz rehberi](analysis/index.md) — `Sim` sonuçlarını bilimsel içgörülere dönüştüren runtime observers, sweep yardımcıları, basins, Lyapunov tanıları ve manifold bulucuları açıklar.
