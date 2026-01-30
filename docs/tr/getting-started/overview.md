# Genel Bakış

Eğer dynlib ile yeni tanışıyorsanız, temel bilgiler için [ana sayfa](../index.md)'yı incelemeyi ihmal etmeyin.

## Gereksinimler

- Çalışan bir Python ortamı (3.10+ önerilir). 
- Zorunlu olmasa da kurulum için bir sanal ortam (virtualenv) veya benzeri bir izolasyon katmanı.
- Numerik hesaplamalar için Numpy paketi (`python -m pip install numpy`).
- Çizim için Matplotlib paketi (`python -m pip install matplotlib`).
- Paketin kendisi (`python -m pip install dynlib`). Kurulum detayları için [Hızlı Başlangıç](quickstart.md) rehberine bakabilirsiniz.

!!! important "Yüksek performanslı simülasyon ve analiz yapabilmek için Numba paketi şiddetle tavsiye edilir (`python -m pip install numba`)."

## Bu bölüm nasıl kullanılır

1. dynlib'i kurmak, CLI sağlamasını yapmak (`dynlib --version`, `dynlib model validate` vb.) ve yerleşik modellerden birini Python üzerinden çalıştırmak için **[Hızlı Başlangıç](quickstart.md)** kılavuzunu takip edin.
2. Bir TOML modeli yazmak, onu doğrulamak (`dynlib model validate first-model.toml`) ve "inline" (satır içi) metinlerle denemeler yapmak için **[İlk Modeliniz](first-model.md)** bölümüne geçin.
3. Stepper'lar, kaydediciler (recorders) ve diğer DSL özellikleriyle çalışmanız gerektiğinde [Modelleme rehberi](../guides/modeling/index.md) ile [Simülasyon rehberi](../guides/simulation/index.md)'ni keşfedin. 
4. Simülasyon sonuçlarını kullanmak ve çizim yapmak için [Simülasyon Sonuçları](../guides/simulation/results.md) ve [Çizim](../guides/plotting/index.md) rehberlerine göz atın.
5. Çatallanma (bifurcation) diyagramlarından nöron modellerine kadar tam iş akışlarını görmek için [Örnekler](../examples/index.md) ve [Analiz rehberi](../guides/analysis/index.md) bölümlerine bakın.
