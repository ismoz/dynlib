# Analiz Örnekleri

## Genel Bakış

`examples/analysis/` klasörü, dynlib'in analiz araçlarının nasıl kullanıldığını gösteren örnekleri içerir. Burada da bu örneklerden bazıları verilmektedir. Kullanılan analiz araçlarının detaylı kullanımı için [Analiz rehberine](../guides/analysis/index.md) bakınız. 

## Çekim havzası haritalama ve sınıflandırma

### Henon haritası havza keşfi

Yerleşik Henon haritası üzerinde 512×512'lik bir ızgarada, PCR-BM algoritması ile `basin_auto` analiz aracı ile Henon haritasının çekerleri ve bu çekerlerin çekim havzaları numerik olarak hesaplanmaktadır. `basin_auto` çeker yapılarını otomatik olarak bulmaya çalışır.

```python
--8<-- "examples/analysis/basin_henon_auto.py"
```

### Bilinen çekerlerin havzası

Bu örnekte ise yine 512×512'lik bir ızgarada yerleşik Henon haritası modeli için `basin_known` analiz aracı kullanımı gösterilmektedir. Bu aracın kullanılması için çeker yapıları önceden `ReferenceRun` veya `FixedPoint` ile tanımlanmalıdır.

```python
--8<-- "examples/analysis/basin_henon_known.py"
```

### Limit döngü (limit-cycle) havzaları

Bu örnekte `basin_auto` kullanarak bir limit döngüye ait çekim havzası hesaplanmaktadır. `basin_auto` limit döngüyü otomatik olarak tespit etmektedir.

```python
--8<-- "examples/analysis/basin_limit_cycle_auto.py"
```

Bu örnekte ise yine aynı limit döngü bu sefer `ReferenceRun` ile tanımlandıktan sonra `basin_known` aracı ile çekim havzası hesaplanmaktadır. 

```python
--8<-- "examples/analysis/basin_limit_cycle_known.py"
```

## Lyapunov üstelleri ve spektrumu

### Lojistik harita Lyapunov üsteli hesabı

Bu örnekte `observer` kullanarak maksimum Lyapunov üsteli (MLE) ve Lyapunov spektrumu hesabının nasıl yapılacağı gösterilmektedir. Bir boyutlu lojistik harita için spektrum analizi mantıksız olsa da `lyapunov_mle_observer` ve `lyapunov_spectrum_observer` araçlarının numerik olarak ne kadar farklı olduğuna bakılmaktadır.

```python
--8<-- "examples/analysis/lyapunov_logistic_map_demo.py"
```

### Lorenz sistemi Lyapunov üsteli hesabı

Bu örnekte ise Lorenz sistemi için `lyapunov_mle_observer` ve `lyapunov_spectrum_observer` kullanarak Lyapunov üsteli hesaplanmaktadır.

```python
--8<-- "examples/analysis/lyapunov_lorenz_demo.py"
```

### Parametre taraması ile Lyapunov üsteli hesaplama

`sweep.lyapunov_mle_sweep` ile seçilen bir parametre değeri için belli bir aralıkta maksimum Lyapunov üsteli hesabı yapılabilmektedir. Bu örnekte lojistik harita için `r` ∈ [2.5, 4] aralığında maksimum Lyapunov üsteli hesaplanmakta ve sonuç çizdirilmektedir.

```python
--8<-- "examples/analysis/lyapunov_sweep_mle_demo.py"
```

`sweep.lyapunov_spectrum_sweep` ise seçilen bir parametre değeri için belli bir aralıkta tüm Lyapunov spektrumunu hesaplamaktadır. Bu örnekte Lorenz sisteminin `rho` parametresi için [0, 200] aralığında Lyapunov spektrumu hesabı yapılmaktadır.

```python
--8<-- "examples/analysis/lyapunov_sweep_spectrum_demo.py"
```

## Manifoldlar ve yörünge bulucular

### Manifold izleme

Bu örnekte `trace_manifold_1d_map` aracı kullanılarak Henon haritasının bir sabit noktası için hem kararlı hem de kararsız 1-boyutlu manifold hesaplanmaktadır. Sonuç `plot.manifold` ile çizdirilmektedir.

```python
--8<-- "examples/analysis/manifold_henon.py"
```

Bu örnekte ise `trace_manifold_1d_ode` aracı kullanılarak 2-boyutlu bir ODE sisteminin eyer noktasına ait kararlı ve kararsız manifoldlar belirlenerek sonuç çizdirilmektedir.

```python
--8<-- "examples/analysis/manifold_ode_saddle.py"
```

### Homoklinik ve heteroklinik yörünge bulucular

Bu örnekte 2-boyutlu bir ODE sistemi için bir parametre ve denge noktası seçilmekte ve -1.0 değerinden başlayarak [-0.6, -1.2] aralığında seçilen parametre değiştirilerek denge noktası için bir homoklinik yörünge aranmaktadır. Arama için `homoclinic_finder` kullanılmaktadır. Sonrasında homoklinik yörünge tespit edilen parametre değeri için `homoclinic_tracer` ve `plot.manifold` kullanılarak homoklinik yörünge çizdirilmektedir.

```python
--8<-- "examples/analysis/homoclinic_finder_tracer.py"
```

Bu örnekte ise 2-boyutlu başka bir ODE sistemi için iki tane denge noktası seçilmekte ve yine bir parametre için belli bir aralıkta bu iki nokta arasında oluşacak bir heteroklinik yönge aranmaktadır. Arama işlemi `heteroclinic_finder` aracı ile yapılmaktadır. Sonrasında heteroklinik yörünge tespit edilen parametre değeri için `heteroclinic_tracer` ve `plot.manifold` kullanılarak heteroklinik yörünge çizdirilmektedir.

```python
--8<-- "examples/analysis/heteroclinic_finder_tracer.py"
```

## Parametre taraması ve yörünge analizi

### Parametre taramaları

Bu örnekte 1-boyutlu basit bir ODE modelinin `a` parametresi için farklı değerler seçilerek `analysis.sweep.traj_sweep` aracı ile her bir parametre için otomatik olarak bir yörünge (trajectory) hesaplanmaktadır. Hesaplanan yörüngeler `plot.series.multi` ile tek seferde çizdirilmektedir.

```python
--8<-- "examples/analysis/parameter_sweep.py"
```

### Yörünge analizi

Bu örnekte bir simülasyon yapıldıktan sonra elde edilen sonuçlar üzerinde nasıl analiz yapılabileceği gösterilmektedir. `res` simülasyon sonucu olmak üzere `res.analyze()` ile `summary()`, `argmax()`, `zero_crossings()` ve `time_above/below` gibi post-analiz opsiyonları kullanılabilmektedir. 

```python
--8<-- "examples/analysis/trajectory_analysis_demo.py"
```
