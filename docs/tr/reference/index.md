# Referans

Referans bölümü, dynlib içerisinde tanımlı yerleşik modelleri listeler. Bu liste `mkdocs-gen-files` eklentisi yardımıyla otomatik olarak oluşturulmaktadır. Kaynak kodlarına sahipseniz bu referansı `tools/gen_model_docs.py` betiğini çalıştırarak yerel olarak üretebilirsiniz.

## Yerleşik modeller

Yerleşik modeller kaynak kodlarında `src/dynlib/models/{map,ode}` kalsörleri içerisinde tanımlı olup `builtin` URI etiketi ile kullanılabilir. Hangi yerleşik modelin hangi eşitliklere sahip olduğunu `print_equations()` metodu yardımıyla çalışma anında da öğrenebilirsiniz.

- [Yerleşik model genel bakışı](models/index.md) — yerleşik map ve ODE modellerinin TOML DSL dosyalarını inceleyin.
