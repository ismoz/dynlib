# Referans

Referans bölümü, dynlib içerisinde tanımlı yerleşik modelleri listeler. Bu liste `mkdocs-gen-files` eklentisi yardımıyla otomatik olarak oluşturulmaktadır. Kaynak kodlarına sahipseniz bu referansı `tools/gen_model_docs.py` betiğini çalıştırarak yerel olarak üretebilirsiniz.

## Yerleşik modeller

Yerleşik modeller kaynak kodlarında `src/dynlib/models/{map,ode}` kalsörleri içerisinde tanımlı olup `builtin` URI etiketi ile kullanılabilir. Hangi yerleşik modelin hangi eşitliklere sahip olduğunu `print_equations()` metodu yardımıyla çalışma anında da öğrenebilirsiniz.

- [Yerleşik model genel bakışı](models/index.md) — yerleşik map ve ODE modellerinin TOML DSL dosyalarını inceleyin.

## Dökümantasyonu lokal olarak üretme
- Dmkümantasyon için `mkdocs` kullanılmıştır. Eğer lokal olarak bu dökümasnyonu kendiniz oluşturmak isterseniz aşağıdaki adımları takip edin:

1. MkDocs ve eklentilerini kurun (sanal ortam ile):
   ```bash
   pip install mkdocs mkdocs-material mkdocs-gen-files mkdocs-literate-nav "mkdocstrings[python]" mkdocstrings mkdocs-static-i18n
   ```

2. İlave Markdown eklentilerini kurun:
   ```bash
   pip install pymdown-extensions
   ```

3. Proje klasörü üzerinden aşağıdaki komutları çalıştırın:
   ```bash
   mkdocs serve
   ```
   Or build them:
   ```bash
   mkdocs build
   ```

4. Otomatik olarak üretilen dökümantasyonu üretmek için aşağıdaki komutu çalıştırın:
   ```bash
   python tools/gen_model_docs.py
   ```

Üretilecek olan dökümantasyon `site/` klasörü içinde olacaktır.
