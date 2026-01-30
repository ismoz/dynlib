# Satır İçi (Inline) Modeller

Modellerin ayrı bir toml dosyasında tanımlanması gerekmez. `inline:` anahtar kelimesini kullanarak modelleri aynı python dosyasında tanımlayabilirsiniz.

## Örnek

```python
model = '''
inline:
[model]
type = "map"
dtype = "int64"
name = "Collatz Conjecture" # Collatz Sanısı

[states]
n = 27

[equations.rhs]
# n çift ise n//2, değilse 3*n + 1
n = "n//2 if n % 2 == 0 else 3*n + 1"

'''
```