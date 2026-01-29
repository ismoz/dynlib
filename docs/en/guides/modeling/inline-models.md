# Inline Models

Models don't have to be defined in a separate toml file. You can define models in the same python file using the `inline:` keyword.

## Example

```python
model = '''
inline:
[model]
type = "map"
dtype = "int64"
name = "Collatz Conjecture"

[states]
n = 27

[equations.rhs]
n = "n//2 if n % 2 == 0 else 3*n + 1"

'''
```