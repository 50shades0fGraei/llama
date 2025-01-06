from .fibonacci import fibonacci_recursive, fibonacci_iterative, fibonacci_memoized

def test_fibonacci_recursive():
    assert fibonacci_recursive(10) == 55

def test_fibonacci_iterative():
    assert fibonacci_iterative(10) == 55

def test_fibonacci_memoized():
    assert fibonacci_memoized(10) == 55


# fibonacci.py

def fibonacci_recursive(n: int) -> int:
    if n <= 1:
        return n
    return fibonacci_recursive(n-1) + fibonacci_recursive(n-2)

def fibonacci_iterative(n: int) -> int:
    if n <= 1:
        return n
    fib_prev = 0
    fib_curr = 1
    for _ in range(2, n+1):
        fib_next = fib_prev + fib_curr
        fib_prev = fib_curr
        fib_curr = fib_next
    return fib_curr

def fibonacci_memoized(n: int, memo: dict = {}) -> int:
    if n < 0:
        raise ValueError("n must be a non-negative integer")
    elif n <= 1:
        return n
    elif n in memo:
        return memo[n]
    else:
        result = fibonacci_memoized(n-1, memo) + fibonacci_memoized(n-2, memo)
        memo[n] = result
        return result


Run pytest:

```
bash
pytest .github/Pytest
```
