Here's the complete updated code with example usage:

# Fibonacci Sequence Implementations
*Recursive Implementation*
```
def fibonacci_recursive(n: int) -> int:
    """Calculate the nth Fibonacci number recursively."""
    if n <= 1:
        return n
    return fibonacci_recursive(n-1) + fibonacci_recursive(n-2)
```

*Iterative Implementation*
```
def fibonacci_iterative(n: int) -> int:
    """Calculate the nth Fibonacci number iteratively."""
    if n <= 1:
        return n
    fib_prev = 0
    fib_curr = 1
    for _ in range(2, n+1):
        fib_next = fib_prev + fib_curr
        fib_prev = fib_curr
        fib_curr = fib_next
    return fib_curr
```

*Memoized Implementation*
```
def fibonacci_memoized(n: int, memo: dict = None) -> int:
    """
    Calculate the nth Fibonacci number with memoization.

    Args:
        n (int): Position of the Fibonacci number.
        memo (dict): Dictionary storing previously calculated Fibonacci numbers.

    Returns:
        int: The nth Fibonacci number.
    """
    if memo is None:
        memo = {}

    if n < 0:
        raise ValueError("n must be a non-negative integer")

    if n <= 1:
        return n
    elif n in memo:
        return memo[n]
    else:
        result = fibonacci_memoized(n-1, memo) + fibonacci_memoized(n-2, memo)
        memo[n] = result
        return result
```

# Example Usage
```
print(fibonacci_recursive(10))  # Output: 55
print(fibonacci_iterative(10))  # Output: 55
print(fibonacci_memoized(10))   # Output: 55
```
