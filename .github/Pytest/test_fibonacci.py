Here are Fibonacci sequence implementations in Python:

# Recursive Implementation
```
def fibonacci(n):
    """
    Calculate the nth Fibonacci number recursively.
    
    Args:
        n (int): Position of the Fibonacci number.
    
    Returns:
        int: The nth Fibonacci number.
    """
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
``)

## Iterative Implementation
```python
def fibonacci(n):
    """
    Calculate the nth Fibonacci number iteratively.
    
    Args:
        n (int): Position of the Fibonacci number.
    
    Returns:
        int: The nth Fibonacci number.
    """
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

# Memoized Implementation (Efficient)
```
def fibonacci(n, memo={}):
    """
    Calculate the nth Fibonacci number with memoization.
    
    Args:
        n (int): Position of the Fibonacci number.
        memo (dict): Dictionary storing previously calculated Fibonacci numbers.
    
    Returns:
        int: The nth Fibonacci number.
    """
    if n <= 1:
        return n
    elif n in memo:
        return memo[n]
    else:
        result = fibonacci(n-1, memo) + fibonacci(n-2, memo)
        memo[n] = result
        return result
```

# Example Usage
```
print(fibonacci(10))  # Output: 55
```Here are Fibonacci sequence implementations in Python:

# Recursive Implementation
```
def fibonacci(n):
    """
    Calculate the nth Fibonacci number recursively.
    
    Args:
        n (int): Position of the Fibonacci number.
    
    Returns:
        int: The nth Fibonacci number.
    """
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
``)

## Iterative Implementation
```python
def fibonacci(n):
    """
    Calculate the nth Fibonacci number iteratively.
    
    Args:
        n (int): Position of the Fibonacci number.
    
    Returns:
        int: The nth Fibonacci number.
    """
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

# Memoized Implementation (Efficient)
```
def fibonacci(n, memo={}):
    """
    Calculate the nth Fibonacci number with memoization.
    
    Args:
        n (int): Position of the Fibonacci number.
        memo (dict): Dictionary storing previously calculated Fibonacci numbers.
    
    Returns:
        int: The nth Fibonacci number.
    """
    if n <= 1:
        return n
    elif n in memo:
        return memo[n]
    else:
        result = fibonacci(n-1, memo) + fibonacci(n-2, memo)
        memo[n] = result
        return result
```

# Example Usage
```
print(fibonacci(10))  # Output: 55
```
