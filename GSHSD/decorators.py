import time
import functools


def timer(func):
    """Print the runtime of the decorated function"""
    @functools.wraps(func)
    def wrapper_decorator(*args, **kwargs):
        start = time.perf_counter()  
        res = func(*args, **kwargs)
        print(f"Finished {func.__name__!r} in {time.perf_counter() - start:.8f} secs")
        return res
    return wrapper_decorator

def debug(func):
    """Print the function signature and return value"""
    @functools.wraps(func)
    def wrapper_debug(*args, **kwargs):
        args_repr = [repr(a) for a in args]
        kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
        signature = ", ".join(args_repr + kwargs_repr)
        print(f"Calling {func.__name__}({signature})")
        value = func(*args, **kwargs)
        print(f"{func.__name__!r} returned {value!r}")
        return value
    return wrapper_debug
