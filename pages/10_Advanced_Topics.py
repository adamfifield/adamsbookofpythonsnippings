import streamlit as st
#import functools
#import asyncio
#import threading
#import logging
#from code_executor import code_execution_widget # In-Browser Code Executor disabled

st.title("Advanced Python Topics")

# Sidebar persistent code execution widget (temporarily disabled)
# code_execution_widget()

st.markdown("## 📌 Advanced Python Techniques")

# Expandable Section: Asynchronous Programming
with st.expander("⚡ Asynchronous Programming with asyncio", expanded=False):
    st.markdown("### Running Async Tasks")
    st.code('''
import asyncio

async def say_hello():
    await asyncio.sleep(2)
    print("Hello, Async!")

# Run the async function
asyncio.run(say_hello())
''', language="python")

    st.markdown("### Running Multiple Async Tasks")
    st.code('''
async def task1():
    await asyncio.sleep(1)
    print("Task 1 Done")

async def task2():
    await asyncio.sleep(2)
    print("Task 2 Done")

# Run tasks concurrently
async def main():
    await asyncio.gather(task1(), task2())

asyncio.run(main())
''', language="python")

# Expandable Section: Threading & Multiprocessing
with st.expander("🔄 Threading & Multiprocessing", expanded=False):
    st.markdown("### Running Multiple Threads")
    st.code('''
import threading

def print_hello():
    print("Hello from thread")

# Create and start a thread
thread = threading.Thread(target=print_hello)
thread.start()
thread.join()
''', language="python")

    st.markdown("### Running Parallel Tasks with multiprocessing")
    st.code('''
from multiprocessing import Pool

def square(x):
    return x * x

with Pool(4) as p:
    results = p.map(square, [1, 2, 3, 4, 5])

print(results)  # [1, 4, 9, 16, 25]
''', language="python")

# Expandable Section: Decorators & Functional Programming
with st.expander("🛠️ Decorators & Functional Programming", expanded=False):
    st.markdown("### Creating a Simple Decorator")
    st.code('''
def my_decorator(func):
    def wrapper():
        print("Before function call")
        func()
        print("After function call")
    return wrapper

@my_decorator
def say_hello():
    print("Hello!")

say_hello()
''', language="python")

    st.markdown("### Using functools.lru_cache for Memoization")
    st.code('''
import functools

@functools.lru_cache(maxsize=100)
def fib(n):
    if n < 2:
        return n
    return fib(n-1) + fib(n-2)

print(fib(10))  # Faster due to caching
''', language="python")

# Expandable Section: Metaprogramming & Reflection
with st.expander("🧩 Metaprogramming & Reflection", expanded=False):
    st.markdown("### Inspecting Objects with Reflection")
    st.code('''
class MyClass:
    def method(self):
        pass

obj = MyClass()

# Get object attributes
print(dir(obj))

# Check if object has a method
print(hasattr(obj, "method"))
''', language="python")

# Expandable Section: Logging & Debugging
with st.expander("📜 Logging & Debugging", expanded=False):
    st.markdown("### Basic Logging in Python")
    st.code('''
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logging.info("This is an info message")
''', language="python")

    st.markdown("### Handling Errors with Logging")
    st.code('''
try:
    1 / 0  # Division by zero
except ZeroDivisionError:
    logging.error("An error occurred", exc_info=True)
''', language="python")

# Expandable Section: Working with Python Internals
with st.expander("⚙️ Understanding Python Internals", expanded=False):
    st.markdown("### Bytecode Inspection with dis")
    st.code('''
import dis

def example_function():
    return sum([1, 2, 3])

# Disassemble function bytecode
dis.dis(example_function)
''', language="python")

    st.markdown("### Using sys for System Information")
    st.code('''
import sys

# Get Python version
print(sys.version)

# Get system path
print(sys.path)
''', language="python")
