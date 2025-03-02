import streamlit as st
#from code_executor import code_execution_widget # In-Browser Code Executor disabled

st.title("Python Basics & Best Practices")

# Sidebar persistent code execution widget
#code_execution_widget()

st.markdown("## 📌 Python Basics Reference")

# Expandable Section: zip()
with st.expander("🧩 zip() - Combining Iterables", expanded=False):
    st.markdown("### Basic Usage")
    st.code('''
names = ["Alice", "Bob", "Charlie"]
scores = [85, 90, 88]

# Pair elements from two lists
paired = list(zip(names, scores))
print(paired)  # [('Alice', 85), ('Bob', 90), ('Charlie', 88)]
''', language="python")

    st.markdown("### Handling Uneven Lists")
    st.code('''
from itertools import zip_longest

names = ["Alice", "Bob", "Charlie"]
scores = [85, 90]  # Shorter list

# zip() stops at the shortest iterable
paired = list(zip(names, scores))
print(paired)  # [('Alice', 85), ('Bob', 90)]

# Use zip_longest() to fill missing values
paired_longest = list(zip_longest(names, scores, fillvalue="No Score"))
print(paired_longest)  # [('Alice', 85), ('Bob', 90), ('Charlie', 'No Score')]
''', language="python")

    st.markdown("### Transposing a Matrix")
    st.code('''
matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

# Transpose rows and columns
transposed = list(zip(*matrix))
print(transposed)  # [(1, 4, 7), (2, 5, 8), (3, 6, 9)]
''', language="python")

# Expandable Section: sorted() and key parameter
with st.expander("🔀 sorted() - Custom Sorting", expanded=False):
    st.markdown("### Basic Sorting")
    st.code('''
numbers = [4, 1, 7, 3, 9, 2]

# Sort numbers in ascending order
sorted_numbers = sorted(numbers)
print(sorted_numbers)  # [1, 2, 3, 4, 7, 9]
''', language="python")

    st.markdown("### Sorting with a Key Function")
    st.code('''
words = ["apple", "banana", "cherry", "date"]

# Sort words by length
sorted_words = sorted(words, key=len)
print(sorted_words)  # ['date', 'apple', 'banana', 'cherry']
''', language="python")

    st.markdown("### Advanced Sorting - Sorting Dictionaries")
    st.code('''
students = [
    {"name": "Alice", "grade": 90},
    {"name": "Bob", "grade": 85},
    {"name": "Charlie", "grade": 95}
]

# Sort students by grade
sorted_students = sorted(students, key=lambda x: x["grade"], reverse=True)
print(sorted_students)
# Output: [{'name': 'Charlie', 'grade': 95}, {'name': 'Alice', 'grade': 90}, {'name': 'Bob', 'grade': 85}]
''', language="python")

# Expandable Section: map(), filter(), reduce()
with st.expander("🔄 Functional Programming - map(), filter(), reduce()", expanded=False):
    st.markdown("### map() - Apply a Function to Every Item")
    st.code('''
numbers = [1, 2, 3, 4, 5]

# Square each number
squared = list(map(lambda x: x**2, numbers))
print(squared)  # [1, 4, 9, 16, 25]
''', language="python")

    st.markdown("### filter() - Select Elements Based on a Condition")
    st.code('''
# Keep only even numbers
even_numbers = list(filter(lambda x: x % 2 == 0, numbers))
print(even_numbers)  # [2, 4]
''', language="python")

    st.markdown("### reduce() - Combine Elements into a Single Value")
    st.code('''
from functools import reduce

# Sum all numbers
total = reduce(lambda x, y: x + y, numbers)
print(total)  # 15
''', language="python")

# Expandable Section: enumerate()
with st.expander("🔢 enumerate() - Cleaner Loops", expanded=False):
    st.markdown("### Basic Usage")
    st.code('''
fruits = ["apple", "banana", "cherry"]

# Get index and value from a list
for index, fruit in enumerate(fruits):
    print(index, fruit)
# Output:
# 0 apple
# 1 banana
# 2 cherry
''', language="python")

    st.markdown("### Custom Start Index")
    st.code('''
# Start numbering from 1 instead of 0
for index, fruit in enumerate(fruits, start=1):
    print(index, fruit)
# Output:
# 1 apple
# 2 banana
# 3 cherry
''', language="python")

    st.markdown("### Using enumerate() with Conditionals")
    st.code('''
# Find the index of a specific element
for index, fruit in enumerate(fruits):
    if fruit == "banana":
        print(f"Found banana at index {index}")
# Output: Found banana at index 1
''', language="python")

# Expandable Section: isinstance() and issubclass()
with st.expander("🧐 isinstance() & issubclass() - Type Checking", expanded=False):
    st.markdown("### Basic Usage - Checking Variable Types")
    st.code('''
# Check if a variable is of a specific type
x = 10
print(isinstance(x, int))  # True
print(isinstance(x, float))  # False
''', language="python")

    st.markdown("### Checking Multiple Types")
    st.code('''
# isinstance() can check against multiple types
y = 3.14
print(isinstance(y, (int, float)))  # True, because it's a float
''', language="python")

    st.markdown("### Using issubclass() to Check Class Inheritance")
    st.code('''
# Define a class hierarchy
class Animal:
    pass

class Dog(Animal):
    pass

# Check if Dog is a subclass of Animal
print(issubclass(Dog, Animal))  # True
print(issubclass(Animal, Dog))  # False
''', language="python")

# Expandable Section: List Comprehensions
with st.expander("🔄 List Comprehensions - Compact Loops", expanded=False):
    st.markdown("### Basic Usage - Transforming a List")
    st.code('''
# Square each number in a list
numbers = [1, 2, 3, 4, 5]
squared = [x**2 for x in numbers]
print(squared)  # [1, 4, 9, 16, 25]
''', language="python")

    st.markdown("### Adding a Condition")
    st.code('''
# Keep only even numbers
evens = [x for x in numbers if x % 2 == 0]
print(evens)  # [2, 4]
''', language="python")

    st.markdown("### Nested List Comprehensions")
    st.code('''
# Flatten a matrix into a single list
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
flat = [num for row in matrix for num in row]
print(flat)  # [1, 2, 3, 4, 5, 6, 7, 8, 9]
''', language="python")

# Expandable Section: Unpacking Operators (* and **)
with st.expander("📦 Unpacking Operators - * and **", expanded=False):
    st.markdown("### Basic Usage - Unpacking Lists")
    st.code('''
# Expand a list into function arguments
def add(a, b, c):
    return a + b + c

numbers = [1, 2, 3]
print(add(*numbers))  # 6
''', language="python")

    st.markdown("### Unpacking Dictionaries")
    st.code('''
# Merge two dictionaries using **
dict1 = {"a": 1, "b": 2}
dict2 = {"c": 3, "d": 4}

merged = {**dict1, **dict2}
print(merged)  # {'a': 1, 'b': 2, 'c': 3, 'd': 4}
''', language="python")

    st.markdown("### Using *args and **kwargs in Functions")
    st.code('''
# Accept multiple arguments dynamically
def multiply(*args):
    result = 1
    for num in args:
        result *= num
    return result

print(multiply(2, 3, 4))  # 24
''', language="python")

# Expandable Section: String Formatting (f-strings, format(), % operator)
with st.expander("📝 String Formatting - f-strings, .format(), %", expanded=False):
    st.markdown("### Basic Usage - f-strings")
    st.code('''
name = "Alice"
age = 30

# Embed variables directly into strings
print(f"My name is {name} and I am {age} years old.")
# Output: My name is Alice and I am 30 years old.
''', language="python")

    st.markdown("### Using .format()")
    st.code('''
# Alternative method
print("My name is {} and I am {} years old.".format(name, age))
''', language="python")

    st.markdown("### Formatting Numbers with Precision")
    st.code('''
pi = 3.1415926535

# Round to two decimal places
print(f"Pi rounded: {pi:.2f}")  # Pi rounded: 3.14
''', language="python")
