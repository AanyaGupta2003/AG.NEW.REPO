# 1 Explain the purpose and advantages of NumPy in scientific computing and data analysis. How does it enhance Python's capabilities for numerical operations?
Purpose of NumPy

    Efficient Multidimensional Arrays: NumPy provides the powerful ndarray object, which is a versatile n-dimensional array for handling large datasets. These arrays are more efficient and faster compared to Python's built-in lists.

    Mathematical Functions: It includes a vast collection of mathematical functions to operate on arrays, making it easier to perform complex computations without the need for writing loops.

    Linear Algebra: NumPy has built-in support for linear algebra, including operations like matrix multiplication, eigenvalue decomposition, and more.

    Random Number Generation: The package includes tools for generating random numbers, which are essential for simulations and statistical methods.

Advantages of NumPy

    Speed: NumPy operations are executed in C, making them much faster than native Python loops. This speed advantage is crucial for data-intensive tasks.

    Memory Efficiency: The ndarray structure is more memory-efficient than lists. It stores data in contiguous blocks of memory, which reduces overhead.

    Vectorization: NumPy allows vectorized operations, meaning you can apply a function to an entire array without writing explicit loops. This leads to more readable and concise code.

    Interoperability: NumPy arrays can be used with a wide range of scientific and data analysis libraries, such as SciPy, pandas, and scikit-learn, creating a cohesive ecosystem for scientific computing.

    Broadcasting: NumPy supports broadcasting, enabling arithmetic operations on arrays of different shapes. This simplifies coding by eliminating the need for manual reshaping of arrays.

Enhancing Python's Capabilities

    Numerical Efficiency: NumPy brings the computational efficiency of languages like C and Fortran to Python, making it suitable for heavy numerical tasks.

    Ease of Use: With its intuitive syntax and powerful functions, NumPy makes complex numerical and scientific computations more accessible to users.

    Community and Documentation: A strong community and extensive documentation support users in leveraging NumPy effectively, offering numerous tutorials, examples, and resources.
# 2 . Compare and contrast np.mean() and np.average() functions in NumPy. When would you use one over the other?
np.mean()

    Function: np.mean(a, axis=None, dtype=None, out=None, keepdims=<no value>)

    Purpose: Computes the arithmetic mean along the specified axis.

    Usage: Use np.mean() when you need to calculate the simple arithmetic mean of an array or along a specific axis.

np.mean()

    Function: np.mean(a, axis=None, dtype=None, out=None, keepdims=<no value>)

    Purpose: Computes the arithmetic mean along the specified axis.

    Usage: Use np.mean() when you need to calculate the simple arithmetic mean of an array or along a specific axis.

np.average()

    Function: np.average(a, axis=None, weights=None, returned=False)

    Purpose: Computes the weighted average along the specified axis.

    Usage: Use np.average() when you need to calculate the weighted average of an array, considering different weights for different elements.
# 3 . Describe the methods for reversing a NumPy array along different axes. Provide examples for 1D and 2D arrays.
import numpy as np

# Create a 1D array
arr_1d = np.array([1, 2, 3, 4, 5])

# Reverse the array using slicing
reversed_1d_slicing = arr_1d[::-1]
print("Reversed 1D array using slicing:", reversed_1d_slicing)

# Reverse the array using np.flip()
reversed_1d_flip = np.flip(arr_1d)
print("Reversed 1D array using np.flip():", reversed_1d_flip)

# Create a 2D array
arr_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Reverse the array using slicing
reversed_2d_slicing = arr_2d[::-1, ::-1]
print("Reversed 2D array using slicing:")
print(reversed_2d_slicing)

# Reverse the array using np.flip()
reversed_2d_flip = np.flip(arr_2d)
print("Reversed 2D array using np.flip():")
print(reversed_2d_flip)
# Reverse along axis 0 using slicing
reversed_2d_axis0_slicing = arr_2d[::-1, :]
print("Reversed 2D array along axis 0 using slicing:")
print(reversed_2d_axis0_slicing)

# Reverse along axis 0 using np.flip()
reversed_2d_axis0_flip = np.flip(arr_2d, axis=0)
print("Reversed 2D array along axis 0 using np.flip():")
print(reversed_2d_axis0_flip)

# Reverse along axis 1 using slicing
reversed_2d_axis1_slicing = arr_2d[:, ::-1]
print("Reversed 2D array along axis 1 using slicing:")
print(reversed_2d_axis1_slicing)

# Reverse along axis 1 using np.flip()
reversed_2d_axis1_flip = np.flip(arr_2d, axis=1)
print("Reversed 2D array along axis 1 using np.flip():")
print(reversed_2d_axis1_flip)
Reversed 1D array using slicing: [5 4 3 2 1]
Reversed 1D array using np.flip(): [5 4 3 2 1]
Reversed 2D array using slicing:
[[9 8 7]
 [6 5 4]
 [3 2 1]]
Reversed 2D array using np.flip():
[[9 8 7]
 [6 5 4]
 [3 2 1]]
Reversed 2D array along axis 0 using slicing:
[[7 8 9]
 [4 5 6]
 [1 2 3]]
Reversed 2D array along axis 0 using np.flip():
[[7 8 9]
 [4 5 6]
 [1 2 3]]
Reversed 2D array along axis 1 using slicing:
[[3 2 1]
 [6 5 4]
 [9 8 7]]
Reversed 2D array along axis 1 using np.flip():
[[3 2 1]
 [6 5 4]
 [9 8 7]]
# 4 How can you determine the data type of elements in a NumPy array? Discuss the importance of data types in memory management and performance
import numpy as np

# Create a NumPy array
arr = np.array([1, 2, 3, 4, 5])

# Determine the data type of elements
data_type = arr.dtype
print("Data type of elements:", data_type)

Importance of Data Types in Memory Management and Performance

1. Memory Efficiency:

    Compact Storage: Different data types have different memory footprints. For instance, int32 uses 4 bytes per element, while int64 uses 8 bytes per element. Choosing the appropriate data type can save memory, especially when dealing with large datasets.

    Fixed Size: Unlike Python lists, where each element is an object with additional overhead, NumPy arrays store elements in contiguous memory blocks with fixed sizes. This reduces the memory overhead and allows efficient memory allocation and deallocation.

2. Performance Optimization:

    Vectorized Operations: NumPy arrays support vectorized operations, which means that mathematical operations are applied to entire arrays at once, rather than element by element. This is made possible by knowing the fixed data types of array elements, leading to faster execution.

    Low-Level Optimizations: Operations on NumPy arrays are implemented in C and Fortran, leveraging low-level optimizations specific to data types. This results in significant performance improvements compared to native Python loops.

    Alignment and Cache Utilization: Knowing the data type allows the compiler to optimize memory alignment and cache utilization, further enhancing performance.

3. Type-Specific Functions:

    Specialized Computations: Certain computations and algorithms are more efficient with specific data types. For example, some algorithms may benefit from using floating-point arithmetic (float32 or float64), while others may be optimized for integer arithmetic (int32 or int64).

    Precision Control: Choosing the right data type allows control over the precision of calculations. For instance, float64 provides higher precision compared to float32, which is crucial for scientific computations.
  Cell In[2], line 10
    Importance of Data Types in Memory Management and Performance
               ^
SyntaxError: invalid syntax
# 5 Define ndarrays in NumPy and explain their key features. How do they differ from standard Python lists?
ndarray stands for N-dimensional array. It is the fundamental data structure in NumPy and represents a grid of values, all of the same type, indexed by a tuple of nonnegative integers.
Key Features of ndarrays

    Homogeneous Data Types: All elements in a NumPy array are of the same type, ensuring consistent behavior and efficient memory usage.

    Multidimensional: NumPy arrays can have multiple dimensions (hence the name ndarray), allowing for the representation of complex data structures like matrices and tensors.

    Fast and Efficient: Operations on NumPy arrays are implemented in C, making them much faster than equivalent operations on Python lists.

    Vectorized Operations: NumPy supports vectorized operations, which means you can perform element-wise operations on arrays without explicit loops. This leads to more concise and readable code.

    Broadcasting: NumPy arrays support broadcasting, which allows arithmetic operations on arrays of different shapes. This can simplify many operations and eliminate the need for manual reshaping.

    Rich Functionality: NumPy provides a plethora of functions for array manipulation, including mathematical, statistical, and linear algebra operations.

How ndarrays Differ from Standard Python Lists

    Homogeneity:

        ndarrays: All elements must be of the same type.

        Python Lists: Can contain elements of different types.

    Performance:

        ndarrays: Operations are faster due to the implementation in C and the use of contiguous memory blocks.

        Python Lists: Slower as they involve more overhead and use pointers for each element.

    Memory Efficiency:

        ndarrays: More memory-efficient due to contiguous memory allocation and fixed-size elements.

        Python Lists: Less memory-efficient due to the overhead associated with dynamic typing and pointers.

    Vectorized Operations:

        ndarrays: Support vectorized operations that apply functions to entire arrays at once.

        Python Lists: Require explicit loops for element-wise operations.

    Broadcasting:

        ndarrays: Support broadcasting, allowing operations on arrays of different shapes.

        Python Lists: Do not support broadcasting; manual handling is required.

    Multidimensional:

        ndarrays: Can represent multidimensional arrays with ease.

        Python Lists: Multidimensional arrays are represented as lists of lists, which can be cumbersome to manage.
                                                                                                                     
  import numpy as np

# Create a 2D array
arr = np.array([[1, 2, 3], [4, 5, 6]])

# Perform a vectorized operation
result = arr + 10
print(result)


 # Create a 2D list
lst = [[1, 2, 3], [4, 5, 6]]

# Perform the same operation (requires explicit loops)
result = [[x + 10 for x in row] for row in lst]
print(result)
                                                                                                                    
Traceback (most recent call last):
  File "/lib/python3.12/site-packages/pyodide_kernel/kernel.py", line 90, in run
    code = await self.lite_transform_manager.transform_cell(code)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/lib/python3.12/site-packages/pyodide_kernel/litetransform.py", line 34, in transform_cell
    lines = await self.do_token_transforms(lines)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/lib/python3.12/site-packages/pyodide_kernel/litetransform.py", line 39, in do_token_transforms
    changed, lines = await self.do_one_token_transform(lines)
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/lib/python3.12/site-packages/pyodide_kernel/litetransform.py", line 59, in do_one_token_transform
    tokens_by_line = make_tokens_by_line(lines)
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/lib/python3.12/site-packages/IPython/core/inputtransformer2.py", line 535, in make_tokens_by_line
    for token in tokenutil.generate_tokens_catch_errors(
  File "/lib/python3.12/site-packages/IPython/utils/tokenutil.py", line 40, in generate_tokens_catch_errors
    for token in tokenize.generate_tokens(readline):
  File "/lib/python312.zip/tokenize.py", line 541, in _generate_tokens_from_c_tokenizer
    raise e from None
  File "/lib/python312.zip/tokenize.py", line 537, in _generate_tokens_from_c_tokenizer
    for info in it:
  File "<string>", line 54
    import numpy as np
                      ^
IndentationError: unindent does not match any outer indentation level

If you suspect this is an IPython 8.23.0 bug, please report it at:
    https://github.com/ipython/ipython/issues
or send an email to the mailing list at ipython-dev@python.org

You can print a more detailed traceback right now with "%tb", or use "%debug"
to interactively debug it.

Extra-detailed tracebacks for bug-reporting purposes can be enabled via:
    %config Application.verbose_crash=True

# 6 Analyze the performance benefits of NumPy arrays over Python lists for large-scale numerical operations.
1. Memory Efficiency
2. Speed and Performance
3. Reduced Python Overhead
4. Rich Mathematical Functions
import numpy as np
import time

# Create two large NumPy arrays
arr1 = np.arange(size)
arr2 = np.arange(size)

# Measure the time for element-wise addition using NumPy arrays
start_time = time.time()
result = arr1 + arr2
end_time = time.time()
print("Time taken using NumPy arrays:", end_time - start_time, "seconds")
  Cell In[3], line 1
    1. Memory Efficiency
       ^
SyntaxError: invalid syntax
# 7 . Compare vstack() and hstack() functions in NumPy. Provide examples demonstrating their usage and output.
vstack()

    Purpose: Stacks arrays in sequence vertically (row-wise).

    Behavior: It requires that the arrays have the same number of columns (or second dimension).
import numpy as np

# Create two 1D arrays
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])

# Vertical stacking
result_vstack = np.vstack((arr1, arr2))
print("Vertical Stacking (vstack):")
print(result_vstack)
Vertical Stacking (vstack):
[[1 2 3]
 [4 5 6]]
hstack()

    Purpose: Stacks arrays in sequence horizontally (column-wise).

    Behavior: It requires that the arrays have the same number of rows (or first dimension).
import numpy as np

# Create two 1D arrays
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])

# Horizontal stacking
result_hstack = np.hstack((arr1, arr2))
print("Horizontal Stacking (hstack):")
print(result_hstack)
Horizontal Stacking (hstack):
[1 2 3 4 5 6]
  Cell In[4], line 3
    Purpose: Stacks arrays in sequence vertically (row-wise).
    ^
IndentationError: unexpected indent
# 8 Explain the differences between fliplr() and flipud() methods in NumPy, including their effects on various array dimensions.
fliplr()

    Purpose: Flips the array in the left-right direction (horizontal flip).

    Behavior: Reverses the order of columns in a 2D array.

    Applicable: Works on arrays with at least two dimensions.
        
flipud()

    Purpose: Flips the array in the up-down direction (vertical flip).

    Behavior: Reverses the order of rows in a 2D array.

    Applicable: Works on arrays with at least one dimension.
# 9  Discuss the functionality of the array_split() method in NumPy. How does it handle uneven splits?
Functionality of array_split()
numpy.array_split(ary, indices_or_sections, axis=0)

Example 1: Splitting Evenly
import numpy as np

# Create an array
arr = np.array([1, 2, 3, 4, 5, 6])

# Split the array into 3 equal parts
result = np.array_split(arr, 3)
print("Even split:")
for sub_arr in result:
    print(sub_arr)
Example 2: Handling Uneven Splits
import numpy as np

# Create an array
arr = np.array([1, 2, 3, 4, 5, 6, 7])

# Split the array into 3 parts (uneven split)
result = np.array_split(arr, 3)
print("Uneven split:")
for sub_arr in result:
    print(sub_arr)
  Cell In[6], line 1
    Functionality of array_split()
                  ^
SyntaxError: invalid syntax
# 10. . Explain the concepts of vectorization and broadcasting in NumPy. How do they contribute to efficient array operations
Concept: Vectorization refers to the process of applying operations to entire arrays or large chunks of data at once, rather than using explicit loops to operate on individual elements. 

import numpy as np

# Create two large arrays
arr1 = np.arange(1e7)
arr2 = np.arange(1e7, 2e7)

# Vectorized addition
result = arr1 + arr2

Broadcasting

Concept: Broadcasting is a mechanism that allows NumPy to perform arithmetic operations on arrays of different shapes

import numpy as np

# Create an array and a scalar
arr = np.array([[1, 2, 3], [4, 5, 6]])
scalar = 10

# Broadcasting addition
result = arr + scalar

 Contribute to Efficient Array Operations
Reduced Overhead
Parallelism
Conciseness
Memory Efficiency
  Cell In[7], line 1
    Concept: Vectorization refers to the process of applying operations to entire arrays or large chunks of data at once, rather than using explicit loops to operate on individual elements.
                           ^
SyntaxError: invalid syntax
#                Practical 
# 1 . Create a 3x3 NumPy array with random integers between 1 and 100. Then, interchange its rows and columns.
import numpy as np

# Create a 3x3 array with random integers between 1 and 100
np.random.seed(42)  # For reproducibility
array_3x3 = np.random.randint(1, 101, (3, 3))
print("Original 3x3 array:")
print(array_3x3)

# Transpose the array (interchange rows and columns)
transposed_array = np.transpose(array_3x3)
print("\nTransposed array:")
print(transposed_array)
Original 3x3 array:
[[52 93 15]
 [72 61 21]
 [83 87 75]]

Transposed array:
[[52 72 83]
 [93 61 87]
 [15 21 75]]
# 2 Generate a 1D NumPy array with 10 elements. Reshape it into a 2x5 array, then into a 5x2 array
import numpy as np

# Generate a 1D NumPy array with 10 elements
array_1d = np.arange(10)
print("Original 1D array:")
print(array_1d)

# Reshape the array into a 2x5 array
array_2x5 = array_1d.reshape((2, 5))
print("\nReshaped into 2x5 array:")
print(array_2x5)

# Reshape the array into a 5x2 array
array_5x2 = array_2x5.reshape((5, 2))
print("\nReshaped into 5x2 array:")
print(array_5x2)
Original 1D array:
[0 1 2 3 4 5 6 7 8 9]

Reshaped into 2x5 array:
[[0 1 2 3 4]
 [5 6 7 8 9]]

Reshaped into 5x2 array:
[[0 1]
 [2 3]
 [4 5]
 [6 7]
 [8 9]]
# 3 Create a 4x4 NumPy array with random float values. Add a border of zeros around it, resulting in a 6x6 array.
import numpy as np

# Create a 4x4 array with random float values
np.random.seed(42)  # For reproducibility
array_4x4 = np.random.rand(4, 4)
print("Original 4x4 array:")
print(array_4x4)

# Add a border of zeros around the 4x4 array to form a 6x6 array
array_6x6 = np.pad(array_4x4, pad_width=1, mode='constant', constant_values=0)
print("\n6x6 array with a border of zeros:")
print(array_6x6)
Original 4x4 array:
[[0.37454012 0.95071431 0.73199394 0.59865848]
 [0.15601864 0.15599452 0.05808361 0.86617615]
 [0.60111501 0.70807258 0.02058449 0.96990985]
 [0.83244264 0.21233911 0.18182497 0.18340451]]

6x6 array with a border of zeros:
[[0.         0.         0.         0.         0.         0.        ]
 [0.         0.37454012 0.95071431 0.73199394 0.59865848 0.        ]
 [0.         0.15601864 0.15599452 0.05808361 0.86617615 0.        ]
 [0.         0.60111501 0.70807258 0.02058449 0.96990985 0.        ]
 [0.         0.83244264 0.21233911 0.18182497 0.18340451 0.        ]
 [0.         0.         0.         0.         0.         0.        ]]
# 4 Using NumPy, create an array of integers from 10 to 60 with a step of 5.
import numpy as np

# Create the array
array = np.arange(10, 65, 5)
print(array)
[10 15 20 25 30 35 40 45 50 55 60]
# 5  Create a NumPy array of strings ['python', 'numpy', 'pandas']. Apply different case transformations (uppercase, lowercase, title case, etc.) to each element
import numpy as np

# Create a NumPy array of strings
array = np.array(['python', 'numpy', 'pandas'])

# Apply different case transformations
uppercase_array = np.char.upper(array)
lowercase_array = np.char.lower(array)
titlecase_array = np.char.title(array)

# Display the results
print("Original array:", array)
print("Uppercase:", uppercase_array)
print("Lowercase:", lowercase_array)
print("Title case:", titlecase_array)
Original array: ['python' 'numpy' 'pandas']
Uppercase: ['PYTHON' 'NUMPY' 'PANDAS']
Lowercase: ['python' 'numpy' 'pandas']
Title case: ['Python' 'Numpy' 'Pandas']
# 6 Generate a NumPy array of words. Insert a space between each character of every word in the array.
import numpy as np

# Create a NumPy array of words
words = np.array(['python', 'numpy', 'pandas'])

# Function to insert space between characters
def insert_spaces(word):
    return ' '.join(word)

# Apply the function to each element in the array
spaced_words = np.vectorize(insert_spaces)(words)

# Display the results
print("Original array:", words)
print("Array with spaces between characters:", spaced_words)
Original array: ['python' 'numpy' 'pandas']
Array with spaces between characters: ['p y t h o n' 'n u m p y' 'p a n d a s']
# 7  Create two 2D NumPy arrays and perform element-wise addition, subtraction, multiplication, and division.
import numpy as np

# Create two 2D arrays
array1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
array2 = np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]])

# Perform element-wise addition
addition = array1 + array2
print("Element-wise addition:")
print(addition)

# Perform element-wise subtraction
subtraction = array1 - array2
print("\nElement-wise subtraction:")
print(subtraction)

# Perform element-wise multiplication
multiplication = array1 * array2
print("\nElement-wise multiplication:")
print(multiplication)

# Perform element-wise division
division = array1 / array2
print("\nElement-wise division:")
print(division)
Element-wise addition:
[[10 10 10]
 [10 10 10]
 [10 10 10]]

Element-wise subtraction:
[[-8 -6 -4]
 [-2  0  2]
 [ 4  6  8]]

Element-wise multiplication:
[[ 9 16 21]
 [24 25 24]
 [21 16  9]]

Element-wise division:
[[0.11111111 0.25       0.42857143]
 [0.66666667 1.         1.5       ]
 [2.33333333 4.         9.        ]]
# 8 Use NumPy to create a 5x5 identity matrix, then extract its diagonal elements.
import numpy as np

# Create a 5x5 identity matrix
identity_matrix = np.eye(5)
print("5x5 Identity Matrix:")
print(identity_matrix)

# Extract the diagonal elements
diagonal_elements = np.diag(identity_matrix)
print("\nDiagonal Elements:")
print(diagonal_elements)
5x5 Identity Matrix:
[[1. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0.]
 [0. 0. 1. 0. 0.]
 [0. 0. 0. 1. 0.]
 [0. 0. 0. 0. 1.]]

Diagonal Elements:
[1. 1. 1. 1. 1.]
# 9 Generate a NumPy array of 100 random integers between 0 and 1000. Find and display all prime numbers in this array
import numpy as np

# Function to check if a number is prime
def is_prime(n):
    if n <= 1:
        return False
    for i in range(2, int(np.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True

# Generate a NumPy array of 100 random integers between 0 and 1000
np.random.seed(42)  # For reproducibility
random_integers = np.random.randint(0, 1001, 100)
print("Array of random integers:")
print(random_integers)

# Find and display all prime numbers in the array
prime_numbers = [num for num in random_integers if is_prime(num)]
print("\nPrime numbers in the array:")
print(prime_numbers)
Array of random integers:
[102 435 860 270 106  71 700  20 614 121 466 214 330 458  87 372  99 871
 663 130 661 308 769 343 491 413 805 385 191 955 276 160 459 313  21 252
 747 856 560 474  58 510 681 475 699 975 782 189 957 686 957 562 875 566
 243 831 504 130 484 818 646  20 840 166 273 387 600 315  13 241 776 345
 564 897 339  91 366 955 454 427 508 775 942  34 205  80 931 561 871 387
   1 389 565 105 771 821 476 702 401 729]

Prime numbers in the array:
[71, 661, 769, 491, 191, 313, 13, 241, 389, 821, 401]
# 10  Create a NumPy array representing daily temperatures for a month. Calculate and display the weekly averages.
import numpy as np

# Generate a NumPy array representing daily temperatures for a month (30 days)
np.random.seed(42)  # For reproducibility
daily_temperatures = np.random.randint(20, 40, 30)
print("Daily temperatures for a month:")
print(daily_temperatures)

# Calculate weekly averages
# We assume the month has exactly 4 weeks (30 days total, with each week being 7 days except the last one with 9 days)
week1_avg = np.mean(daily_temperatures[:7])
week2_avg = np.mean(daily_temperatures[7:14])
week3_avg = np.mean(daily_temperatures[14:21])
week4_avg = np.mean(daily_temperatures[21:30])  # Last week may have fewer days

weekly_averages = [week1_avg, week2_avg, week3_avg, week4_avg]
print("\nWeekly averages:")
for i, avg in enumerate(weekly_averages, start=1):
    print(f"Week {i} average temperature: {avg:.2f}")
Daily temperatures for a month:
[26 39 34 30 27 26 38 30 30 23 27 22 21 31 25 21 20 31 31 36 29 35 34 34
 38 31 39 22 24 38]

Weekly averages:
Week 1 average temperature: 31.43
Week 2 average temperature: 26.29
Week 3 average temperature: 27.57
Week 4 average temperature: 32.78
