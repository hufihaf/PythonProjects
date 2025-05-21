import numpy as np
import matplotlib.pyplot as plt

a = np.array([[[1, 2], [3, 4]],
              [[5, 6], [7, 8]]])
print(a.ndim) # Outputs the amount of dims
print(a)
print(a.shape) # Outputs dimension lengths (2, 2, 2)
print(a[0, 1, 1]) # Output: 4 
                  # dimensions go outisde in
print(a.size) # Output: 8
print(np.empty(3)) # Output is an array with random elements
print("\n\n\n")
print(np.arange(4)) # Output: [0 1 2 3]


