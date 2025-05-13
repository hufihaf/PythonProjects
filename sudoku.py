# 2D Array -> bool
def solve(board):
    empty_cell = select_empty_cell(board)
    if not empty_cell:
        return True
    
    row = empty_cell[0]
    column = empty_cell[1]

    for num in range(1,10):
        if is_valid(board, num, row, column):
            board[row][column] = num
            if solve(board):
                return True
            board[row][column] = 0
    return False

# 2d Array + Array -> Tuple
def select_empty_cell(board, coords):
    pass

# 2D Array -> Array
def gather_empty_cells(board):
    pass

# 2D Array + int + int + int -> bool
def is_valid(board, num, row, column):
    pass

# Test case
board = [[0, 7, 0, 5, 3, 0, 9, 8, 0], 
         [0, 0, 0, 6, 0, 0, 2, 5, 7], 
         [5, 4, 0, 8, 2, 0, 6, 1, 0], 
         [1, 5, 0, 4, 0, 0, 8, 0, 2], 
         [3, 0, 6, 0, 0, 0, 7, 0, 0], 
         [0, 0, 4, 0, 0, 3, 1, 0, 0], 
         [4, 1, 5, 9, 0, 2, 3, 7, 0], 
         [0, 6, 0, 0, 0, 0, 4, 0, 0], 
         [0, 0, 8, 0, 7, 0, 0, 6, 0]]
print(solve(board))
