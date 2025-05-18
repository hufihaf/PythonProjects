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


# Array -> Tuple
def select_empty_cell(board):
    coords = gather_empty_cells(board)
    if not coords:
        return False
    return coords[0]


# 2D Array -> Array
def gather_empty_cells(board):
    coords = []
    for i in range(9):
        for j in range(9):
            coord = (i, j)
            if board[i][j] == 0:
                coords.append(coord)
    return coords


# 2D Array + int + int + int -> bool
def is_valid(board, num, row, column):

    # Check the row and column rule
    for i in range (9):
        if board[row][i] == num or board[i][column] == num:
            return False
        
    # Check the block rule
    block_row, block_col = row % 3, column % 3

    for i in range(3):
        for j in range(3):
            if board[j + row - block_row][i + column - block_col] == num:
                return False
    return True


# # Test case
# board = [[0, 7, 0, 5, 3, 0, 9, 8, 0],
#          [0, 0, 0, 6, 0, 0, 2, 5, 7], 
#          [5, 4, 0, 8, 2, 0, 6, 1, 0], 
#          [1, 5, 0, 4, 0, 0, 8, 0, 2], 
#          [3, 0, 6, 0, 0, 0, 7, 0, 0], 
#          [0, 0, 4, 0, 0, 3, 1, 0, 0], 
#          [4, 1, 5, 9, 0, 2, 3, 7, 0], 
#          [0, 6, 0, 0, 0, 0, 4, 0, 0], 
#          [0, 0, 8, 0, 7, 0, 0, 6, 0]]

print(solve(board))
if solve(board):
    for row in board:
        print(row)
