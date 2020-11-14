# rows, cols = input().split()
# rows = int(rows)
# cols = int(cols)
# seat = [[0]*cols]*rows
# for i in range(rows):
#     for j in range(cols):
#         seat[i][j] = 0

# print(seat)
# ****************************************
# seat = [[1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]]

# def showSeat(seat):
#     for i in range(len(seat)):
#         for j in range(len(seat[0])):
#             if seat[i][j] == 1:
#                 print('X', end=' ')
#             else:
#                 print('_', end=' ')
#         print()
        
# showSeat(seat)
# ****************************************
success = 0
failure = 0

rows, cols = input().split()
rows = int(rows)
cols = int(cols)


seat = [[0]*cols for i in range(rows)]

num = int(input())
for i in range(num):
    arows, acols = input().split()
    if seat[int(arows)][int(acols)] != 1:
        seat[int(arows)][int(acols)] += 1
        success += 1
    else:
        failure += 1

def showSeat(seat):
    for i in range(rows):
        for j in range(cols):
            if seat[i][j] == 1:
                print('X', end=' ')
            else:
                print('_', end=' ')
        print()
    pass

showSeat(seat)
print("Success =",success,", Failure =", failure)