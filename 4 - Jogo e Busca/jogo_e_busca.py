import copy
import heapq
import time
import sys
from collections import deque

# Estado final do Jogo dos 8
GOAL_STATE = [
    [1, 2, 3],
    [8, 0, 4],
    [7, 6, 5]  # 0 representa o espaço vazio
]

# Movimentos possíveis: cima, baixo, esquerda, direita
MOVES = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1)
}

def print_board(board):
    for row in board:
        print(" ".join(str(x) if x != 0 else " " for x in row))
    print()

def find_zero(board):
    for i in range(3):
        for j in range(3):
            if board[i][j] == 0:
                return i, j
    
    raise ValueError(f"Valor 0 não existe no board_state.")

def is_goal(board):
    return board == GOAL_STATE

def get_neighbors(board):
    neighbors = []
    x, y = find_zero(board)
    for move, (dx, dy) in MOVES.items():
        nx, ny = x + dx, y + dy
        if 0 <= nx < 3 and 0 <= ny < 3:
            new_board = copy.deepcopy(board)
            new_board[x][y], new_board[nx][ny] = new_board[nx][ny], new_board[x][y]
            neighbors.append((new_board, move))
    return neighbors

# ===================== BUSCA EM LARGURA =====================
def bfs(start_board):
    queue = deque([(start_board, [])])
    visited = set()
    visited.add(str(start_board))

    while queue:
        current_board, path = queue.popleft()

        if is_goal(current_board):
            return path

        for neighbor, move in get_neighbors(current_board):
            if str(neighbor) not in visited:
                visited.add(str(neighbor))
                queue.append((neighbor, path + [move]))
    return None

# ===================== BUSCA A* =====================
def distance_from_goal_position(value):
    for goal_x in range(3):
         for goal_y in range(3):
            if value == GOAL_STATE[goal_x][goal_y]:
                return goal_x, goal_y
    raise ValueError(f"Valor {value} não existe no GOAL_STATE.")

def manhattan_distance(board):
    distance = 0
    for i in range(3):
        for j in range(3):
            value = board[i][j]
            if value != 0:
                goal_x, goal_y = distance_from_goal_position(value)
                distance += abs(i - goal_x) + abs(j - goal_y)
    return distance

def a_star(start_board):
    heap = []
    heapq.heappush(heap, (manhattan_distance(start_board), 0, start_board, []))
    visited = {str(start_board): 0}

    while heap:
        _, cost, current_board, path = heapq.heappop(heap)
        if is_goal(current_board):
            return path

        for neighbor, move in get_neighbors(current_board):
            new_cost = cost + 1
            neighbor_key = str(neighbor)
            
            # só visita se for novo ou se encontrou caminho melhor
            if neighbor_key not in visited or new_cost < visited[neighbor_key]:
                visited[neighbor_key] = new_cost
                priority = new_cost + manhattan_distance(neighbor)
                heapq.heappush(heap, (priority, new_cost, neighbor, path + [move]))
    return None

# ===================== INTERFACE =====================
def main():
    
    # Exemplo de estado inicial
    """
    start_board = [
        [2, 0, 3],
        [1, 7, 4],
        [6, 8, 5]
    ]
    """
    """
    start_board = [
        [8, 7, 6],
        [1, 0, 5],
        [2, 3, 4]
    ]
    """
    
    start_board = [
        [2, 1, 3],
        [8, 0, 4],
        [7, 6, 5]
    ]
    
    
    while True:
    
        print("="*50)
        print("Estado inicial:")
        print_board(start_board)

        choice = input("Escolha o método de busca: (1) BFS / (2) A* / (3) Sair: ")

        if choice == "1":
            start = time.time()
            solution = bfs(start_board)
            end = time.time()
        
            print(f"Tempo de execução: {end - start:.8f} segundos")
            print("Solução encontrada com BFS:")
        elif choice == "2":
            start = time.time()
            solution = a_star(start_board)
            end = time.time()
        
            print(f"Tempo de execução: {end - start:.8f} segundos")
            print("Solução encontrada com A*:")
        elif choice == "3":
            sys.exit()
        else:
            raise ValueError(f"Opção inexistente: {choice}")

        if solution:
            print(f"Movimentos: {solution}")
            current = copy.deepcopy(start_board)

            for move in solution:
                for neighbor, m in get_neighbors(current):
                    if m == move:
                        current = neighbor
                        print_board(current)
                        break
        else:
            print("Nenhuma solução encontrada.")

if __name__ == "__main__":
    main()
