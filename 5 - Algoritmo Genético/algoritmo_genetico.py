import numpy as np
import random

# Parâmetros do Algoritmo Genético

POP_SIZE = 10            # Número de indivíduos na população
NUM_GENERATIONS = 25     # Número de gerações a serem executadas
CROSSOVER_RATE = 0.70   # Taxa de Crossover (70%)
MUTATION_RATE = 0.01    # Taxa de Mutação (1%)
CHROMOSOME_LENGTH = 9   # Comprimento do cromossomo binário (para [-10, 10])
LOWER_BOUND = -10.0     # Limite inferior do intervalo X
UPPER_BOUND = 10.0      # Limite superior do intervalo X
TORNEIO_SIZE = 2        # Tamanho do Torneio para Seleção

# --- 2. Funções Auxiliares do AG ---

def fitness_function(x):
    """
    Função de aptidão a ser maximizada: f(x) = x^2 - 3x + 4
    """
    return x**2 - 3*x + 4

def binary_to_decimal(chromosome):
    """
    Converte o cromossomo binário (vetor) em seu valor decimal.
    """
    return int("".join(map(str, chromosome)), 2)

def decode_chromosome(chromosome):
    """
    Decodifica o cromossomo binário para o valor real 'x' no intervalo [LOWER_BOUND, UPPER_BOUND].
    
    Mapeamento Linear:
    x = LOWER_BOUND + (UPPER_BOUND - LOWER_BOUND) * (decimal_value / (2^CHROMOSOME_LENGTH - 1))
    """
    decimal_value = binary_to_decimal(chromosome)
    
    # Range total de valores representáveis
    max_decimal = (2 ** CHROMOSOME_LENGTH) - 1
    
    # Mapeamento para o intervalo [LOWER_BOUND, UPPER_BOUND]
    x = LOWER_BOUND + (UPPER_BOUND - LOWER_BOUND) * (decimal_value / max_decimal)
    return x

def initialize_population(pop_size, chromo_len):
    """
    Cria a população inicial de forma aleatória.
    """
    # np.random.randint(0, 2, size=(pop_size, chromo_len)) gera uma matriz
    # de tamanho (pop_size x chromo_len) com valores 0 ou 1.
    return np.random.randint(0, 2, size=(pop_size, chromo_len)).tolist()

def tournament_selection(population, fitnesses, k):
    """
    Seleção por Torneio (seleciona o indivíduo com melhor fitness de 'k' candidatos aleatórios).
    """
    selected_parent = None
    best_fitness = -float('inf')
    
    # Seleciona 'k' indivíduos aleatórios para o torneio
    tournament_indices = random.sample(range(len(population)), k)
    
    for index in tournament_indices:
        if fitnesses[index] > best_fitness:
            best_fitness = fitnesses[index]
            selected_parent = population[index]
            
    return selected_parent

def one_point_crossover(parent1, parent2, rate):
    """
    Crossover de um ponto.
    """
    if random.random() < rate:
        # Escolhe um ponto de corte aleatório (excluindo os extremos)
        crossover_point = random.randint(1, len(parent1) - 1)
        
        # Cria os filhos trocando as caudas
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        return child1, child2
    else:
        # Sem crossover, os pais são copiados diretamente para a próxima geração
        return parent1[:], parent2[:]

def mutation(chromosome, rate):
    """
    Mutação: inverte o bit com a probabilidade 'rate'.
    """
    mutated_chromosome = chromosome[:] # Copia para não modificar o original
    
    for i in range(len(mutated_chromosome)):
        if random.random() < rate:
            # Inverte o bit: 0 -> 1, 1 -> 0
            mutated_chromosome[i] = 1 - mutated_chromosome[i]
            
    return mutated_chromosome

# --- 3. Implementação Principal do Algoritmo Genético ---

def run_genetic_algorithm():
    """
    Executa o Algoritmo Genético completo.
    """
    
    print("Iniciando Algoritmo Genético...")
    print(f"Parâmetros: População={POP_SIZE}, Gerações={NUM_GENERATIONS}, Crossover={CROSSOVER_RATE*100}%, Mutação={MUTATION_RATE*100}%")
    
    # 3.1. População Inicial
    population = initialize_population(POP_SIZE, CHROMOSOME_LENGTH)
    
    best_individual = None
    best_fitness_overall = -float('inf')

    for generation in range(NUM_GENERATIONS):
        
        # 3.2. Avaliação de Fitness
        fitnesses = []
        decoded_values = []
        for individual in population:
            x = decode_chromosome(individual)
            decoded_values.append(x)
            f = fitness_function(x)
            fitnesses.append(f)
        
        # 3.3. Acompanhamento do Melhor Indivíduo da Geração
        current_best_index = np.argmax(fitnesses)
        current_best_fitness = fitnesses[current_best_index]
        current_best_individual = population[current_best_index]
        current_best_x = decoded_values[current_best_index]

        # Atualiza o melhor global
        if current_best_fitness > best_fitness_overall:
            best_fitness_overall = current_best_fitness
            best_individual = current_best_individual
            
        print(f"\n--- Geração {generation + 1} ---")
        print(f"Melhor x: {current_best_x:.4f} | Fitness: {current_best_fitness:.4f}")
        
        # Verifica se é a última geração
        if generation + 1 == NUM_GENERATIONS:
            break

        # 3.4. Criação da Próxima População (Elitismo Simples + Seleção/Crossover/Mutação)
        
        new_population = []
        
        # Elitismo: Mantém o melhor indivíduo da geração anterior
        new_population.append(current_best_individual[:])

        # Preenche o restante da nova população
        while len(new_population) < POP_SIZE:
            
            # 3.5. Seleção (Torneio)
            parent1 = tournament_selection(population, fitnesses, TORNEIO_SIZE)
            parent2 = tournament_selection(population, fitnesses, TORNEIO_SIZE)

            # 3.6. Crossover
            child1, child2 = one_point_crossover(parent1, parent2, CROSSOVER_RATE)
            
            # 3.7. Mutação
            mutated_child1 = mutation(child1, MUTATION_RATE)
            mutated_child2 = mutation(child2, MUTATION_RATE)
            
            # Adiciona à nova população, garantindo o tamanho correto
            new_population.append(mutated_child1)
            if len(new_population) < POP_SIZE:
                new_population.append(mutated_child2)
        
        # A nova população se torna a população atual para a próxima iteração
        population = new_population

    # --- Resultados Finais ---
    best_x_overall = decode_chromosome(best_individual)
    
    print("\n==================================")
    print("Resultados Finais do Algoritmo Genético")
    print(f"Melhor x encontrado: {best_x_overall:.6f}")
    print(f"Valor máximo da função f(x): {best_fitness_overall:.6f}")
    print(f"Cromossomo do melhor indivíduo: {best_individual}")
    print("==================================")

# --- Execução ---
if __name__ == '__main__':    
    run_genetic_algorithm()