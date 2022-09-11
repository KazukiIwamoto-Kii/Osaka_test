import random
import copy
from stringprep import in_table_d1
import numpy as np
from statistics import mean
#import matplotlib.pyplot as plt

class Agent:
    # 初期化
    def __init__(self, M):
        # 遺伝子の初期化
        """8/29追加"""
        #self.gene = [random.randint(0,1) for i in range(M)]
        # 0,1の目が出るサイコロ(dice)を用意
        dice = list(range(0,2))
        # 0が出やすいように重みを設定する(1が多いと全部時間オーバーになって適応度0でゼロ除算発生)
        w = [5, 1]
        # 歪んだサイコロを30回振ってサンプルを得る
        samples = random.choices(dice, k = M, weights = w)
        """ここまで追加"""
        self.gene = samples
        self.weight = 0   # 時間
        self.fitness = 0  # 適応度（価値の合計）
    
    # 適応度計算
    def calc_fitness(self, items, MAX_WEIGHT):
        self.weight = sum([n*item[0] for n, item in zip(self.gene, items)])
        self.fitness = sum([n*item[1] for n, item in zip(self.gene, items)])
        if self.weight > MAX_WEIGHT: # 制限時間を超えたら適応度0
            self.fitness = 0        


class Simulation:
    # 初期化
    def __init__(self, items, max_weight, N):
        self.items = items           # アイテムをタプルのリストとして受け取る
        self.max_weight = max_weight # 制限時間
        self.N = N                   # 個体数
        self.mutation_rate = 0.1     # 突然変異率
        
        # 個体群の生成
        self.population = [Agent(len(self.items)) for i in range(self.N)]
        # 次世代の個体群
        self.offspring = []
        
    # ルーレット選択
    def roulette_selection(self):
        roulette = []
        for i in self.population:
            roulette.append(i.fitness)
        probs = np.array(roulette)/sum(roulette)
        #print("p:" +str(sum(roulette)))
       
        parent = []
        for i in range(2):
            selected = np.random.choice(self.population, p=probs)
            parent.append(copy.deepcopy(selected))
    
        return parent
    
    # 一点交叉
    def crossover(self, parent1, parent2):
        r = random.randint(0, len(parent1.gene)-1)
        offspring1 = copy.deepcopy(parent1)
        offspring2 = copy.deepcopy(parent2)
        offspring1.gene[r:] = parent2.gene[r:]
        offspring2.gene[r:] = parent1.gene[r:]

        # 次世代プールに追加
        self.offspring.append(offspring1)
        self.offspring.append(offspring2)
    
    # 突然変異
    def mutate(self, agent):
        r = random.choice(range(len(agent.gene)))
        agent.gene[r] = 1 if agent.gene[r] == 0 else 0
        
    # 世代交代    
    def generation_change(self):
        self.population.clear()
        self.population = copy.deepcopy(self.offspring)
        self.offspring.clear()

    # ナップサック問題を解く（=GAで個体群を進化させる）
    def solve(self):
        # 平均適応度，最大適応度をリストに追加
        mean_list.append(self.mean_fitness())
        max_list.append(self.max_fitness())
        
        # 適応度計算
        for agent in self.population:
            agent.calc_fitness(self.items, self.max_weight)
            
        # 個体群の情報出力
        self.print_population()

        # 選択，交叉
        while len(self.offspring) < len(self.population):
            parent = self.roulette_selection()
            self.crossover(parent[0], parent[1])

        # 突然変異
        for agent in self.population:
            if random.random() < self.mutation_rate:
                self.mutate(agent)
        # 世代交代    
        self.generation_change()
    
    # 個体群の情報出力
    def print_population(self):
        ind1 = []
        for i in self.population:
            #print(i.gene, i.fitness)
            ind = [i for i, x in enumerate(i.gene) if x == 1]
            ind1.append(ind)
            arr = list(map(list, set(map(tuple, ind1)))) #最終結果のインデックスをリストとして表示([[0,1,5,9], [0,2,5,8]]的な)
            #ind2 = set(ind1)
        return arr
    
    # 集団の平均適応度
    def mean_fitness(self):
        fitness = []
        for i in self.population:
            fitness.append(i.fitness)
            
        return mean(fitness)
            
    # 集団の最大適応度
    def max_fitness(self):
        fitness = []
        for i in self.population:
            fitness.append(i.fitness)
        
        return max(fitness)

'''8/28追加 ココカラ'''
#30×30
distance_matrix = np.array([[0.0, 2.0, 2.2, 1.8, 1.5, 0.7, 1.0, 0.2, 3.0, 2.5, 1.6, 1.2, 1.0, 2.5, 2.5, 2.1, 3.3, 0.4, 0.2, 1.5, 2.1, 4.2, 2.5, 1.9, 2.6, 3.2, 1.2, 2.0, 2.4, 1.0], #経路の時間
                            [2.0, 0.0, 0.5, 0.6, 0.4, 0.8, 0.5, 1.5, 1.1, 1.6, 2.1, 1.1, 1.9, 0.5, 1.9, 1.3, 0.3, 0.9, 0.4, 3.5, 1.1, 0.5, 2.1, 1.0, 2.1, 1.2, 3.6, 2.2, 0.4, 2.0],
                            [2.2, 0.5, 0.0, 0.2, 0.8, 0.8, 0.4, 1.7, 1.9, 1.3, 1.1, 2.5, 0.9, 0.3, 1.2, 0.4, 0.9, 2.9, 2.4, 2.1, 2.4, 3.5, 1.1, 1.6, 2.1, 1.7, 2.6, 0.6, 1.8, 2.6],
                            [1.8, 0.6, 0.2, 0.0, 1.0, 1.0, 0.5, 1.2, 1.3, 1.6, 2.6, 1.5, 3.5, 1.3, 2.4, 2.4, 1.3, 1.4, 1.9, 2.4, 2.2, 1.5, 2.1, 0.6, 1.1, 1.3, 1.6, 0.8, 1.2, 1.6],
                            [1.5, 0.4, 0.8, 1.0, 0.0, 0.4, 0.7, 0.9, 1.0, 1.8, 1.2, 1.8, 1.2, 2.6, 1.9, 1.4, 0.3, 2.3, 0.9, 1.4, 2.2, 1.7, 1.1, 4.6, 1.4, 0.3, 0.6, 0.8, 0.2, 2.6],
                            [0.7, 0.8, 0.8, 1.0, 0.4, 0.0, 0.4, 0.3, 0.8, 1.4, 1.3, 1.0, 0.8, 0.5, 1.5, 1.1, 1.6, 2.3, 3.3, 0.4, 0.2, 1.5, 0.4, 0.8, 1.9, 0.6, 1.2, 0.3, 0.1, 2.2],
                            [1.0, 0.5, 0.4, 0.5, 0.7, 0.4, 0.0, 0.6, 0.9, 1.4, 2.0, 2.2, 1.8, 1.0, 2.5, 0.9, 0.4, 3.5, 1.9, 4.2, 2.5, 1.9, 2.2, 0.4, 2.0, 0.8, 0.5, 1.5, 3.2, 0.6],
                            [0.2, 1.5, 1.7, 1.2, 0.9, 0.3, 0.6, 0.0, 2.3, 1.5, 1.0, 1.2, 0.8, 2.4, 0.3, 1.3, 0.5, 1.1, 2.5, 0.2, 3.0, 2.5, 0.6, 0.4, 0.8, 0.5, 0.4, 2.5, 1.6, 1.3],
                            [3.0, 1.1, 1.9, 1.3, 1.0, 0.8, 0.9, 2.3, 0.0, 0.8, 2.5, 1.6, 1.3, 2.5, 1.5, 1.8, 0.2, 3.0, 1.4, 0.4, 4.2, 2.0, 2.2, 0.6, 1.2, 2.4, 1.9, 1.2, 0.6, 1.9],
                            [2.5, 1.6, 1.3, 1.6, 1.8, 1.4, 1.4, 1.5, 0.8, 0.0, 1.2, 1.1, 1.2, 0.8, 1.8, 0.2, 0.4, 2.4, 3.2, 1.2, 1.7, 2.2, 1.8, 1.5, 1.1, 1.6, 2.1, 0.2, 1.5, 1.7],
                            [1.6, 2.1, 1.1, 2.6, 1.2, 1.3, 2.0, 1.0, 2.5, 1.2, 0.0, 1.5, 2.5, 0.3, 1.2, 2.4, 1.9, 0.8, 0.4, 1.7, 2.1, 1.3, 0.4, 1.9, 1.0, 1.6, 2.5, 1.2, 0.2, 3.2],
                            [1.2, 1.1, 2.5, 1.5, 1.8, 1.0, 2.2, 1.2, 1.6, 1.1, 1.5, 0.0, 2.1, 1.3, 0.4, 1.7, 1.2, 0.9, 1.6, 2.6, 1.5, 1.5, 0.4, 0.8, 2.1, 1.1, 2.4, 1.4, 2.0, 2.2],
                            [1.0, 1.9, 0.9, 3.5, 1.2, 0.8, 1.8, 0.8, 1.3, 1.2, 2.5, 2.1, 0.0, 0.8, 0.4, 0.9, 2.9, 1.4, 0.6, 4.6, 0.8, 1.8, 1.2, 0.2, 1.6, 1.2, 1.0, 1.3, 1.0, 0.8],
                            [2.5, 0.5, 0.3, 1.3, 2.6, 0.5, 1.0, 2.4, 2.5, 0.8, 0.3, 1.3, 0.8, 0.0, 0.5, 0.2, 0.4, 1.9, 1.3, 1.1, 2.5, 1.6, 1.3, 0.8, 0.4, 1.7, 1.8, 0.2, 2.4, 1.0],
                            [2.5, 1.9, 1.2, 2.4, 1.9, 1.5, 2.5, 0.3, 1.5, 1.8, 1.2, 0.4, 0.4, 0.5, 0.0, 1.2, 0.8, 1.8, 2.0, 1.0, 2.5, 1.0, 0.5, 0.4, 2.1, 1.1, 1.4, 1.3, 1.9, 1.7],
                            [2.1, 1.3, 0.4, 2.4, 1.4, 1.1, 0.9, 1.3, 1.8, 0.2, 2.4, 1.7, 0.9, 0.2, 1.2, 0.0, 0.5, 0.2, 0.4, 1.0, 2.2, 1.2, 0.8, 1.8, 0.2, 1.7, 1.1, 4.6, 1.3, 1.9],
                            [3.3, 0.3, 0.9, 1.3, 0.3, 1.6, 0.4, 0.5, 0.2, 0.4, 1.9, 1.2, 2.9, 0.4, 0.8, 0.5, 0.0, 1.4, 2.3, 1.9, 1.2, 2.4, 2.5, 1.6, 1.3, 1.8, 0.2, 2.5, 1.9, 1.2],
                            [0.4, 0.9, 2.9, 1.4, 2.3, 2.3, 3.5, 1.1, 3.0, 2.4, 0.8, 0.9, 1.4, 1.9, 1.8, 0.2, 1.4, 0.0, 2.5, 2.2, 1.5, 3.5, 1.0, 2.4, 1.4, 0.3, 0.5, 1.1, 2.5, 1.9],
                            [0.2, 0.4, 2.4, 1.9, 0.9, 3.3, 1.9, 2.5, 1.4, 3.2, 0.4, 1.6, 0.6, 1.3, 2.0, 0.4, 2.3, 2.5, 0.0, 2.5, 2.2, 1.5, 3.5, 1.0, 2.4, 1.4, 0.3, 0.5, 1.1, 2.5],
                            [1.5, 3.5, 2.1, 2.4, 1.4, 0.4, 4.2, 0.2, 0.4, 1.2, 1.7, 2.6, 4.6, 1.1, 1.0, 1.0, 1.9, 2.2, 2.5, 0.0, 0.4, 1.2, 1.3, 1.4, 1.1, 0.2, 1.3, 2.5, 1.2, 2.9],
                            [2.1, 1.1, 2.4, 2.2, 2.2, 0.2, 2.5, 3.0, 4.2, 1.7, 2.1, 1.5, 0.8, 2.5, 2.5, 2.2, 1.2, 1.5, 2.2, 0.4, 0.0, 1.0, 1.2, 1.8, 0.3, 1.5, 2.5, 1.3, 1.4, 1.1],                      
                            [4.2, 0.5, 3.5, 1.5, 1.7, 1.5, 1.9, 2.5, 2.0, 2.2, 1.3, 1.5, 1.8, 1.6, 1.0, 1.2, 2.4, 3.5, 1.5, 1.2, 1.0, 0.0, 1.4, 0.2, 2.8, 1.4, 2.2, 1.7, 0.5, 1.4],
                            [2.5, 2.1, 1.1, 2.1, 1.1, 0.4, 2.2, 0.6, 2.2, 1.8, 0.4, 0.4, 1.2, 1.3, 0.5, 0.8, 2.5, 1.0, 3.5, 1.3, 1.2, 1.4, 0.0, 3.4, 0.2, 4.8, 2.0, 1.2, 0.7, 0.5],
                            [1.9, 1.0, 1.6, 0.6, 4.6, 0.8, 0.4, 0.4, 0.6, 1.5, 1.9, 0.8, 0.2, 0.8, 0.4, 1.8, 1.6, 2.4, 1.0, 1.4, 1.8, 0.2, 3.4, 0.0, 0.6, 0.8, 1.8, 2.1, 0.2, 0.9],
                            [2.6, 2.1, 2.1, 1.1, 1.4, 1.9, 2.0, 0.8, 1.2, 1.1, 1.0, 2.1, 1.6, 0.4, 2.1, 0.2, 1.3, 1.4, 2.4, 1.1, 0.3, 2.8, 0.2, 0.6, 0.0, 1.2, 0.5, 0.3, 1.5, 0.5],
                            [3.2, 1.2, 1.7, 1.3, 0.3, 0.6, 0.8, 0.5, 2.4, 1.6, 1.6, 1.1, 1.2, 1.7, 1.1, 1.7, 1.8, 0.3, 1.4, 0.2, 1.5, 1.4, 4.8, 0.8, 1.2, 0.0, 2.4, 1.5, 0.4, 0.5],
                            [1.2, 3.6, 2.6, 1.6, 0.6, 1.2, 0.5, 0.4, 1.9, 2.1, 2.5, 2.4, 1.0, 1.8, 1.4, 1.1, 0.2, 0.5, 0.3, 1.3, 2.5, 2.2, 2.0, 1.8, 0.5, 2.4, 0.0, 1.6, 0.7, 0.2],
                            [2.0, 2.2, 0.6, 0.8, 0.8, 0.3, 1.5, 2.5, 1.2, 0.2, 1.2, 1.4, 1.3, 0.2, 1.3, 4.6, 2.5, 1.1, 0.5, 2.5, 1.3, 1.7, 1.2, 2.1, 0.3, 1.5, 1.6, 0.0, 1.4, 0.6],
                            [2.4, 0.4, 1.8, 1.2, 0.2, 0.1, 3.2, 1.6, 0.6, 1.5, 0.2, 2.0, 1.0, 2.4, 1.9, 1.3, 1.9, 2.5, 1.1, 1.2, 1.4, 0.5, 0.7, 0.2, 1.5, 0.4, 0.7, 1.4, 0.0, 2.1],
                            [1.0, 2.0, 2.6, 1.6, 2.6, 2.2, 0.6, 1.3, 1.9, 1.7, 3.2, 2.2, 0.8, 1.0, 1.7, 1.9, 1.2, 1.9, 2.5, 2.9, 1.1, 1.4, 0.5, 0.9, 0.5, 0.5, 0.2, 0.6, 2.1, 0.0]])

#print(distance_matrix)
a = 0.5
c = 0.2
sentaku =  [1, 0, 0, 1, 0, 0, 0, 0, 1] #selection
d = []
for i in sentaku:
  d.append(i* 0.1)
#print(d)
p = [0.4, 0.3, 0.2, 0.9, 0.5, 0.8, 0.1, 0.1, 0.7]
sougo_kankei = []
for i in range(9):
  for j in range(30):
    if distance_matrix[i,j] != 0:
      sougo_kankei.append(a/distance_matrix[i,j] + c*p[i] + d[i])
    else:
      sougo_kankei.append(a/1.0 + c*p[i] + d[i])
sougo_kankei2 = np.array(sougo_kankei)
sougo_kankei3 = np.round(sougo_kankei2.reshape(9,30)) #相互関係行列
#print(sougo_kankei3)
yuusendo = np.dot(sentaku, sougo_kankei3) #優先度
#print(yuusendo)
syoyouzikan = [1.5, 0.5, 0.5, 0.9, 6.0, 2.0, 1.0, 0.3, 4.0, 3.0, 2.5, 1.3, 4.0, 2.4, 3.6, 2.5, 0.3, 0.7, 3.9, 5.0, 2.0, 1.7, 1.3, 4.0, 3.0, 2.2, 2.3, 1.0, 5.4, 2.4]
ITEMS = []
for i in range(30):
  ITEMS.append((syoyouzikan[i], yuusendo[i]))
#print(ITEMS)

'''8/28追加 ココマデ'''
# テストデータ
#ITEMS = [(1.5, 50), (0.5, 20), (0.5, 10), (0.9, 40), (6.0, 100), (2.0,70), (1.0,30), (0.3,5), (4.0, 30), (3.0, 30), (1.5, 50), (0.5, 20), (0.5, 10), (0.9, 40), (3.0, 20), (1.5, 40), (0.9, 50), (1.4, 81)] # (時間，価値)価値は可変
'''t = [[0.0, 2.0, 2.2, 1.8, 1.5, 0.7, 1.0, 0.2], #経路の時間
     [2.0, 0.0, 0.5, 0.6, 0.4, 0.8, 0.5, 1.5],
     [2.2, 0.5, 0.0, 0.2, 0.8, 0.8, 0.4, 1.7],
     [1.8, 0.6, 0.2, 0.0, 1.0, 1.0, 0.5, 1.2],
     [1.5, 0.4, 0.8, 1.0, 0.0, 0.4, 0.7, 0.9],
     [0.7, 0.8, 0.8, 1.0, 0.4, 0.0, 0.4, 0.3],
     [1.0, 0.5, 0.4, 0.5, 0.7, 0.4, 0.0, 0.6],
     [0.2, 1.5, 1.7, 1.2, 0.9, 0.3, 0.6, 0.0]]
print(t)'''
# ランダムにデータ生成
#ITEMS = [(random.randint(0, 15), random.randint(0, 100)) for i in range (10)]
print('ITEMS:', ITEMS)
MAX_TIME = 10
MAX_WEIGHT = MAX_TIME - 3  # 制限時間
N = 100         # 個体数
GENERATION = 20 # 世代数

# グラフ用リスト
g_list = []
mean_list = []
max_list = []

sim = Simulation(ITEMS, MAX_WEIGHT, N)
for i in range(GENERATION):
    #print('generation:', i)
    g_list.append(i+1)
    sim.solve()
print('最終結果')
"""こっからセールスマン問題"""

'''[1 0 1 0 1 1 1 0]の場合
 [0.0 2.2 1.5 0.7 1.0]
 [2.2 0.0 0.8 0.8 0.4]
 [1.5 0.8 0.0 0.4 0.7]
 [0.7 0.8 0.4 0.0 0.4]
 [1.0 0.4 0.7 0.4 0.0] としたい'''
#distance_matrix = 
#print(distance_matrix)
def calculate_total_distance(order, distance_matrix):
    """Calculate total distance traveled for given visit order"""
    idx_from = np.array(order)
    idx_to = np.array(order[1:] + [order[0]])
    distance_arr = distance_matrix[idx_from, idx_to]

    return np.sum(distance_arr)




def calculate_2opt_exchange_cost(visit_order, i, j, distance_matrix):
    """Calculate the difference of cost by applying given 2-opt exchange"""
    n_cities = len(visit_order)
    a, b = visit_order[i], visit_order[(i + 1) % n_cities]
    c, d = visit_order[j], visit_order[(j + 1) % n_cities]

    cost_before = distance_matrix[a, b] + distance_matrix[c, d]
    cost_after = distance_matrix[a, c] + distance_matrix[b, d]
    return cost_after - cost_before

def apply_2opt_exchange(visit_order, i, j):
    """Apply 2-opt exhanging on visit order"""

    tmp = visit_order[i + 1: j + 1]
    tmp.reverse()
    visit_order[i + 1: j + 1] = tmp

    return visit_order

def improve_with_2opt(visit_order, distance_matrix):
    """Check all 2-opt neighbors and improve the visit order"""
    n_cities = len(visit_order)
    cost_diff_best = 0.0
    i_best, j_best = None, None

    for i in range(0, n_cities - 2):
        for j in range(i + 2, n_cities):
            if i == 0 and j == n_cities - 1:
                continue

            cost_diff = calculate_2opt_exchange_cost(
                visit_order, i, j, distance_matrix)

            if cost_diff < cost_diff_best:
                cost_diff_best = cost_diff
                i_best, j_best = i, j

    if cost_diff_best < 0.0:
        visit_order_new = apply_2opt_exchange(visit_order, i_best, j_best)
        return visit_order_new
    else:
        return None

def local_search(visit_order, distance_matrix, improve_func):
    """Main procedure of local search"""
    cost_total = calculate_total_distance(visit_order, distance_matrix)

    while True:
        improved = improve_func(visit_order, distance_matrix)
        if not improved:
            break

        visit_order = improved

    return visit_order

# 適当に初期解を生成
#test_order = list(np.random.permutation(N))
#visualize_visit_order(test_order, city_xy)
#total_distance = calculate_total_distance(test_order, distance_matrix)
#print('初期解の総移動時間 = {}'.format(total_distance))


print(sim.print_population()) #最終結果のインデックスをリストとして表示([[0,1,5,9], [0,2,5,8]]的な)
print(sim.print_population()[0]) #最終結果のインデックスのリストの0番目を表示(上の例だったら[0,1,5,9])
kouho = len(sim.print_population()) #地点の候補(1のときもあるしたくさんあるときもある、上の例だと2)
NO = []
for k in range(kouho):
    NO.append(len(sim.print_population()[k])) #それぞれの候補地の数(上の例だと[4, 4])
print(NO)
#MAP_SIZE = 100


for k in range(kouho):
    x = []
    z = []
    for i in sim.print_population()[k]:
        for j in sim.print_population()[k]:
            x.append(distance_matrix[i,j])
    y = np.array(x)
    z = y.reshape(NO[k],NO[k]) #新たに行列を生成

    #print(z)
#試しに距離を計算してみる
    test_order = list(np.random.permutation(NO[k]))
#print('訪問順序 = {}'.format(test_order))

    total = calculate_total_distance(test_order, z)
    #print('試しの総移動時間 = {}'.format(total))
# 近傍を計算
    improved = local_search(test_order, z, improve_with_2opt) #訪問順序を返す
#visualize_visit_order(improved, city_xy)
    total_distance = calculate_total_distance(improved, z)
    #print('訪問順序 = {}'.format(improved))
    seisiki = []
    for m in improved:
        seisiki.append(sim.print_population()[k][m])
    print('正式な訪問地 = {}'.format(seisiki))
    kai2 = []
    for l in seisiki:
        print(str(l) + 'での活動目安時間 : ' + str(ITEMS[l][0]))
        kai2.append(ITEMS[l][0])
    for n in range(NO[k]-1):
        print(str(seisiki[n]) + "から" + str(seisiki[n+1]) + "の移動時間 : " +str(z[improved[n]][improved[n+1]]))

    #print('近傍探索適用後の総移動時間 = {}'.format(total_distance))
    kai = total_distance - z[improved[0]][improved[NO[k]-1]]
    print('近傍探索適用後の総移動時間(戻ってこない) = {}'.format(kai))
    
    '''8/28追加 ココカラ'''
    if MAX_TIME >= kai + sum(kai2):
        print('すべて合わせた時間 = {}'.format(kai + sum(kai2)))
    else:
        print('ボツ')
    print('               ')

    '''8/28追加 ココマデ'''