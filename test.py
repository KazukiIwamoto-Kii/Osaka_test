#from queue import PriorityQueue
import random
import copy
#from stringprep import in_table_d1
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
        # mean_list.append(self.mean_fitness())
        # max_list.append(self.max_fitness())
        
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
"""こっからセールスマン問題"""
#経路決定部分

#総移動距離の計算モジュール
def calculate_total_distance(order, distance_matrix):
    """Calculate total distance traveled for given visit order"""
    idx_from = np.array(order)
    idx_to = np.array(order[1:] + [order[0]])
    distance_arr = distance_matrix[idx_from, idx_to]

    return np.sum(distance_arr)

#近傍探索前後の総移動時間の差を算出
def calculate_2opt_exchange_cost(visit_order, i, j, distance_matrix):
    """Calculate the difference of cost by applying given 2-opt exchange"""
    n_cities = len(visit_order)
    a, b = visit_order[i], visit_order[(i + 1) % n_cities]
    c, d = visit_order[j], visit_order[(j + 1) % n_cities]

    cost_before = distance_matrix[a, b] + distance_matrix[c, d]
    cost_after = distance_matrix[a, c] + distance_matrix[b, d]
    return cost_after - cost_before

#訪問順序の更新
def apply_2opt_exchange(visit_order, i, j):
    """Apply 2-opt exhanging on visit order"""

    tmp = visit_order[i + 1: j + 1]
    tmp.reverse()
    visit_order[i + 1: j + 1] = tmp

    return visit_order

#2opt法による近傍探索
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

#近傍探索
def local_search(visit_order, distance_matrix, improve_func):
    """Main procedure of local search"""
    cost_total = calculate_total_distance(visit_order, distance_matrix)

    while True:
        improved = improve_func(visit_order, distance_matrix)
        if not improved:
            break

        visit_order = improved

    return visit_order
'''8/28追加 ココカラ'''
#30×30
distance_matrix = np.array([
    [ 0, 80, 40, 35, 35, 40, 60, 40, 40, 40, 40, 45, 50, 50, 40, 40, 40, 50, 40, 40, 90, 90, 90, 40, 50, 20, 50, 80, 70, 90],
    [80,  0, 70, 50, 50, 60, 70, 70, 60, 60, 60, 50, 60, 60, 60, 50, 70, 80, 70, 70, 120, 120, 120, 70, 60, 70, 70, 5, 70, 120],
    [40, 70,  0, 30, 20, 30, 40, 40, 20, 30, 20, 30, 30, 30, 30, 30, 30, 40, 30, 30, 70, 70, 80, 30, 40, 20, 40, 70, 70, 90],
    [35, 50, 30,  0, 10, 10, 20, 30, 20, 10, 10, 20,  5,  5, 10, 20, 20, 30, 20, 20, 50, 60, 60, 20, 30, 30, 30, 60, 50, 70],
    [35, 50, 20, 10,  0, 20, 30, 30, 20, 30, 25, 10, 15, 15, 15, 10, 20, 30, 15, 15, 60, 75, 75, 15, 25, 15, 35, 45, 45, 80],
    [40, 60, 30, 10, 20,  0, 10, 25, 10, 15, 10, 30, 15, 15, 15, 15, 10, 20, 25, 25, 50, 60, 60, 25, 30, 15, 20, 60, 60, 65],
    [60, 70, 40, 20, 30, 10,  0, 40, 20, 20, 35, 55, 40, 40, 40, 50, 35, 30, 50, 50, 80, 75, 100, 50, 60, 45, 45, 90, 75, 80],
    [40, 70, 40, 30, 30, 25, 40,  0, 50, 40, 45, 45, 40, 50, 45, 30, 45, 60, 45, 45, 100, 90, 105, 30, 40, 40, 40, 90, 60, 100],
    [40, 60, 20, 20, 20, 10, 20, 50,  0,  5,  5, 40, 25, 25, 25, 30, 30, 25, 35, 35, 60, 60, 70, 25, 35, 25, 35, 75, 60, 75],
    [40, 60, 30, 10, 30, 15, 20, 40,  5,  0,  5, 40, 25, 20, 25, 25, 25, 25, 35, 35, 60, 60, 75, 25, 35, 20, 30, 75, 60, 75],
    [40, 60, 20, 10, 25, 10, 35, 45,  5,  5,  0, 40, 25, 25, 25, 30, 30, 25, 35, 35, 60, 60, 70, 25, 35, 25, 35, 75, 60, 75],
    [45, 50, 30, 20, 10, 30, 55, 45, 40, 40, 40,  0, 30, 40, 30, 30, 45, 50, 30, 30, 90, 85, 95, 30, 35, 25, 45, 65, 60, 100],
    [50, 60, 30,  5, 15, 15, 40, 40, 25, 25, 25, 30,  0,  5, 15, 20, 40, 25, 30, 35, 65, 60, 65, 20, 30, 30, 30, 65, 55, 75],
    [50, 60, 30,  5, 15, 15, 40, 50, 25, 20, 25, 40,  5,  0, 15, 35, 35, 30, 30, 35, 60, 65, 70, 25, 40, 30, 35, 75, 60, 70],
    [40, 60, 30, 10, 15, 15, 40, 45, 25, 25, 25, 30, 15, 15,  0, 25, 35, 35, 30, 30, 65, 65, 80, 25, 35, 35, 35, 70, 55, 80],
    [40, 50, 30, 20, 10, 15, 50, 30, 30, 25, 30, 30, 20, 35, 25,  0, 30, 35, 25, 25, 75, 70, 85, 15, 20, 30, 40, 60, 45, 90],
    [40, 70, 30, 20, 20, 10, 35, 45, 30, 25, 30, 45, 40, 35, 35, 30,  0, 35, 45, 45, 70, 70, 85, 35, 40, 30, 30, 80, 60, 85],
    [50, 80, 40, 30, 30, 20, 30, 60, 25, 25, 25, 50, 25, 30, 35, 35, 35,  0, 50, 50, 60, 60, 65, 40, 50, 40, 50, 90, 75, 80],
    [40, 70, 30, 20, 15, 25, 50, 45, 35, 35, 35, 30, 30, 30, 30, 25, 45, 50,  0,  5, 90, 90, 95, 35, 35, 25, 50, 65, 50, 100],
    [40, 70, 30, 20, 15, 25, 50, 45, 35, 35, 35, 30, 35, 35, 30, 25, 45, 50,  5,  0, 90, 90, 95, 35, 35, 25, 50, 65, 50, 100],
    [90, 120, 70, 50, 60, 50, 80, 100, 60, 60, 60, 90, 65, 60, 65, 75, 70, 60, 90, 90, 0, 20, 50, 85, 100, 70, 85, 125, 120, 110],
    [90, 120, 70, 60, 75, 60, 75, 90, 60, 60, 60, 85, 60, 65, 65, 70, 70, 60, 90, 90, 20, 0, 35, 80, 95, 80, 90, 125, 120, 100],
    [90, 120, 80, 60, 75, 60, 100, 105, 70, 75, 70, 95, 65, 70, 80, 85, 85, 65, 95, 95, 50, 35, 0, 90, 120, 80, 95, 140, 130, 130],
    [40, 70, 30, 20, 15, 25, 50, 30, 25, 25, 25, 30, 20, 25, 25, 15, 35, 40, 35, 35, 85, 80, 90,  0, 30, 30, 50, 70, 40, 100],
    [50, 60, 40, 30, 25, 30, 60, 40, 35, 35, 35, 35, 30, 40, 35, 20, 40, 50, 35, 35, 100, 95, 120, 30, 0, 40, 40, 70, 45, 105],
    [20, 70, 20, 30, 15, 15, 45, 40, 25, 20, 25, 25, 30, 30, 35, 30, 30, 40, 25, 25, 70, 80, 80, 30, 40,  0, 40, 65, 60, 80],
    [50, 70, 40, 30, 35, 20, 45, 40, 35, 30, 35, 45, 30, 35, 35, 40, 30, 50, 50, 50, 85, 90, 95, 50, 40, 40,  0, 80, 60, 100],
    [80,  5, 70, 60, 45, 60, 90, 90, 75, 75, 75, 65, 65, 75, 70, 60, 80, 90, 65, 65, 125, 125, 140, 70, 70, 65, 80, 0, 90, 160],
    [70, 70, 70, 50, 45, 60, 75, 60, 60, 60, 60, 60, 55, 60, 55, 45, 60, 75, 50, 50, 120, 120, 130, 40, 45, 60, 60, 90, 0, 135],
    [90, 120, 90, 70, 80, 65, 80, 100, 75, 25, 75, 100, 75, 70, 80, 90, 85, 80, 100, 100, 110, 100, 130, 100, 105, 80, 100, 160, 135, 0]
    ])
MAX_TIME = 400
MAX_WEIGHT = MAX_TIME * 1.0  # 制限時間
N = 50        # 個体数
GENERATION = 20 # 世代数

#選択された観光地を表示
#cal(ITEMS) = [[0, 1, 3, 5, 6, 7], [0, 1, 3, 5, 6]]



selection =  [1, 0, 0, 1, 0, 0, 0, 0, 1] #selection(今は適当)
a = 60
c = 20
#d = []
#for i in selection:
#  d.append(i* 0.1)


p = [4, 3, 2, 9, 5, 8, 1, 1, 7, 5, 3, 4, 2, 1, 5, 8, 9, 2, 1, 3, 6, 8, 3, 4, 5, 2, 4, 6, 7, 8] #priority(人気度)
mutualRelationship = [] #相互関係行列(横に長い1行)
for i in range(9):
  for j in range(30):
    if distance_matrix[i,j] != 0:
        mutualRelationship.append(a/distance_matrix[i,j] + c*p[j] + selection[i])
    else:
        mutualRelationship.append(a/5.0 + c*p[j] + selection[i])
mutualRelationship = np.array(mutualRelationship)
mutualRelationship_reshape = np.round(mutualRelationship.reshape(9,30)) #相互関係行列

priority_matrix = np.dot(selection, mutualRelationship_reshape) #優先度

requiredTime = [180, 120, 480, 60, 120, 60, 60, 90, 30, 150, 360, 60, 150, 60, 60, 30, 90, 30, 60, 90, 30, 240, 150, 30, 90, 240, 60, 90, 300, 180] #想定される各地点での所要時間
ITEMS = []
for i in range(30):
  ITEMS.append((requiredTime[i], priority_matrix[i]))
sim = Simulation(ITEMS, MAX_WEIGHT, N)

for _ in range(GENERATION):
    #print('generation:', i)
    sim.solve()
print('最終結果')
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

'''[1 0 1 0 1 1 1 0]の場合
 [0.0 2.2 1.5 0.7 1.0]
 [2.2 0.0 0.8 0.8 0.4]
 [1.5 0.8 0.0 0.4 0.7]
 [0.7 0.8 0.4 0.0 0.4]
 [1.0 0.4 0.7 0.4 0.0] としたい'''

print('ITEMS:', ITEMS)

#print(distance_matrix)

#print(sim.print_population()[0]) #最終結果のインデックスのリストの0番目を表示(上の例だったら[0,1,5,9])
pop = sim.print_population() #最終結果のインデックスをリストとして表示([[0,1,5,9], [0,2,5,8]]的な)
candidateCount = len(pop) #地点の候補(1のときもあるしたくさんあるときもある、上の例だと2)
NO = []
def reload_function(candidateCount):
    #pop = cal(ITEMS)
    reload = 0
    appear = [] #表示用
    trial = 1
    while reload == 0:
        print('最終結果' + str(trial) +'回目')
        for k in range(candidateCount):
            NO.append(len(sim.print_population()[k])) #それぞれの候補地の数(上の例だと[4, 4])
            #print(NO)

            x = [] 
            new_distance_matrix = []
            for i in pop[k]:
                for j in pop[k]:
                    x.append(distance_matrix[i,j])
            y = np.array(x)
            new_distance_matrix = y.reshape(NO[k],NO[k]) #新たに行列を生成

            #print(z)
            #試しに距離を計算してみる
            first_order = list(np.random.permutation(NO[k]))
            #print('訪問順序 = {}'.format(test_order))

            #total = calculate_total_distance(test_order, new_distance_matrix)
         
            #2-opt法の適応
            #最適訪問順序
            opt_order = local_search(first_order, new_distance_matrix, improve_with_2opt) #訪問順序を返す

            #最適総移動時間
            total_move_time = calculate_total_distance(opt_order, new_distance_matrix)
            #print('訪問順序 = {}'.format(improved))
            best_order = []
            for m in opt_order:
                best_order.append(pop[k][m])
            print('正式な訪問地 = {}'.format(best_order))
            required_time = [] #各観光地での所要時間
            for l in best_order:
                print(str(l) + 'での活動目安時間 : ' + str(ITEMS[l][0]))
                required_time.append(ITEMS[l][0])
            for n in range(NO[k]-1):
                print(str(best_order[n]) + "から" + str(best_order[n+1]) + "の移動時間 : " +str(new_distance_matrix[opt_order[n]][opt_order[n+1]]))

            #print('近傍探索適用後の総移動時間 = {}'.format(total_distance))
            total_time = total_move_time - new_distance_matrix[opt_order[0]][opt_order[NO[k]-1]]
            print('近傍探索適用後の総移動時間(戻ってこない) = {}'.format(total_time))
        
            '''8/28追加 ココカラ'''
            if MAX_TIME >= total_time + sum(required_time):
                print('すべて合わせた時間 = {}'.format(total_time + sum(required_time)))
                priority = 0
                for l in best_order:
                    priority += ITEMS[l][1]
                print('優先度 = {}'.format(priority))
                appear.append((best_order, priority))
                reload += 1
            else:
                print('すべて合わせた時間 = {}'.format(total_time + sum(required_time)))
                print('ボツ')
            print('               ')

            '''8/28追加 ココマデ'''
        print(reload) #これが0なら再実行
        print(appear) #正式な訪問順序と優先度が入った配列
        appear.sort(key = lambda x: x[1], reverse=True)  #優先度でソート
        print(appear) #ソートした結果
        #print(appear[0])
        print('最終結果' + str(trial) +'回目')
        trial +=1 
    return reload

'''9/15追加 ココカラ'''

reload_function(candidateCount)


'''9/15追加 ココマデ'''