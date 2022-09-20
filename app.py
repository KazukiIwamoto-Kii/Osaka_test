from flask import Flask, render_template, request, flash, get_flashed_messages, redirect, session, url_for
import random
import copy
#from stringprep import in_table_d1
import numpy as np
from statistics import mean
import math


app = Flask(__name__)


#観光地選択部分
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
    [90, 120, 90, 70, 80, 65, 80, 100, 75, 75, 75, 100, 75, 70, 80, 90, 85, 80, 100, 100, 110, 100, 130, 100, 105, 80, 100, 160, 135, 0]
    ])


MAX_TIME = 600
MAX_WEIGHT = MAX_TIME * 0.7  # 制限時間
N = 50        # 個体数
GENERATION = 50 # 世代数

#選択された観光地を表示
#cal(ITEMS) = [[0, 1, 3, 5, 6, 7], [0, 1, 3, 5, 6]]
def cal(ITEMS):
    sim = Simulation(ITEMS, MAX_WEIGHT, N) 
    for _ in range(GENERATION):
        sim.solve()
    return sim.print_population()



def item(name1, name2, name3, name4, name5, name6, name7, name8, name9):
    # priority = [[50, 20, 40, 30, 10, 50, 60, 20, 80],
    #             [50, 20, 40, 50, 10, 90, 60, 20, 70],
    #             [30, 30, 40, 40, 10, 20, 40, 20, 60],
    #             [40, 40, 40, 20, 60, 10, 60, 10, 30],
    #             [40, 50, 40, 10, 70, 50, 30, 10, 50],
    #             [10, 40, 50, 80, 30, 40, 10, 10, 30],
    #             [20, 10, 80, 60, 20, 70, 60, 30, 40],
    #             [60, 20, 90, 40, 10, 60, 20, 60, 30],
    #             [70, 30, 10, 20, 70, 60, 10, 80, 20]]
    
    if name1 == "1":
        a1 = 1
    else:
        a1 = 0
    if name2 == "2":
        a2 = 1
    else:
        a2 = 0
    if name3 == "3":
        a3 = 1
    else:
        a3 = 0
    if name4 == "4":
        a4 = 1
    else:
        a4 = 0
    if name5 == "5":
        a5 = 1
    else:
        a5 = 0
    if name6 == "6":
        a6 = 1
    else:
        a6 = 0
    if name7 == "7":
        a7 = 1
    else:
        a7 = 0
    if name8 == "8":
        a8 = 1
    else:
        a8 = 0
    if name9 == "9":
        a9 = 1
    else:
        a9 = 0
    
    selection = []
    selection.append(a1)
    selection.append(a2)
    selection.append(a3)
    selection.append(a4)
    selection.append(a5)
    selection.append(a6)
    selection.append(a7)
    selection.append(a8)
    selection.append(a9)

    a = 60
    c = 20

    
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
    
    return ITEMS


spot_list = { 0 : "海遊館", 1 : "万博公園", 2 : "USJ", 3 : "なんば", 4 : "梅田",
                5 : "天王寺", 6 : "長居公園", 7 : "大阪城", 8 : "新世界", 9 : "天王寺動物園",
                10 : "スパワールド", 11 : "梅田スカイビル", 12 : "なんばグランド花月", 13 : "なんばパークス", 14 : "アメリカ村",
                15 : "大阪天満宮", 16 : "四天王寺", 17 : "住吉大社", 18 : "国立国際美術館", 19 : "大阪市立美術館",
                20 : "関西国際空港", 21 : "りんくうアウトレット", 22 : "泉南りんくう公園", 23 : "中之島", 24 : "造幣美術館",
                25 : "空庭温泉", 26 : "生野コリアンタウン", 27 : "エキスポシティ", 28 : "ひらかたパーク", 29 : "ハーベストの丘"}


@app.route('/', methods = ['GET', 'POST'])#メイン画面
def home():
    return render_template('index.html')




@app.route('/result', methods = ['GET', 'POST'])#結果画面
def result():
    reload = 0
    trial = 0
    appear = [] #表示用
    while reload == 0:
        trial += 1
        if request.method == 'POST':
            # 入力情報の取得
            name1 = request.form.get('name1')
            name2 = request.form.get('name2')
            name3 = request.form.get('name3')
            name4 = request.form.get('name4')
            name5 = request.form.get('name5')
            name6 = request.form.get('name6')
            name7 = request.form.get('name7')
            name8 = request.form.get('name8')
            name9 = request.form.get('name9')

            N = 30 #観光地候補数
            ITEMS = item(name1, name2, name3, name4, name5, name6, name7, name8, name9)
            pop = cal(ITEMS) #選択された観光地を表示[[0, 1, 3, 5, 6, 7], [0, 1, 3, 5, 6]]
            candidateCount = len(pop) #ルートの候補数(例：2)
            reload = candidateCount #判定用の変数(これが0になったらもう一度)
            No = [] #各ルートの訪問観光地数
          
            #first_order = []
            opt_order = []
            best_order = []
            solution = []
            move_time = []
            total_move_time = []
            required_time = []
            total_time = []
            
            visit_spot = []
            
            for k in range(candidateCount):
                No.append(len(pop[k])) #それぞれの候補地の数(上の例だと[6, 5])

                x = []
                new_distance_matrix = []
                for i in pop[k]:
                    for j in pop[k]:
                        x.append(distance_matrix[i,j])
                y = np.array(x) #訪問予定の距離(1 × n^2)
                new_distance_matrix = y.reshape(No[k],No[k]) #距離行列(No × No)

                #初期解をランダムに生成
                first_order = list(np.random.permutation(No[k]))

                #2-opt法の適応
                #最適訪問順序
                opt_order.append(local_search(first_order, new_distance_matrix, improve_with_2opt))

                #最適総移動時間
                total_move_time.append(calculate_total_distance(opt_order[k], new_distance_matrix))

                best_order = []
                #正式な訪問順序
                for m in opt_order[k]:
                    best_order.append(pop[k][m])
                solution.append(best_order)

                each_required_time = []
                #各観光地での所要時間
                for l in best_order:
                    each_required_time.append(ITEMS[l][0])
                required_time.append(each_required_time)
                
                each_move_time= []
                #各観光地間の移動時間
                for n in range(No[k]-1):
                    each_move_time.append(new_distance_matrix[opt_order[k][n]][opt_order[k][n+1]])
                move_time.append(each_move_time)
                
                # 近傍探索適用後の総移動時間(戻ってこない)
                total_time.append(total_move_time[k] - new_distance_matrix[opt_order[k][0]][opt_order[k][No[k]-1]] + sum(each_required_time))
            
         
                spot_name = []
                for l in range(len(pop[k])):
                    spot_name.append(spot_list[solution[k][l]])
                visit_spot.append(spot_name)
                priority = 0
                for l in best_order:
                    priority += ITEMS[l][1]
                appear.append([best_order, priority])
      
                if total_time[k] >= MAX_TIME: #制限時間超えたら0にする
                    pop[k] = 0 
                    opt_order[k] = 0
                    total_move_time[k] = 0
                    total_time[k] = 0
                    solution[k] = 0
                    visit_spot[k] = 0
                    appear[k][1] = 0
                    reload -= 1
        start_time = 600
        start_hour = math.floor(start_time / 60)
        start_min = int(start_time % 60)
        start_minutes = f'{start_min:02}'

        appear.sort(key = lambda x: x[1], reverse=True)  #優先度でソート
            
    return render_template('result.html', trial = trial, pop = pop, opt_order = opt_order, solution = solution, total_move_time = total_move_time, total_time = total_time, 
                                         visit_spot = visit_spot, appear = appear, start_hour = start_hour, start_minutes = start_minutes)
        

if __name__ == "__main__":
    app.run(debug = True)