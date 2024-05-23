import numpy as np
import random
import copy
import matplotlib.pyplot as plt
import time

# 指定字体
plt.rcParams['font.sans-serif'] = 'SimHei'  # 使用宋体字体
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号 有中文出现的情况，需要u'内容'


# 设置随机数种子
# random.seed(13)
# np.random.seed(0)

"""
    油车：
    平均1km，价格：0.5 ~ 0.8 元
    电动汽车：
    平均1km，耗电量：0.1 ~ 0.2 度（kWh），耗电量百分比：0.2% ~ 0.4%，价格：0.08 ~ 0.48 元
    平均1度电（2%），里程：5km ~ 10km，
    1%，里程：2.5km ~ 5km，价格：0.4 ~ 1.2 元
    5%，里程：12.5km ~ 25km
    平均总电量：50 度（kWh），作为百分百电量，公里数：250 ~ 500 km
    
    充电站每度电的价格为，0.8 ~ 2.4 元/度（kWh）    
"""


# 电动汽车类
class EV:
    def __init__(self, start_coordinate, current_energy, expected_energy, energy_consumption_pk):
        self.start_coordinate = start_coordinate  # 电动汽车的起始坐标
        self.current_energy = current_energy  # 电动汽车的当前电量，单位：%   5% ~ 75%
        self.expected_energy = expected_energy  # 电动汽车的预期电量，单位：%  80% ~ 100%
        self.energy_consumption_pk = energy_consumption_pk  # 电动汽车平均每公里消耗的百分比电量   0.2% ~ 0.4%


# 充电站类
class CS:
    def __init__(self, idx, coordinate, sell_price, buy_price):
        self.idx = idx                # 充电站的编号
        self.coordinate = coordinate  # 充电站的坐标
        self.sell_price = sell_price  # 向充电站出售电量的价格
        self.buy_price = buy_price    # 向充电站购买电量的价格
        self.path = []                # 电动汽车到达本充电站的路径
        self.cost = []                # 电动汽车到达本充电站的总成本


def draw_map(flag):  # 和10x10不一样
    """
    功能: 画出初始化地图和找到最优充电站的路线地图
    :param flag: int类型，0表示画出初始化地图，1表示画出找到最优充电站的路线地图。
    :return: None
    """
    # 地图网格的大小
    n = grid.shape[0]
    # 调整不同值对应的颜色
    colors = np.empty((n, n, 4), dtype='float')
    colors[grid == 0] = color_map[0]  # 0，白色，透明度0.6
    colors[grid == 1] = color_map[1]  # 1，黑色，透明度0.6
    colors[grid == 2] = color_map[2]  # 2，蓝色，透明度0.6
    colors[grid == 3] = color_map[3]  # 3，绿色，透明度0.6

    # 绘制栅格图
    plt.imshow(colors, interpolation='nearest')

    # 添加网格之间的线条
    for i in range(-1, n):
        plt.plot([-0.5, n - 0.5], [i + 0.5, i + 0.5], color='black')  # 行
        plt.plot([i + 0.5, i + 0.5], [-0.5, n - 0.5], color='black')  # 列

    # 将x轴放置在顶部
    plt.gca().xaxis.tick_top()

    # 隐藏坐标轴刻度
    plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)

    # 设置横坐标和纵坐标的显示范围和步长
    plt.xticks(np.arange(-0.5, n + 1, 2), map(int, np.arange(-0.5, n + 1, 2) + 0.5))  # 横坐标从0到20，步长为2
    plt.yticks(np.arange(-0.5, n + 1, 2), map(int, np.arange(-0.5, n + 1, 2) + 0.5))  # 纵坐标从0到20，步长为2

    # 写标题
    plt.text(n // 2 - 0.5, n + 1, 'EV使用蚁群算法寻找最优充电站(20x20)', ha='center', va='center', fontdict={'weight': 'bold', 'size': 14})

    # 标记x轴，y轴
    plt.text(n // 2 - 0.5, -2.7, 'x轴', ha='center', va='center', fontdict={'weight': 'bold', 'size': 12})
    plt.text(-2.7, n // 2 - 0.5, 'y轴', ha='center', va='center', fontdict={'weight': 'bold', 'size': 12})

    # 画出找到最优充电站的路线地图
    if flag:
        position_dis = [-0.10, -0.05, 0, 0.05, 0.05]
        for i, cs in enumerate(cs_set.values()):
            if len(cs.cost) == 0:
                continue
            optimal_path_idx = cs.cost.index(min(cs.cost))
            cost, path = cs.cost[optimal_path_idx], cs.path[optimal_path_idx]
            line_width = 3.5
            for start, end in zip(path[:len(path) - 1], path[1:]):
                plt.plot([start[0] + position_dis[i], end[0] + position_dis[i]],
                         [start[1] + position_dis[i], end[1] + position_dis[i]],
                         color=color_map2[i], linewidth=line_width)
        # 在找到的最优充电站的位置画一个三角形，表示最优充电站的位置，marker='^'，表示绘制成三角形，s=80，设置点的大小
        # tmp = round(aco.optimal_cost, 2)
        plt.scatter(aco.optimal_cs[0], aco.optimal_cs[1], marker='^', s=80, color='red', label='最优充电站', zorder=3)
        # 在电动汽车的起始位置画一个圆形，表示电动汽车的起始位置，marker='o'，表示绘制成圆形，s=80，设置点的大小
        plt.scatter(ev_coordinate[0], ev_coordinate[1], marker='o', s=80, color='red', label='EV', zorder=3)
        # 显示图例
        # plt.legend()

    plt.show()


def draw_data(flag):
    """
    功能: 画出每次迭代的成本曲线
    :param flag: int类型，0表示画出最优成本，1表示画出平均成本。
    :return: None
    """
    if flag:
        plt.plot(range(1, max_iter + 1), aco.all_average_costs, label='ACS')
        y_label = 'Average cost'
        title = '每一次迭代中的平均成本'
    else:
        plt.plot(range(1, max_iter + 1), aco.all_optimal_costs, label='ACS')
        y_label = 'Optimal cost'
        title = '每一次迭代中的最优成本'
    plt.xlabel('Iteration')
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()

    plt.show()


def init():  # 和10x10不一样
    """
    功能: 画出初始化地图，初始化电动汽车对象和充电站对象。
    :return: ev: 电动汽车对象；CS_set: 充电站对象的集合。
    """
    # 电动汽车 EV 的初始化
    # draw_map(0)  # 画出初始化地图
    current_energy = 15  # 电动汽车的当前电量  random.randint(5, 75)
    expected_energy = 100  # 电动汽车的预期电量 random.randint(80, 100)
    energy_consumption_pk = 0.4  # 电动汽车平均每公里消耗的百分比电量 random.randint(20, 40) / 100
    ev = EV(start_coordinate=ev_coordinate, current_energy=current_energy, expected_energy=expected_energy,
            energy_consumption_pk=energy_consumption_pk)

    # 充电站 CS 的初始化
    n = grid.shape[0]  # 地图尺寸
    cs_set = {}        # 充电站集合
    idx = 0            # 充电站的编号
    sell_price = [0.8, 1.0, 1.1, 0.5, 0.9]  # 向充电站出售电量的价格，buy_price - sell_price = 0.5
    buy_price = [1.5, 1.3, 1.6, 1.2, 1.4]   # 向充电站购买电量的价格

    for i in range(n):
        for j in range(n):
            if grid[i][j] == 3:
                cs = CS(idx, coordinate=(j, i), sell_price=sell_price[idx], buy_price=buy_price[idx])
                idx += 1
                cs_set[(j, i)] = cs

    return ev, cs_set


# 蚂蚁类
class Ant:
    def __init__(self, ev):
        # 参数设置
        self.ev = ev
        self.successful = False  # 标志蚂蚁是否成功抵达充电站

        self.position = ev.start_coordinate  # 蚂蚁当前的位置
        self.path = [self.position]          # 蚂蚁走过的路径
        self.visited_position = {self.position: True}  # 蚂蚁已经走过的位置
        self.total_distance = 0              # 蚂蚁走过的路径总长度
        self.cost = float('inf')             # 蚂蚁的总成本
        self.destination = None              # 目的地充电站

    # 蚂蚁构建路径
    def run(self, pheromones):
        # 不断找下一节点，直到找到充电站或者死路
        while True:
            result = self.select_next_position(pheromones)
            if not result or result and self.successful:
                break

    # 选择下一个位置
    def select_next_position(self, pheromones):
        """
        功能：选择下一个位置，仅返回一个状态码True/False标志选择的成功与否。
        传递参数时，要注意不可变对象（整数，浮点数，字符串）按值传递，可变对象（列表，字典）按引用传递.
        :param pheromones: 字典类型的数据，路径上的信息素浓度，例如，pheromones[((0,0),(0,1))] = 1, 表示(0,0)-->(0,1)路径上的信息素浓度为1。
        :return: bool类型，表示选择下一个位置的成功与否。
        """
        n = grid.shape[0]  # 地图大小 n*n
        x, y = self.position  # 蚂蚁当前的位置

        # Step 1:计算蚂蚁可以走的下一个位置
        selected_next_position = [(x, y - 1), (x, y + 1), (x - 1, y), (x + 1, y)]  # 上，下，左，右

        # Step 2:排除非法位置、障碍物以及已经访问过的位置
        tmp_selected_next_position = copy.deepcopy(selected_next_position)
        for position_x, position_y in tmp_selected_next_position:
            next_position = (position_x, position_y)
            # 坐标越界
            if position_x < 0 or position_y < 0 or position_x >= n or position_y >= n:
                selected_next_position.remove(next_position)  # 删除这些无法访问的位置
                continue

            position_type = grid[position_y][position_x]
            # 遇到障碍物，或者这个位置已经访问过
            if position_type == 1 or self.visited_position.get(next_position, False):
                selected_next_position.remove(next_position)  # 删除这些无法访问的位置
                continue

            # 遇到充电站
            if position_type == 3:
                cs = cs_set[next_position]                   # 当前充电站的信息
                self.successful = True                       # 成功找到充电站

                self.position = next_position                # 更新蚂蚁当前的位置
                self.path.append(next_position)              # 更新蚂蚁走过的路径
                self.visited_position[next_position] = True  # 标记该位置已访问
                self.total_distance += 1                     # 更新蚂蚁走过的路径总长度
                if self.ev.current_energy < self.ev.energy_consumption_pk:  # 当前电量不足以到达充电站
                    return False
                self.ev.current_energy -= self.ev.energy_consumption_pk  # 更新电动汽车的当前电量
                self.cost = (self.ev.expected_energy - self.ev.current_energy) / 100 * 50 * cs.buy_price  # 计算电动汽车的总成本
                self.destination = cs.coordinate
                return True

        # 如果没有合法位置，则直接结束蚂蚁的路径构建
        if len(selected_next_position) == 0:
            return False

        # Step 3:计算下一个位置与最近充电站之间的距离，构建距离启发因子
        distance = []  # 距离启发因子, 距离的倒数
        # 遍历所有可以走的下一个位置
        for next_position in selected_next_position:
            min_dis = float('inf')  # 到最近充电站的距离
            # 遍历每一个充电站，cs表示每个充电站的坐标
            for cs in cs_set.keys():
                tmp_dis = ((next_position[0] - cs[0]) ** 2 + (next_position[1] - cs[1]) ** 2) ** 0.5  # 计算与充电站的距离
                min_dis = min(min_dis, tmp_dis)  # 更新最短的距离
            distance.append(1 / min_dis)   # 将最短距离的倒数作为距离启发因子

        # Step 4:计算下一个位置被选中的概率
        probs = []
        total_prob = 0.0
        for i, next_position in enumerate(selected_next_position):
            # 状态转移概率公式，计算选择下一个位置的概率
            if pheromones.get((self.position, next_position), None) is None:
                pheromones[(self.position, next_position)] = phe_init
            p = ((pheromones[(self.position, next_position)] ** alpha) * (distance[i] ** beta))
            total_prob += p
            probs.append((next_position, p))

        # Step 5:轮盘赌，根据概率选择下一个位置
        next_position = -1
        temp_prob = random.uniform(0.0, total_prob)  # 生成一个随机浮点数: [0, total_prob)
        for position, p in probs:
            temp_prob -= p
            if temp_prob < 0.0:
                next_position = position
                break

        if next_position == -1:
            print("程序出问题！！！")
            exit(0)

        # Step 6:位置更新操作
        self.position = next_position                # 更新蚂蚁当前的位置
        self.path.append(next_position)              # 更新蚂蚁走过的路径
        self.visited_position[next_position] = True  # 标记该位置已访问
        self.total_distance += 1                     # 更新蚂蚁走过的路径总长度
        if grid[self.position[1]][self.position[0]] == 2:
            self.ev.current_energy += 0.5 * self.ev.energy_consumption_pk
        if self.ev.current_energy < self.ev.energy_consumption_pk:   # 当前电量不足以到达下一个位置
            return False
        self.ev.current_energy -= self.ev.energy_consumption_pk  # 更新车辆的当前电量
        return True


# 蚁群算法
class ACO:
    def __init__(self):
        # 参数定义及赋值
        self.rho = max_rho        # 信息素挥发系数
        self.pheromones = {}      # 信息素矩阵
        self.count = 0            # 找到的充电站个数

        self.optimal_cost = float('inf')   # 最优成本
        self.optimal_path = []             # 最优路径
        self.optimal_path_length = None    # 最优路径的长度
        self.optimal_cs = None             # 最优充电站

        self.all_optimal_costs = []        # 每一次迭代的最优成本
        self.all_average_costs = []        # 每一次迭代的平均成本

    def run(self):
        # 总迭代开始
        for i in range(max_iter):
            iter_optimal_cost = float('inf')  # 本次迭代的最优成本
            iter_optimal_path = []            # 本次迭代的最优路径
            iter_optimal_path_length = None   # 本次迭代的最优路径的长度
            iter_optimal_cs = None            # 本次迭代的最优充电站

            all_successful_ant = []  # 所有成功找到充电站的蚂蚁构建的路径和成本
            # Step 1:初始化每只蚂蚁依次构建路径
            for j in range(num_ant):
                ant = Ant(copy.deepcopy(ev))
                ant.run(self.pheromones)
                if ant.successful:  # 成功找到充电站
                    if cs_set[ant.destination].cost == float('inf'):
                        self.count += 1
                    if ant.path not in cs_set[ant.destination].path:  # 更新到达该充电站的新的路径和成本
                        cs_set[ant.destination].cost.append(ant.cost)
                        cs_set[ant.destination].path.append(ant.path)
                    all_successful_ant.append((ant.path, ant.cost))
                    # 更新本次迭代的最优解
                    if ant.cost < iter_optimal_cost:
                        iter_optimal_cost = ant.cost
                        iter_optimal_path = ant.path
                        iter_optimal_path_length = ant.total_distance
                        iter_optimal_cs = ant.destination

            print("------------------------")
            print(f"Iteration {i}:")
            print(f"成功率为: {len(all_successful_ant) / num_ant * 100}%")
            print(f"iter_optimal_cost: {iter_optimal_cost}")
            print(f"iter_optimal_path: {iter_optimal_path}")
            print(f"iter_optimal_path_length: {iter_optimal_path_length}")
            print(f"iter_optimal_cs: {cs_set[iter_optimal_cs].idx}, {iter_optimal_cs}, {cs_set[iter_optimal_cs].buy_price}")
            print("------------------------")

            # Step 2:更新信息素浓度
            # Step 2.1:信息素挥发
            tmp_phe = copy.deepcopy(self.pheromones)
            for key, value in tmp_phe.items():
                self.pheromones[key] = max(value * (1 - self.rho), min_phe)

            all_update_ant = []
            for item in all_successful_ant:
                if item not in all_update_ant:
                    all_update_ant.append(item)

            tmp_all_cost = 0
            # Step 2.2:新增信息素
            for path, cost in all_update_ant:
                tmp_all_cost += cost
                num = len(path)
                aug_phe = Q / cost
                for k in range(num - 1):
                    start, end = path[k], path[k + 1]
                    self.pheromones[(start, end)] = min(self.pheromones.get((start, end), phe_init) + aug_phe, max_phe)

            # Step 3:更新最优解
            if iter_optimal_cost < self.optimal_cost:
                self.optimal_cost = iter_optimal_cost
                self.optimal_path = iter_optimal_path
                self.optimal_path_length = iter_optimal_path_length
                self.optimal_cs = iter_optimal_cs

            # # 更新全局最优解的信息素浓度
            # num = self.optimal_path_length
            # aug_phe = self.Q / self.optimal_cost
            # for k in range(1, num):
            #     start, end = self.optimal_path[k-1], self.optimal_path[k]
            #     self.pheromones[(start, end)] = self.pheromones.get((start, end), self.phe_init) + aug_phe / 50

            # Step 4:记录每一次迭代的最优成本和平均成本，用于绘图
            self.all_optimal_costs.append(iter_optimal_cost)
            self.all_average_costs.append(tmp_all_cost / len(all_update_ant))

            # 更新信息素挥发系数
            self.rho = min_rho + (num_cs - self.count) / num_cs * (max_rho - min_rho)


if __name__ == "__main__":
    # 颜色定义，0：白色，1：黑色，2：蓝色，3：绿色
    color_map = {0: [1, 1, 1, 0.6], 1: [0, 0, 0, 0.6], 2: [0, 0, 1, 0.6], 3: [0, 1, 0, 0.6]}

    # 路径颜色定义，1.黄色，2.红色，3.绿色，4.棕色，5.粉色，6.橙色，7.紫色，8.青色, 9.黄色
    color_map1 = ['#FFFF00', '#FF0000', '#00FF00', '#C76813', '#FFC0CB', '#FA842B', '#992572', '#00FFFF', '#FFBF00']
    # 指定每条路径对应的颜色
    color_map2 = [color_map1[1], color_map1[0], color_map1[6], color_map1[0], color_map1[5], color_map1[7]]

    # 地图数据，np.array类型的二维数组，网格地图，每个不同的值代表不同的含义，0：道路，1：障碍物，2：动态充电道路，3：充电站。
    # 注意！！！注意！！！注意！！！
    # 每个网格的grid索引和实际的坐标有区别，实际坐标为：(x,y)，而对应的grid索引为：(y,x)，也就是取grid值，为grid[y][x]。
    grid = np.array(
        # 0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19
        [[0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 2, 2, 0, 0, 2, 2, 0, 1, 1],   # 0
         [1, 0, 0, 0, 2, 0, 2, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 3, 1],   # 1
         [1, 0, 1, 1, 1, 2, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],   # 2
         [1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],   # 3
         [1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 2, 2, 2, 0, 1, 1, 0, 1, 1],   # 4
         [1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 2, 1, 0, 0, 1],   # 5
         [0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 2, 1, 1, 0, 1],   # 6
         [1, 1, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],   # 7
         [1, 1, 0, 1, 0, 1, 1, 2, 1, 1, 1, 0, 0, 0, 0, 3, 1, 1, 0, 1],   # 8
         [1, 0, 0, 1, 0, 1, 1, 2, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1],   # 9
         [1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 2, 1, 1, 1, 1, 0, 1],   # 10
         [1, 2, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 2, 1, 0, 0, 0, 0, 0],   # 11
         [0, 0, 0, 1, 1, 0, 1, 2, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 2, 1],   # 12
         [0, 1, 2, 1, 1, 0, 1, 2, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 2, 1],   # 13
         [0, 1, 2, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],   # 14
         [0, 1, 2, 1, 0, 1, 1, 1, 3, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0],   # 15
         [0, 1, 2, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 3, 1, 0, 0],   # 16
         [0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1],   # 17
         [1, 3, 1, 0, 0, 0, 0, 0, 2, 2, 0, 0, 1, 0, 2, 2, 0, 1, 0, 1],   # 18
         [1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1]])  # 19

    # 电动汽车的起始坐标设置
    ev_coordinate = (0, 0)

    # 画出初始化地图，初始化电动汽车对象和充电站对象，ev为电动汽车对象，CS_set为充电站对象的集合
    ev, cs_set = init()

    # 参数设置
    max_iter = 2000        # 最大迭代次数
    num_ant = 10          # 蚂蚁数量
    num_cs = len(cs_set)  # 充电站的数量
    alpha = 1.0           # 信息素影响因子
    beta = 2.0            # 启发函数影响因子
    max_rho = 0.8         # 最大信息素挥发系数
    min_rho = 0.5         # 最小信息素挥发系数
    Q = 100               # 信息素增量
    phe_init = 5.0        # 初始的信息素浓度
    min_phe = 0.001       # 最小的信息素浓度
    max_phe = 10.0        # 最大的信息素浓度

    aco = ACO()
    aco.run()


    print("电动汽车的信息如下: ")
    print(f"电动汽车的起始坐标: {ev.start_coordinate}")
    print(f"电动汽车的当前电量: {ev.current_energy}%")
    print(f"电动汽车的预期电量: {ev.expected_energy}%")
    print(f"电动汽车每公里耗电量: {ev.energy_consumption_pk}%")
    print("------------------------")

    print("------------------------")
    print(f"最优成本: {aco.optimal_cost}")
    print(f"最优路径: {aco.optimal_path}")
    print(f"最优路径的长度: {aco.optimal_path_length}")
    print(f"最优充电站: {cs_set[aco.optimal_cs].idx}, {aco.optimal_cs}, {cs_set[aco.optimal_cs].buy_price}")
    print("------------------------")

    print("------------------------")
    print("电动汽车到达各个充电站的信息: ")
    for key, value in cs_set.items():
        print(f"{value.idx}, {key}, {value.buy_price}\n{value.cost}\n{value.path}")


    print("------------------------")
    # 画出每次迭代的成本曲线，0表示画出最优成本，1表示画出平均成本
    draw_data(1)
    # 画出找到最优充电站的路线地图
    draw_map(1)


'''
20*20的数据：
最优解：
CS编号 CS坐标   cost   price len path
0     (18,1), 65.39, 1.5,  21, [(0, 0), (1, 0), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1), (8, 1), (9, 1), (10, 1), (10, 0), (11, 0), (12, 0), (13, 0), (14, 0), (15, 0), (16, 0), (17, 0), (17, 1), (18, 1)]
1     (15,8), 57.07, 1.3,  23, [(0, 0), (1, 0), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (5, 2), (5, 3), (6, 3), (7, 3), (7, 4), (8, 4), (9, 4), (10, 4), (11, 4), (12, 4), (13, 4), (14, 4), (14, 5), (15, 5), (15, 6), (15, 7), (15, 8)]
2     (7,14), 70.08, 1.6,  21, [(0, 0), (1, 0), (1, 1), (1, 2), (1, 3), (2, 3), (3, 3), (3, 4), (3, 5), (4, 5), (4, 6), (4, 7), (5, 7), (6, 7), (7, 7), (7, 8), (7, 9), (7, 10), (7, 11), (7, 12), (7, 13), (7, 14)]
3     (16,16), n.nn, 1.2,  nn, []
4     (1,18), 62.43, 1.4,  25, [(0, 0), (1, 0), (1, 1), (1, 2), (1, 3), (2, 3), (3, 3), (3, 4), (3, 5), (2, 5), (2, 6), (2, 7), (2, 8), (2, 9), (1, 9), (1, 10), (1, 11), (1, 12), (2, 12), (2, 13), (2, 14), (2, 15), (2, 16), (2, 17), (1, 17), (1, 18)]
'''