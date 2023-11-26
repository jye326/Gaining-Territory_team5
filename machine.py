import random
from itertools import combinations
from shapely.geometry import LineString, Point


class MACHINE():
    """
        [ MACHINE ]
        MinMax Algorithm을 통해 수를 선택하는 객체.
        - 모든 Machine Turn마다 변수들이 업데이트 됨

        ** To Do **
        MinMax Algorithm을 이용하여 최적의 수를 찾는 알고리즘 생성
           - class 내에 함수를 추가할 수 있음
           - 최종 결과는 find_best_selection을 통해 Line 형태로 도출
               * Line: [(x1, y1), (x2, y2)] -> MACHINE class에서는 x값이 작은 점이 항상 왼쪽에 위치할 필요는 없음 (System이 organize 함)
    """
    def __init__(self, score=[0, 0], drawn_lines=[], whole_lines=[], whole_points=[], location=[]):
        self.id = "MACHINE"
        self.score = [0, 0] # USER, MACHINE
        self.drawn_lines = [] # Drawn Lines
        self.board_size = 7 # 7 x 7 Matrix
        self.num_dots = 0
        self.whole_points = []
        self.location = []
        self.triangles = [] # [(a, b), (c, d), (e, f)]
        
        self.sim_drawnline = []  #  simulation시에 사용할 그려진 라인들
        self.avail_lines_num = 0 #self.count_available() #처음에 하는거 의미 없는듯
        self.sim_score = [0,0]  # 가상의 점수
        self.sim_triangles = []
        #print(self.avail_lines_num)
        

        

    def find_best_selection(self):
        available = [[point1, point2] for (point1, point2) in list(combinations(self.whole_points, 2)) if self.check_availability([point1, point2])]
        self.avail_lines_num = self.count_available()
        print(self.avail_lines_num)
        return random.choice(available) #모든 가능한 선 중에 랜덤으로 한개의 선분 반환
    
    #턴마다 system에서 machine의 변수들을 업데이트 해 주는 것임

    def check_all_lines(self):
        available = [[point1, point2] for (point1, point2) in list(combinations(self.whole_points, 2)) if self.check_availability([point1, point2])]
        return available
        # 조건에 부합하는 모든 가능한 라인을 반환

    # count_avail_lines를 성공적으로 반환함
    def count_available(self):
        #board판과 whole_points정보를 토대로 게임을 시뮬레이션을 돌려서 그려질 수 있는 라인수를 반환
        # sim_drawnline 과 sim_check_availability으로 끝까지 simulation해보고 라인을 반환
        available = self.check_all_lines()
        for line1 in available:
            if self.sim_check_availability(line1):
                self.sim_drawnline.append(line1)

        print(self.sim_drawnline)
        avail_lines_num = len(self.sim_drawnline)
        return avail_lines_num
        
    # 한번 끝까지 시뮬레이션 하기 위한 가상의 판단함수
    def sim_check_availability(self, line):
        line_string = LineString(line)

        # Must be one of the whole points
        condition1 = (line[0] in self.whole_points) and (line[1] in self.whole_points)
        
        # Must not skip a dot
        condition2 = True
        for point in self.whole_points:
            if point==line[0] or point==line[1]:
                continue
            else:
                if bool(line_string.intersection(Point(point))):
                    condition2 = False

        # Must not cross another line
        condition3 = True
        for l in self.sim_drawnline:
            if len(list(set([line[0], line[1], l[0], l[1]]))) == 3:
                continue
            elif bool(line_string.intersection(LineString(l))):
                condition3 = False

        # Must be a new line
        condition4 = (line not in self.sim_drawnline)

        if condition1 and condition2 and condition3 and condition4:
            return True
        else:
            return False    

    def check_availability(self, line):
        line_string = LineString(line)

        # Must be one of the whole points
        condition1 = (line[0] in self.whole_points) and (line[1] in self.whole_points)
        
        # Must not skip a dot
        condition2 = True
        for point in self.whole_points:
            if point==line[0] or point==line[1]:
                continue
            else:
                if bool(line_string.intersection(Point(point))):
                    condition2 = False

        # Must not cross another line
        condition3 = True
        for l in self.drawn_lines:
            if len(list(set([line[0], line[1], l[0], l[1]]))) == 3:
                continue
            elif bool(line_string.intersection(LineString(l))):
                condition3 = False

        # Must be a new line
        condition4 = (line not in self.drawn_lines)

        if condition1 and condition2 and condition3 and condition4:
            return True
        else:
            return False    

    
# Minmax Tree Node
class Node:
    def __init__(self, sim_drawnlines):
        self.sim_drawnlines = sim_drawnlines    # 현재 노드에서 그려진 라인들
        self.score = None    # leaf Node score = 내 점수 - 상대 점수
        self.children = []
        self.parent = None
        
    def add_child(self, child):
        child.parent = self
        self.children.append(child)

    def get_level(self):
        level = 0
        p = self.parent
        while p:
            level += 1
            p = p.parent
        return level
    
    def print_tree(self):
        spaces = ' ' * self.get_level() * 3
        prefix = spaces + "|__" if self.parent else ""
        print(prefix + self.score)
        if self.children :
            for child in self.children:
                child.print_tree()

    def minimax(self, depth, isMaximizingPlayer, alpha, beta):
        if depth == 0 or self.is_terminal_node():  # depth == 0 > 미리 지정한 depth에 도달, is_terminal_node() -> 해당 시뮬레이션이 게임 종료에 도달
            return self.score  # 중간에 서로의 점수를 반환해야 함 #근데 시뮬레이션 시의 가상의 점수를 반환해야하는거 아닌가?
        
        # machine.py에서 시뮬레이션 도중 게임의 종료여부를 판단해야함
        # machine.py에서 시뮬레이션 별로 원하는 시점에서 해당 게임의 점수를 반환할 수 있어야 함
        if isMaximizingPlayer:
            maxEval = -float('inf')
            for child in self.children:
                eval = child.minimax(depth - 1, False, alpha, beta)
                maxEval = max(maxEval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return maxEval

        else:
            minEval = float('inf')
            for child in self.children:
                eval = child.minimax(depth - 1, True, alpha, beta)
                minEval = min(minEval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return minEval
            




# node - 위에 구현한 Node를 사용하면 될 듯 
# depth - 임의로 설정
# isMaximizingPlayer -> turn확인 후 내 turn일때 true넣자
# alpah, beta => 처음에는 각각 -무한, +무한으로 넣자





