import random
from itertools import combinations
from shapely.geometry import LineString, Point

import copy  # for deepcopy


class MACHINE():
    """
        [ MACHINE ]..
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
        #222#

        self.drawn_lines_copy = copy.deepcopy(drawn_lines)
        

        

    def find_best_selection(self):
        available = [[point1, point2] for (point1, point2) in list(combinations(self.whole_points, 2)) if self.check_availability([point1, point2])]
        self.avail_lines_num = self.count_available()
        print(self.avail_lines_num)
        drawn_lines = self.drawn_lines
        # 이미 그어진 선분의 모든 좌표를 얻기
        drawn_points = [point for line in drawn_lines for point in line]
        # 이미 그어진 선분의 두 좌표를 제외하다
        remaining_lines = [[point1, point2] for [point1, point2] in available if point1 not in drawn_points and point2 not in drawn_points]
        if not remaining_lines:
            return random.choice(available)#모든 가능한 선 중에 랜덤으로 한개의 선분 반환
        else:
            return random.choice(remaining_lines)
    
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

    # 주어진 점이 삼각형 내부에 있는지 확인
    def is_point_inside_triangle(self, point, triangle):
        triangle_polygon = Polygon(triangle)
        return triangle_polygon.contains(Point(point))

    # 삼각형을 완성했을 때 짝수 개의 삼각형을 유지할 수 있는지 확인
    def can_form_even_number_of_triangles(self, triangle):
        num_triangles = len(self.triangles) + 1  # 현재 삼각형 포함
        return num_triangles % 2 == 0

    # 짝수 균형을 유지할 수 있는 선분을 선택
    def select_line_to_maintain_even_balance(self, triangle):
        for line in combinations(triangle, 2):
            if line not in self.drawn_lines and self.check_availability(line):
                return line
        return None

    # 사용 가능한 모든 선분을 찾는 함수
    def find_available_lines(self):
        available = [[point1, point2] for (point1, point2) in list(combinations(self.whole_points, 2)) if self.check_availability([point1, point2])]
        return available
    '''
    여기서부터 trick 관련 함수
    '''
    just_3 = list(combinations(self.whole_points, 3))
    triangle_point = []                               # point of completing a triangle - (1,1), (1,4), (4,4)
    trick_point = []                                  # point of completing a trick - (1,1), (1,4), (4,4), (2,2)
    avail_trick_point = []
    most_completed_trick = []                       # point of completing the most completed trick
    trick_line = []                                 # line needed to complete the trick 
    
    # function of calculating triangle area
    def area(p1, p2, p3):
        area_val = 0.5 * abs(p1[0]*p2[1] + p2[0]*p3[1] + p3[0]*p1[1] - p1[0]*p3[1] - p3[0]*p2[1] - p2[0]*p1[1])
        return area_val
    
    # find triangle_point
    for p in just_3:
        p_set = list(p)
        p1, p2, p3 = p_set[0], p_set[1], p_set[2]
        area_val = area(p1, p2, p3)
        line = []
        line1 = LineString([p1,p2]) ; line2 = LineString([p1,p3]) ; line3 = LineString([p2,p3])
        line.append(line1) ; line.append(line2); line.append(line3)
        if area_val > 0:
          i=0
          for point in self.whole_points: # 선분 안에 점이 들어가는 삼각형 필터링
            if point == p1 or point == p2 or point == p3:
              continue
            else:
              if bool(line1.intersection(Point(point))) or bool(line2.intersection(Point(point))) or bool(line3.intersection(Point(point))):
                i += 1
          if i == 0:  
            triangle_point.append(p_set)       
        
    # find trick_point
    for i, p in enumerate(triangle_point): # 삼각형 안에 점이 1개 들어가면 trick_point에 저장
        inside_num = 0
        for q in self.whole_points: #
            if q not in p:
                if area(p[0],p[1],p[2]) == area(p[0],p[1],q) + area(p[0],q,p[2]) + area(q,p[1],p[2]):
                    inside_num += 1
                    inside_point = q
        if inside_num == 1:
            p_copy = p.copy()  # p를 복사해서 새로운 리스트 생성
            p_copy.append(inside_point)  
            trick_point.append(p_copy)  
            
    # find trick_line
    i=0
    for p in trick_point:
        trick_line.append(list(combinations(p, 2)))
        
    # validate trick_point
    for j in range(0,len(trick_point)): # 유효한 trick만 avail_trick_point에 저장: trick이 깨졌거나 trick에 걸리는 상황은 제외
        outer_point = list(trick_point[j][0:3])
        total_line = list(combinations(trick_point, 2))
        outside_line = list(combinations(outer_point,2))
        inside_line = [i for i in total_line if i not in outside_line]
    
        out_drawen_check = [line in self.drawn_lines_copy for line in outside_line]
        in_drawen_check = [line in self.drawn_lines_copy for line in inside_line]
        out_count = sum(out_drawen_check)
        in_count = sum(in_drawen_check)
    
        avail_candidate = list(trick_point[j])
    
        if out_count == 3:
            if in_count == 0 or 2:
                avail_trick_point.append(avail_candidate)
        else:
            if in_count == 0:
                avail_trick_point.append(avail_candidate)
    print(f"avail_trick_point is {avail_trick_point}")

    # find most_completed_trick
    highest_count = -1
    for j in range(0,len(avail_trick_point)):
        total_line = list(combinations(avail_trick_point[j],2))
        out_line = list(combinations(avail_trick_point[j][0:3],2))
        in_line = [i for i in total_line if i not in out_line]
        trick_candidate = []
    
        alredy_line = [tuple(sorted(line)) for line in self.drawn_lines_copy]
        total_line = [tuple(sorted(line)) for line in total_line]
    
        in_check = [line in alredy_line for line in in_line] ; in_count = sum(in_check)
        out_check = [line in alredy_line for line in out_line] ; out_count = sum(out_check)
        count = in_count + out_count
        
        first_check=0
    
        if out_count == 3:
          if in_count == 1 or in_count == 3: # 이미 선이 모드 그어졌거나 trick에 걸리는 상황 그냥 pass
              pass
          elif in_count == 2:
            trick_candidate.append(avail_trick_point[j])
          elif in_count == 0:
            trick_candidate.append(avail_trick_point[j])                  
        else:
          if in_count == 0:
            trick_candidate.append(avail_trick_point[j])

        for i in range(sum(len(sublist) for sublist in trick_candidate)):
          total_line = list(combinations(avail_trick_point[i],2))
          count = sum([(line in alredy_line) for line in total_line])
          if count > highest_count:
            most_completed_trick = []
            most_completed_trick.append(trick_candidate[0][i])
            highest_count = count
          elif count == highest_count:
            most_completed_trick.append(trick_candidate[0][i])

    sub_factory = []
    for i in range(0, len(most_completed_trick), 4):
      sub_factory.append(most_completed_trick[i:i+4])
    most_completed_trick = sub_factory
    return most_completed_trick
        
    
    def complete_trick(self, most_completed_trick):
        best_trick_line = []
        for i in range(len(most_completed_trick)):
            outer_point = list(most_completed_trick[i][0:3])
            total_line = list(combinations(most_completed_trick[i], 2))
            outside_line = list(combinations(outer_point,2))
            inside_line = [i for i in total_line if i not in outside_line]
        
            out_drawen_check = [line in self.drawn_lines_copy for line in outside_line]
            in_drawen_check = [line in self.drawn_lines_copy for line in inside_line]
            out_count = sum(out_drawen_check)
            in_count = sum(in_drawen_check)
        
            if out_count == len(outside_line):  # 바깥에 있는 점이 모두 연결되어 있는 경우
              if in_count == 0 or in_count == 2:  # 안쪽에 선이 0개 또는 2개 완성되어 있을 때
                  candidate = [c for c in inside_line if c not in self.drawn_lines_copy]
                  best_trick_line.append(candidate)
            else:  # 바깥에 있는 점이 모두 연결되어 있지 않은 경우
                candidate = [c for c in outside_line if c not in self.drawn_lines_copy]
                best_trick_line.append(candidate)
        
        # 하위 리스트를 허물어 저장할 새로운 리스트 생성
        flattened_best_trick_line = []
        
        # 각 하위 리스트의 원소를 새로운 리스트에 추가
        for sublist in best_trick_line:
            for item in sublist:
                flattened_best_trick_line.append(item)
    
        return flattened_best_trick_line

    
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
