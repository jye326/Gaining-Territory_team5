import random
from itertools import combinations
from shapely.geometry import LineString, Point, Polygon

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
        
   
    def find_triangle_completing_line(self):
        """
        삼각형을 완성하는 선분을 찾는 함수
        """
        for line in self.drawn_lines:
            connected_lines = self.find_connected_lines(line)
            for connected_line in connected_lines:
                possible_triangle = self.form_triangle(line, connected_line)
                if possible_triangle and self.is_triangle_empty(possible_triangle):
                    completing_line = self.find_completing_line(possible_triangle)
                    if completing_line:
                        completing_line = [list(completing_line)]
                        return completing_line
        return None

    def find_completing_line(self, triangle):
        """
        삼각형을 완성할 수 있는 선분을 찾는 함수
        """
        for line in combinations(triangle, 2):
            if line not in self.drawn_lines and self.check_availability(line):
                return line
        return None  
        

    def find_best_selection(self):
        # 0. 가능한 모든 선분
        available = [[point1, point2] for (point1, point2) in list(combinations(self.whole_points, 2)) if self.check_availability([point1, point2])]
        
        # 1. 상대방에게 점수 안 주는 전략
        self.avail_lines_num = self.count_available()
        print(self.avail_lines_num)
        drawn_lines = self.drawn_lines
        # 이미 그어진 선분의 모든 좌표를 얻기
        drawn_points = [point for line in drawn_lines for point in line]
        #  이미 연결점이 있는 선을 제외하고, 아무것도 연결하지 않은 선의 집합 (다른 점과 연결되어 있지 않은 점 2개를 이은 선분들의 집합)
        remaining_lines = [[point1, point2] for [point1, point2] in available if point1 not in drawn_points and point2 not in drawn_points]
        
        # 2. Trick 전략
        three = Trick.just_3(self)
        triangle_point = Trick.find_triangle_point(self, three)
        trick_point = Trick.find_trick_point(self, triangle_point)
        avail_trick_point = Trick.validate_trick_point(self, trick_point)
        most_completed_trick = Trick.find_most_completed_trick(self, avail_trick_point)
        best_trick_line = Trick.complete_trick(self, most_completed_trick)

        # 3. 삼각형 완성 전략
        triangle_completing_line = self.find_triangle_completing_line()
         
        # 4. 짝수 삼각형 전략
        even_triangle_lines = [line for line in available if self.can_form_even_number_of_triangles_after_drawing_line(line)]

        # RULE
        if triangle_completing_line:
            print(f"triangle_completing_line -> {triangle_completing_line}")
            return random.choice(triangle_completing_line)
        else:
            if best_trick_line:
                print(f"best_trick_line -> {best_trick_line}")
                return random.choice(best_trick_line)
            elif remaining_lines:
                print(f"remaining_lines -> {remaining_lines}")
                return random.choice(remaining_lines)
            elif even_triangle_lines:
                print(f"even_triangle_lines -> {even_triangle_lines}")
                return random.choice(even_triangle_lines)
            else:
                print(f"available -> {available}")
                return random.choice(available)   
        '''
        <일단 주석 처리. 트릭 전략, 삼각형 완성 전략, 짝수 삼각형 전략, 기존로직 위에 종합함.>
        # 2. 삼각형 완성 전략
        triangle_completing_line = self.find_triangle_completing_line()
        if triangle_completing_line:
            triangle_completing_line = list(triangle_completing_line)
            return triangle_completing_line
         
        # 3. 짝수 삼각형 전략
        even_triangle_lines = [line for line in available if self.can_form_even_number_of_triangles_after_drawing_line(line)]
        if even_triangle_lines:
            return random.choice(even_triangle_lines)

        # 4. 기존 로직에 따라 선분을 선택
        if not remaining_lines:
            return random.choice(available)#모든 가능한 선 중에 랜덤으로 한개의 선분 반환
        else:
            return random.choice(remaining_lines)
        '''

       
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


        # == 짝수개의 삼각형 ==
        # 짝수 개의 삼각형 전략을 사용하여 선분을 찾음
    def find_even_triangle_strategy(self):
        for line in self.drawn_lines:
            connected_lines = self.find_connected_lines(line)
            for connected_line in connected_lines:
                possible_triangle = self.form_triangle(line, connected_line)
                if possible_triangle and self.is_triangle_empty(possible_triangle):
                    if self.can_form_even_number_of_triangles(possible_triangle):
                        return self.select_line_to_maintain_even_balance(possible_triangle)
        return None

    # 주어진 선분과 연결된 다른 선분들을 찾음
    def find_connected_lines(self, line):
        connected = []
        for other_line in self.drawn_lines:
            if line != other_line and (line[0] in other_line or line[1] in other_line):
                connected.append(other_line)
        return connected

    # 주어진 두 선분으로 삼각형을 형성할 수 있는지 확인
    def form_triangle(self, line1, line2):
        points = set(line1 + line2)
        if len(points) == 3:
            return list(points)
        return None
        
    # 삼각형 내부에 다른 점이 없는지 확인
    def is_triangle_empty(self, triangle):
        triangle_set = set(triangle)
        for point in self.whole_points:
            if point not in triangle_set and self.is_point_inside_triangle(point, triangle):
                return False
        return True

    
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
    def can_form_even_number_of_triangles_after_drawing_line(self, line):
        # 가정: 새로운 선분을 추가하고, 새로운 삼각형을 만들어낼 수 있는지 확인
        # 만들어진 새로운 삼각형의 개수를 계산
        new_triangles = 0

        # 새로운 선분을 추가한 상태에서 연결된 모든 선분을 검사
        for existing_line in self.drawn_lines:
            if self.is_connected(line, existing_line):
                # 새로운 삼각형을 형성하는지 확인
                if self.forms_new_triangle(line, existing_line):
                    new_triangles += 1

        # 전체 삼각형 개수 (기존의 개수 + 새로운 개수)가 짝수인지 확인
        total_triangles = len(self.triangles) + new_triangles
        return total_triangles % 2 == 0

    def is_connected(self, line1, line2):
        # 두 선분이 연결되어 있는지 확인하는 로직
        return set(line1) & set(line2) != set()

    def forms_new_triangle(self, line1, line2):
        # 두 선분으로 형성되는 새로운 삼각형 확인
        combined_points = set(line1 + line2)
        return len(combined_points) ==3


# Trick 
class Trick:
    def __init__(self, machine_instance):
            self.machine_instance = machine_instance
        
    def just_3(self):
        three = list(combinations(self.whole_points, 3))
        return three
        
    # function of calculating triangle area
    def area(p1, p2, p3):
        area_val = 0.5 * abs(p1[0]*p2[1] + p2[0]*p3[1] + p3[0]*p1[1] - p1[0]*p3[1] - p3[0]*p2[1] - p2[0]*p1[1])
        return area_val
    
    # find triangle_point
    def find_triangle_point(self, three):
        triangle_point = []
        for p in three:
            p_set = list(p)
            p1, p2, p3 = p_set[0], p_set[1], p_set[2]
            area_three = Trick.area(p1, p2, p3)
            lines = [LineString([p1,p2]), LineString([p1,p3]), LineString([p2,p3])]
            if area_three > 0:
                is_point_on_line = False
                for point in self.whole_points: # 선분 안에 점이 들어가는 삼각형 필터링
                    if point in p_set:
                        continue
                    point = Point(point)
                    if any(line.touches(point) or line.intersection(point) for line in lines):
                        is_point_on_line = True
                        break
                if not is_point_on_line:
                    triangle_point.append(p_set)
        return triangle_point
        
    # find trick_point
    def find_trick_point(self, triangle_point):
        trick_point = []
        for i, p in enumerate(triangle_point): # 삼각형 안에 점이 1개 들어가면 trick_point에 저장
            inside_num = 0
            for q in self.whole_points: #
                if q not in p:
                    if Trick.area(p[0],p[1],p[2]) == Trick.area(p[0],p[1],q) + Trick.area(p[0],q,p[2]) + Trick.area(q,p[1],p[2]):
                        inside_num += 1
                        inside_point = q
            if inside_num == 1:
                p = list(p)
                p.append(inside_point)
                trick_point.append(p)
        return trick_point  
            
    # find trick_line
    def find_trick_line(self, trick_point):
        trick_line = []
        for p in trick_point:
            trick_line.append(list(combinations(p, 2)))
        return trick_line  
        
    # validate trick_point
    def validate_trick_point(self, trick_point):
        avail_trick_point = []
        for j in range(0,len(trick_point)): # 유효한 trick만 avail_trick_point에 저장: trick이 깨졌거나 trick에 걸리는 상황은 제외
            out_point = list(trick_point[j][0:3])
            total_line = list(combinations(trick_point, 2))
            out_line = list(combinations(out_point,2))
            in_line = [i for i in total_line if i not in out_line]
        
            out_drawen_check = [line in self.drawn_lines for line in out_line]
            in_drawen_check = [line in self.drawn_lines for line in in_line]
            out_count = sum(out_drawen_check)
            in_count = sum(in_drawen_check)
        
            avail_candidate = list(trick_point[j])
        
            if out_count == 3:
                if in_count == 0 or in_count == 2: # out_count가 3일 때, in_count는 0 또는 2인 경우에 추가
                    avail_trick_point.append(avail_candidate)
            else:
                if in_count == 0: # out_count가 3이 아닐 때, in_count는 0인 경우에만 추가
                    avail_trick_point.append(avail_candidate)
        return avail_trick_point

    # find most_completed_trick
    def find_most_completed_trick(self, avail_trick_point):
        most_completed_trick = []
        highest_out_count = -1
        highest_in_count = -1
        trick_candidate = []
        for j in range(0,len(avail_trick_point)):
            total_line = [sorted(line) for line in combinations(avail_trick_point[j], 2)]
            out_line = [sorted(line) for line in combinations(avail_trick_point[j][0:3], 2)]
            in_line = [line for line in total_line if line not in out_line]
            
            already_line = [sorted(line) for line in self.drawn_lines]
            in_check = [line in already_line for line in in_line]
            out_check = [line in already_line for line in out_line]
            in_count = sum(in_check)
            out_count = sum(out_check)

            if out_count == 3:
                if in_count != 2:  # 이미 선이 모드 그어졌거나 trick에 걸리는 상황 그냥 pass
                    pass
                else:
                    trick_candidate.append(avail_trick_point[j])
            else:
                if in_count == 0:
                    trick_candidate.append(avail_trick_point[j])

            if out_count == 3:
                if in_count > highest_in_count:
                    highest_in_count = in_count
                    most_completed_trick = [avail_trick_point[j]]
                elif in_count == highest_in_count:
                    most_completed_trick = [avail_trick_point[j]]                   
            elif highest_in_count == -1:
                if out_count > highest_out_count:
                    highest_out_count = out_count
                    most_completed_trick = [avail_trick_point[j]]
                elif out_count == highest_out_count:
                    most_completed_trick.append(avail_trick_point[j])
        return most_completed_trick

    def complete_trick(self, most_completed_trick):
        best_trick_line = []
        for i in range(len(most_completed_trick)):
            total_line = [sorted(line) for line in combinations(most_completed_trick[i], 2)]
            out_line = [sorted(line) for line in list(combinations(most_completed_trick[i][0:3], 2))]
            in_line = [i for i in total_line if i not in out_line]        
            already_line = [sorted(line) for line in self.drawn_lines]
            in_check = [line in already_line for line in in_line]
            out_check = [line in already_line for line in out_line]
            in_count = sum(in_check)
            out_count = sum(out_check)

            if out_count == 3:
                if in_count == 3 or in_count == 1:
                    pass
                else:
                    candidate = [c for c in in_line if c not in self.drawn_lines]
                    best_trick_line.append(candidate)
            else:
                candidate = [c for c in out_line if c not in self.drawn_lines]
                best_trick_line.append(candidate)

        
        # 하위 리스트를 허물어 저장할 새로운 리스트 생성
        flattened_best_trick_line = []
        
        # 각 하위 리스트의 원소를 새로운 리스트에 추가
        for sublist in best_trick_line:
            for item in sublist:
                flattened_best_trick_line.append(item)
        
        # 그을 수 있는 선
        available_best_trick_line = []
        
        # 그을 수 있는 선만을 best_trick_line에 저장
        for i in range(len(flattened_best_trick_line)):
            if bool(self.check_availability(flattened_best_trick_line[i])):
                available_best_trick_line.append(self.check_availability(flattened_best_trick_line[i]))
        
        best_trick_line= available_best_trick_line
        best_trick_line = [flattened_best_trick_line[i:i+1] for i in range(0, len(flattened_best_trick_line))] 
        best_trick_line = [list(item[0]) for item in best_trick_line]
        return best_trick_line

    
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
