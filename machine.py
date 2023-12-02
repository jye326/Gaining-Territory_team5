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
        
        self.sim_drawn_line = []  #  simulation시에 사용할 그려진 라인들
        self.avail_lines_num = 0 #self.count_available() #처음에 하는거 의미 없는듯
        self.sim_score = [0,0]  # 가상의 점수
        self.sim_triangles = []
        
        
        #print(self.remaind_available_lines(self.drawn_lines)) # system이 board 정보를 업데이트 하는 시점이 machine초기화 이후이기 때문에 시작 전에 미리 알 수 없을 것 같음
        # count_available() -> len(remaind_available_lines())로 변환
        self.drawn_lines_copy = copy.deepcopy(drawn_lines)
        


    def find_best_selection(self):
        available = [[point1, point2] for (point1, point2) in list(combinations(self.whole_points, 2)) if self.check_availability([point1, point2])]
        
        if self.avail_lines_num == 0:
            empty = []
            self.avail_lines_num = len(self.remaind_available_lines(empty))
        
        drawn_lines = self.drawn_lines
        # 이미 그어진 선분의 모든 좌표를 얻기
        #drawn_points = [point for line in drawn_lines for point in line]
        # 이미 그어진 선분의 두 좌표를 제외하다
        #remaining_lines = [[point1, point2] for [point1, point2] in available if point1 not in drawn_points and point2 not in drawn_points]
        # 왜 선분이 아니라 선분의 두 좌표를 제외?

        print(self.avail_lines_num) #초기에 짝/홀 정하고 이후 ->

        remaining_lines = [line for line in available if line not in drawn_lines and self.check_availability(line)] # 후공일 경우 남아있는 라인 업데이트

        print(self.remaind_available_lines(drawn_lines= self.drawn_lines))
        #여기까지는 정상=============================================================================
        #다음 스텝에 그릴 수 있는 모든 라인들 반환하는 함수
        
        if not remaining_lines:#남은 라인이 아무것도 없는 상태
            # RootNode Random으로 설정?
            return random.choice(available)#모든 가능한 선 중에 랜덤으로 한개의 선분 반환
        else:
            # 트리 생성 및 루트 노드에 가능한 모든 라인 추가
            
            tree = Tree(self.drawn_lines, self.avail_lines_num)
            tree.root.sim_drawn_lines.clear()#트리 생성시에 drawn_lines는 비워준다.
            root_node = tree.root
            
            # 가능한 모든 라인을 루트 노드의 자식으로 추가
            
            for line in remaining_lines:    
                child_node = tree.add_node(root_node, line)
                #child_node.sim_drawn_lines.append(line)
                self.populate_tree(child_node)  #root_node아닌가?
                print("한번은 끝나니?????????????????????????????")
            # 트리 출력 (디버깅용)
            tree.print_tree()
            # 임시로 랜덤한 라인을 선택하여 반환
            return random.choice(remaining_lines)
        
    def populate_tree(self, node):
        # 현재 노드의 sim_drawn_lines에 부모의 drawn_lines와 현재 라인 추가
        if node.parent and node.parent.sim_drawn_lines:
            node.sim_drawn_lines.extend(node.parent.sim_drawn_lines)  # extend를 사용하여 리스트의 요소를 추가
        
        # 현재 상황에서 그릴 수 있는 모든 라인을 찾아 자식 노드로 추가
        available_lines = self.sim_check_all_lines()
        for line in available_lines:
            if self.sim_check_availability(line, node.sim_drawn_lines):
                child_node = Tree.Node(line, 0)
                node.add_child(child_node)
                # if not child_node.is_terminal_node():
                #     self.populate_tree(child_node)
                if child_node.get_level() < 3:
                    self.populate_tree(child_node)
               


            # #민맥스를 처음부터 적용할 일은 없다? -> 모른다 일단 둘다 만들자            
            # tree = Tree(drawn_lines, self.avail_lines_num)#현재 그려진 라인을 루트로 하는 트리를 생성
            # #print(remaining_lines) #왜 3개밖에 안들어갔지?
            # #남은 모든 라인수에 대해 루트의 자식 트리를 만든다.
            # for line in remaining_lines:
            #     line_node = tree.Node(line)
            #     tree.root.add_child(line_node)
            # tree.print_tree()

            # return random.choice(remaining_lines)   #완성전까지는 일단 랜덤 반환
    
    #턴마다 system에서 machine의 변수들을 업데이트 해 주는 것임

    def check_all_lines(self):
        print("this is check_all_lines========================================")
        available = [[point1, point2] for (point1, point2) in list(combinations(self.whole_points, 2)) if self.check_availability([point1, point2])]
        return available
        # 조건에 부합하는 모든 가능한 라인을 반환
    def sim_check_all_lines(self):  #그냥 모든 라인을 반환
        empty = []
        print("this is sim_chek_all_lines========================================")
        available = [[point1, point2] for (point1, point2) in list(combinations(self.whole_points, 2)) if self.sim_check_availability([point1, point2], empty)]
        return available

    # count_avail_lines를 성공적으로 반환함 #len(remaind_available_lines)-> 남은 그릴 수 있는 선 수
    def remaind_available_lines(self, drawn_lines):
        print("this is remaind_available_lines========================================")
        #board판과 whole_points정보를 토대로 게임을 시뮬레이션을 돌려서 그려질 수 있는 라인수를 반환
        # sim_drawnline 과 sim_check_availability으로 끝까지 simulation해보고 라인을 반환
        if self.avail_lines_num ==0 :   # machine의 가능한 라인수 초기화가 필요한 경우
            available = self.sim_check_all_lines() # 후보 라인은 모든 라인이다.
        else:#이미 전체 그릴 수 있는 수는 등록이 되어 있고, 현재 상황에서 그릴 수 있는 line list가 필요한 경우
            available = self.check_all_lines()  # 현재 보드판을 available로 사용

        sim_drawn_lines = copy.deepcopy(drawn_lines)    #현재 그려진 라인

        for line1 in available:
            if self.sim_check_availability(line1, sim_drawn_lines=sim_drawn_lines):
                sim_drawn_lines.append(line1)
        
        return sim_drawn_lines
        
    # 한번 끝까지 시뮬레이션 하기 위한 가상의 판단함수
    def sim_check_availability(self, line, sim_drawn_lines):
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

        # Must not cross another
        condition3 = True
        for l in sim_drawn_lines:
            if len(list(set([line[0], line[1], l[0], l[1]]))) == 3: # 한점을 공유하는 경우
                continue
            elif bool(line_string.intersection(LineString(l))): # 교차한 경우 false
                condition3 = False

        

        # Must be a new line
        condition4 = (line not in sim_drawn_lines)

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
        triangle_polygon = self.Polygon(triangle)
        return triangle_polygon.contains(Point(point))

    # 삼각형을 완성했을 때 짝수 개의 삼각형을 유지할 수 있는지 확인
    def can_form_even_number_of_triangles(self):
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
    
# minmax는 node에서 돌릴 수 있어야함
# node는 tree와는 별개인가?
# Tree클래스는 1개만 생성한다. tree 내부에 node를 계속 만들어나간다.
# Tree는 machine의 모든 인자를 가지고 있어야함

class Tree:
    max_lines_num = 0
    class Node:
        def __init__(self, sim_drawn_lines, turn):
            self.sim_drawn_lines = [copy.deepcopy(sim_drawn_lines)] #node별로 개별 sim_drawnliens가 필요
            self.score = None   #sim_score
            self.children = []  # 그릴 수 있는 선택지들
            self.parent = None
            self.turn = turn # 턴 구분용 변수
            

        def add_child(self, child): #add_child(Node(이미 그려진 라인))
            child.parent = self
            self.children.append(child)

        def get_level(self):
            level = 0
            p = self.parent
            while p:
                level += 1
                p = p.parent
            return level
        
        def get_turn(self):
            return self.turn

        def is_terminal_node(self):
            # 터미널 노드 여부를 판단하는 로직 구현
            # 노드의 라인 수 == 처음에 판단한 총 라인수와 같아질 경우에 종료
            if self.get_level() == Tree.max_lines_num:
                return True
            else:
                return False

    def __init__(self, drawn_lines, max_lines_num): #tree class 만들기
        self.root = self.Node(drawn_lines, turn=0)
        Tree.max_lines_num = max_lines_num

    def add_node(self, parent_node, line): #line이 들어오면 해당 라인을 
        child_turn = 0 if parent_node.get_turn() == -1 else -1
        child_node = self.Node(line, turn=child_turn)
        parent_node.add_child(child_node)
        return child_node
    
    def print_tree(self, node=None, level=0):
        if node is None:
            node = self.root
        spaces = ' ' * level * 4
        prefix = spaces + "|__ " if node.parent else ""
        #print(prefix + str(node.sim_drawn_lines) + " (Score: " + str(node.score) + ")")
        print(prefix + f"Turn: {node.get_turn()}, {node.sim_drawn_lines} (Score: {node.score})")

        for child in node.children:
            self.print_tree(child, level + 1)

    def minimax(self, node, depth, isMaximizingPlayer):
        if depth == 0 or node.is_terminal_node():
            return node.score
        if isMaximizingPlayer:
            maxEval = -float('inf')
            for child in node.children:
                eval = self.minimax(child, depth - 1, False)
                maxEval = max(maxEval, eval)
            return maxEval
        else:
            minEval = float('inf')
            for child in node.children:
                eval = self.minimax(child, depth - 1, True)
                minEval = min(minEval, eval)
            return minEval

# 사용 예시
# tree = Tree(initial_data)
# root_node = tree.root
# child_node = tree.add_node(root_node, child_data)
# result = tree.minimax(root_node, depth, True)
        
# node - 위에 구현한 Node를 사용하면 될 듯 
# depth - 임의로 설정
# isMaximizingPlayer -> turn확인 후 내 turn일때 true넣자
# alpah, beta => 처음에는 각각 -무한, +무한으로 넣자



    # m = MACHINE()
    # trick_instance = m.Trick(m)

'''
    여기서부터 trick 관련 함수
    '''
class Trick:
    def __init__(self, machine_instance):
            self.machine_instance = machine_instance
            # 여기에 필요한 초기화 코드 추가
            self.just_3 = list(combinations(self.whole_points, 3))
            self.triangle_point = []                               # point of completing a triangle - (1,1), (1,4), (4,4)
            self.trick_point = []                                  # point of completing a trick - (1,1), (1,4), (4,4), (2,2)
            self.avail_trick_point = []
            self.most_completed_trick = []                       # point of completing the most completed trick
            self.trick_line = []         # line needed to complete the trick 

            
    # function of calculating triangle area
    def area(self, p1, p2, p3):
        area_val = 0.5 * abs(p1[0]*p2[1] + p2[0]*p3[1] + p3[0]*p1[1] - p1[0]*p3[1] - p3[0]*p2[1] - p2[0]*p1[1])
        return area_val
    
    # find triangle_point
    def find_triangle_point(self):
        for p in self.just_3:
            p_set = list(p)
            p1, p2, p3 = p_set[0], p_set[1], p_set[2]
            area_val = self.area(p1, p2, p3)
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
                self.triangle_point.append(p_set)       
        
    # find trick_point
    def find_trick_point(self):
        for i, p in enumerate(self.triangle_point): # 삼각형 안에 점이 1개 들어가면 trick_point에 저장
            inside_num = 0
            for q in self.whole_points: #
                if q not in p:
                    if self.area(p[0],p[1],p[2]) == self.area(p[0],p[1],q) + self.area(p[0],q,p[2]) + self.area(q,p[1],p[2]):
                        inside_num += 1
                        inside_point = q
            if inside_num == 1:
                p_copy = p.copy()  # p를 복사해서 새로운 리스트 생성
                p_copy.append(inside_point)  
                self.trick_point.append(p_copy)  
            
    # find trick_line
    def find_trick_line(self):
        i=0
        for p in self.trick_point:
            self.trick_line.append(list(combinations(p, 2)))
        
    # validate trick_point
    def validate_trick_point(self):
        for j in range(0,len(self.trick_point)): # 유효한 trick만 avail_trick_point에 저장: trick이 깨졌거나 trick에 걸리는 상황은 제외
            outer_point = list(self.trick_point[j][0:3])
            total_line = list(combinations(self.trick_point, 2))
            outside_line = list(combinations(outer_point,2))
            inside_line = [i for i in total_line if i not in outside_line]
        
            out_drawen_check = [line in self.drawn_lines_copy for line in outside_line]
            in_drawen_check = [line in self.drawn_lines_copy for line in inside_line]
            out_count = sum(out_drawen_check)
            in_count = sum(in_drawen_check)
        
            avail_candidate = list(self.trick_point[j])
        
            if out_count == 3:
                if in_count == 0 or 2:
                    self.avail_trick_point.append(avail_candidate)
            else:
                if in_count == 0:
                    self.avail_trick_point.append(avail_candidate)
        print(f"avail_trick_point is {self.avail_trick_point}")

    # # find most_completed_trick
    def find_most_completed_trick(self):
        highest_count = -1
        for j in range(0,len(self.avail_trick_point)):
            total_line = list(combinations(self.avail_trick_point[j],2))
            out_line = list(combinations(self.avail_trick_point[j][0:3],2))
            in_line = [i for i in total_line if i not in out_line]
            trick_candidate = []
        
            alredy_line = [tuple(sorted(line)) for line in self.drawn_lines_copy]
            total_line = [tuple(sorted(line)) for line in total_line]
        
            in_check = [line in alredy_line for line in in_line] ; in_count = sum(in_check)
            out_check = [line in alredy_line for line in out_line] ; out_count = sum(out_check)
            count = in_count + out_count
            
            self.first_check=0
        
            if out_count == 3:
                if in_count == 1 or in_count == 3: # 이미 선이 모드 그어졌거나 trick에 걸리는 상황 그냥 pass
                    pass
                elif in_count == 2:
                    trick_candidate.append(self.avail_trick_point[j])
                elif in_count == 0:
                    trick_candidate.append(self.avail_trick_point[j])                  
            else:
                if in_count == 0:
                    trick_candidate.append(self.avail_trick_point[j])

            for i in range(sum(len(sublist) for sublist in trick_candidate)):
                total_line = list(combinations(self.avail_trick_point[i],2))
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
