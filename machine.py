import random
import copy  # for deepcopy
from shapely.geometry import LineString, Point, Polygon
from itertools import product, chain, combinations


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
        
        
        print(self.remaind_available_lines(self.drawn_lines)) # system이 board 정보를 업데이트 하는 시점이 machine초기화 이후이기 때문에 시작 전에 미리 알 수 없을 것 같음
        # count_available() -> len(remaind_available_lines())로 변환
        self.drawn_lines_copy = copy.deepcopy(self.drawn_lines)
        self.whole_points_copy = copy.deepcopy(self.whole_points)
        #턴마다 system에서 machine의 변수들을 업데이트 해 주는 것임
        
    def find_best_selection(self):
        # rule search로 선택
        best_selection = self.rule_search()
        
        # tree_search로 선택
        #best_selection = self.tree_search()
        return best_selection
        

    def tree_search(self):
        # 0. 가능한 모든 선분
        available = [[point1, point2] for (point1, point2) in list(combinations(self.whole_points, 2)) if self.check_availability([point1, point2])]
        # 첫 시작 : avail_lines_num이 업데이트 안되어 있을 경우 초기화
        if self.avail_lines_num == 0:
            empty = []
            self.avail_lines_num = len(self.remaind_available_lines(empty))
        

        if self.can_make_triangle(available): # 두점으로 삼각형 되면 그거 return

            return self.can_make_triangle(available)

        drawn_lines = self.drawn_lines

        remaining_lines = [line for line in available if line not in drawn_lines and self.check_availability(line)] # 후공일 경우 남아있는 라인 업데이트

        print(self.remaind_available_lines(drawn_lines= self.drawn_lines))  # 남은 가능한 라인들 반환
        
        if not remaining_lines:#남은 라인이 아무것도 없는 상태
            # RootNode Random으로 설정?
            return random.choice(available)#모든 가능한 선 중에 랜덤으로 한개의 선분 반환
        else:
            # 트리 생성 및 루트 노드에 가능한 모든 라인 추가  
            tree = Tree(drawn_lines=self.drawn_lines, max_lines_num=self.avail_lines_num)
            
            root_node = tree.root
            
            # 가능한 모든 라인을 루트 노드의 자식으로 추가
            for line in remaining_lines:
                node = tree.Node(sim_score=self.score, sim_drawn_lines=self.drawn_lines, sim_whole_points=self.whole_points, sim_triangles=self.triangles, turn = 1)
                node.sim_drawn_lines.append(line)
                root_node.add_child(node)
            
            # root를 시작으로 트리를 생성
            for child in root_node.children:
                self.populate_tree(child)

            # 트리 출력 (디버깅용)
            tree.print_tree()
            # 임시로 랜덤한 라인을 선택하여 반환
            return random.choice(remaining_lines)

    # Organization Functions
    def organize_points(self, point_list):
        point_list.sort(key=lambda x: (x[0], x[1]))
        return point_list

    def can_make_triangle(self, available):
        for line in available:
            if(self.check_if_triangle(line)):
                return line
        return None

    def populate_tree(self, node):
        # 현재 노드의 sim_drawn_lines에 부모의 drawn_lines와 현재 라인 추가
        # if node.parent and node.parent.sim_drawn_lines:
        #     node.sim_drawn_lines.extend(node.parent.sim_drawn_lines)  # extend를 사용하여 리스트의 요소를 추가

        
        # 현재 상황에서 그릴 수 있는 모든 라인을 찾아 자식 노드로 추가
        available_lines = self.sim_check_all_lines()
        for line in available_lines:
            if self.sim_check_availability(line, node.sim_drawn_lines): #내 보드에 line을 그릴 수 있나?
                child_turn = 1 if node.get_turn() == 0 else 0
                child_node = Tree.Node(sim_score=node.sim_score,sim_drawn_lines=node.sim_drawn_lines,sim_whole_points=node.sim_whole_points,sim_triangles=node.sim_triangles,turn=child_turn)
                
                child_node.check_triangle(line=line)# line을 그렸을 때, 삼각형이 되나?
                child_node.sim_drawn_lines.append(line) #line을 그리는 자식노드를 생성

                node.add_child(child_node)  # 현재 노드의 자식에 추가
                print(child_node.get_level())
                # if not child_node.is_terminal_node():
                #    self.populate_tree(child_node)
                if child_node.get_level() < 3:
                    self.populate_tree(child_node)
                
      
        #여기까지 트리파트----머신에서 트리를 생성------------------------------------------------
        #여기부터 트릭파트----------------------------------------------------------------------
        # 1. 상대방에게 점수 안 주는 전략

    # 룰 기반 트리 서치
    def rule_search(self):
        # *. Trick 피하는 전략
        fatal_lines = Trick.find_fatal_lines(self, 2, 2)
        fatal_lines = [sorted(line) for line in fatal_lines]
        print(f"fatal_lines -> {fatal_lines}")
        
        # 0. 가능한 모든 선분
        last_lines = [[point1, point2] for (point1, point2) in list(combinations(self.whole_points, 2)) if self.check_availability([point1, point2])]
        available_lines = [sorted(line) for line in last_lines]
        available_lines = [sorted(line) for line in available_lines if line not in fatal_lines]
        
        # 1. Trick 파는 전략      
        three = Trick.just_3(self)
        triangle_point = Trick.find_triangle_point(self, three)
        trick_point = Trick.find_trick_point(self, triangle_point)
        avail_trick_point = Trick.validate_trick_point(self, trick_point)
        most_completed_trick = Trick.find_most_completed_trick(self, avail_trick_point)
        best_trick_line = Trick.complete_trick(self, most_completed_trick)
        
                
        # 2. 상대방에게 점수 안 주는 전략
        # 내가 얻는 점수가 0점 일 때, 상대방이 얻는 점수도 0점인 선분 긋는 전략
        
        if self.avail_lines_num == 0:
            empty = []
            self.avail_lines_num = len(self.remaind_available_lines(empty))
        drawn_lines = self.drawn_lines
        # 이미 그어진 선분의 모든 좌표를 얻기
        drawn_points = [point for line in drawn_lines for point in line]
        #  이미 연결점이 있는 선을 제외하고, 아무것도 연결하지 않은 선의 집합 (다른 점과 연결되어 있지 않은 점 2개를 이은 선분들의 집합)
        remaining_lines = [[point1, point2] for [point1, point2] in available_lines if point1 not in drawn_points and point2 not in drawn_points]
        
        # 3. 삼각형 완성 전략
        triangle_completing_line = self.find_triangle_completing_lines()
        triangle_completing_line = [sorted(line) for line in triangle_completing_line] 
        triangle_completing_line = [line for line in triangle_completing_line if line not in fatal_lines]
         
        # 4. 짝수 삼각형 전략
        even_triangle_lines = [line for line in available_lines if self.can_form_even_number_of_triangles_after_drawing_line(line)]
        even_triangle_lines = [sorted(line) for line in even_triangle_lines] 
        even_triangle_lines = [line for line in even_triangle_lines if line not in fatal_lines]
        

        # 규칙 기반 결정 과정
        if triangle_completing_line:
            print(f"triangle_completing_line -> {triangle_completing_line}")
            return random.choice(triangle_completing_line)       
        elif best_trick_line:
            print(f"best_trick_line -> {best_trick_line}")
            return random.choice(best_trick_line)
        elif remaining_lines:
            print(f"remaining_lines -> {remaining_lines}")
            return random.choice(remaining_lines)
        elif even_triangle_lines:
            print(f"even_triangle_lines -> {even_triangle_lines}")
            return random.choice(even_triangle_lines)
        elif available_lines:
            print(f"available_lines -> {available_lines}")
            return random.choice(available_lines)  
        else:
            print(f"last_lines -> {last_lines}")
            return random.choice(last_lines)


    def find_triangle_completing_lines(self):
        """
        삼각형을 완성하는 선분을 찾는 함수. 한 번에 두 개 이상의 삼각형을 완성할 수 있는 경우를 고려한다.
        """
        max_score = 0 
        complete_lines = []
        for line in self.check_all_lines():
            #print(f"line -> {line}")
            my_score = self.obtained_score(self.drawn_lines, line)
            #print(f"my_score -> {my_score}")
            if my_score > 0:
                if my_score > max_score:
                    complete_lines = []
                    complete_lines.append(line)
                    max_score = my_score
                elif my_score == max_score:
                    complete_lines.append(line)
        return complete_lines

    
    def find_completing_line(self, triangle):
        """
        삼각형을 완성할 수 있는 선분을 찾는 함수
        """
        for line in combinations(triangle, 2):
            if line not in self.drawn_lines and self.check_availability(line):
                return line
        return None  
    
    # 삼각형의 개수
    def check_triangle_number(self, drawn_lines, whole_points):
        triangles = []
        for combination in combinations(drawn_lines, 3):
            points = set(sum(combination, []))  # 선분들의 점들을 모두 합친다.
            if len(points) == 3:  # 점이 3개면 삼각형입니다.
                triangles.append(points)
        
        # 각 삼각형이 다른 점을 포함하지 않는지 확인
        empty_triangles = []
        for triangle in triangles:
            polygon = Polygon(triangle)
            for point in whole_points:
                if polygon.contains(Point(point)) and point not in triangle:  # 삼각형의 안에 점이 있고, 그 점이 삼각형의 꼭짓점이 아니면
                    break  # 이 삼각형은 비어있지 않습니다.
            else:  # for loop가 break 없이 끝나면, 이 삼각형은 안에 점이 없음을 의미한다.
                empty_triangles.append(triangle)
        
        # '안에 점이 없는 삼각형'의 개수
        triangle_number = len(empty_triangles)
        return triangle_number
        
    def check_all_lines(self):
        #print("this is check_all_lines========================================")
        available = [[point1, point2] for (point1, point2) in list(combinations(self.whole_points, 2)) if self.check_availability([point1, point2])]
        return available
        # 조건에 부합하는 모든 가능한 라인을 반환

    def sim_check_all_lines(self):  #그냥 모든 라인을 반환
        empty = []
        available = [[point1, point2] for (point1, point2) in list(combinations(self.whole_points, 2)) if self.sim_check_availability([point1, point2], empty)]
        return available

        # system.check_triangle이용, line이 추가되면 삼각형이 만들어지는 지 T/F 반환

    def check_if_triangle(self, line):
        self.get_score = False

        point1 = line[0]
        point2 = line[1]

        point1_connected = []
        point2_connected = []

        for l in self.drawn_lines:
            if l==line: # 자기 자신 제외
                continue
            if point1 in l:
                point1_connected.append(l)
            if point2 in l:
                point2_connected.append(l)

        if point1_connected and point2_connected: # 최소한 2점 모두 다른 선분과 연결되어 있어야 함
            for line1, line2 in product(point1_connected, point2_connected):
                
                # Check if it is a triangle & Skip the triangle has occupied
                triangle = self.organize_points(list(set(chain(*[line, line1, line2]))))
                if len(triangle) != 3 or triangle in self.triangles:
                    continue

                empty = True
                for point in self.whole_points:
                    if point in triangle:
                        continue
                    if bool(Polygon(triangle).intersection(Point(point))):
                        empty = False

                if empty:   # 삼각형을 만들 수 있는 경우
                    return True
                else:
                    return False

    # count_avail_lines를 성공적으로 반환함 #len(remaind_available_lines)-> 남은 그릴 수 있는 선 수
    def remaind_available_lines(self, drawn_lines):
        #("this is remaind_available_lines========================================")
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

    # 완성된 삼각형의 개수
    def triangle_number(self, drawn_lines):   
        triangles = []
        for combination in combinations(drawn_lines, 3):
            points = set(sum(combination, []))  # 선분들의 점들을 모두 합친다.
            if len(points) == 3:  # 점이 3개면 삼각형입니다.
                triangles.append(points)
        
        # 각 삼각형이 다른 점을 포함하지 않는지 확인
        empty_triangles = []
        for triangle in triangles:
            polygon = Polygon(triangle)
            for point in self.whole_points:
                if polygon.contains(Point(point)) and point not in triangle:  # 삼각형의 안에 점이 있고, 그 점이 삼각형의 꼭짓점이 아니면
                    break  # 이 삼각형은 비어있지 않습니다.
            else:  # for loop가 break 없이 끝나면, 이 삼각형은 안에 점이 없음을 의미한다.
                empty_triangles.append(triangle)
        
        # '안에 점이 없는 삼각형'의 개수
        triangle_number = len(empty_triangles)
        return triangle_number

    
    # 선분을 그으면 얻게되는 점수 확인하는 함수
    def obtained_score(self, before_lines, selected_line):
        after_lines = before_lines.copy()
        after_lines.append(selected_line)
        
        before_triangle_number = self.triangle_number(before_lines)
        after_triangle_number = self.triangle_number(after_lines)
        draw_and_get_score = after_triangle_number - before_triangle_number
        return draw_and_get_score
        
    # 한 점과 연결된 모든 선분들 반환
    def is_point_connected(self, point):
        point_connected = [] #점에 연결된 선분들
        for l in self.drawn_lines:
            if point in l:
                point_connected.append(l)
        return point_connected


'''
    여기서부터 class Trick
    # m = MACHINE()
    # trick_instance = m.Trick(m)
'''

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
            
            #def is_point_connected(self, point):
        
            if out_count == 3:
                if in_count == 0 or in_count == 2: # out_count가 3일 때, in_count는 0 또는 2인 경우에 추가
                    avail_trick_point.append(avail_candidate)
            else:
                if in_count == 0: # out_count가 3이 아닐 때, in_count는 0인 경우에만 추가
                    avail_trick_point.append(avail_candidate)
            
            update_trick_point = []

            # 트릭 내부의 점과 트릭 외부의 점을 잇는 선분이 그어졌을 경우, 사용 가능한 트릭에서 제외
            for i in range(len(avail_trick_point)):
                connected_line = self.is_point_connected(avail_trick_point[i][3])
                trick_line = list(combinations(avail_trick_point[i], 2))
                trick_out_line = list(combinations(avail_trick_point[i][0:3],2))
                trick_in_line = [line for line in trick_line if line not in trick_out_line]
          
                arranged_connected = [sorted(line) for line in connected_line]
                arranged_trick_in = [sorted(line) for line in trick_in_line]
                critical_line = [sorted(line) for line in arranged_connected if line not in arranged_trick_in]
                if len(critical_line) > 0:
                    avail_trick_point.pop(i)                               
        return avail_trick_point


    # find most_completed_trick
    def find_most_completed_trick(self, avail_trick_point: list) -> list:
        most_completed_trick = []
        max_out_count = -1
        max_in_count = -1
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
                if in_count > max_in_count:
                    max_in_count = in_count
                    most_completed_trick = [avail_trick_point[j]]
                elif in_count == max_in_count:
                    most_completed_trick = [avail_trick_point[j]]                   
            elif max_in_count == -1:
                if out_count > max_out_count:
                    max_out_count = out_count
                    most_completed_trick = [avail_trick_point[j]]
                elif out_count == max_out_count:
                    most_completed_trick.append(avail_trick_point[j])                
        return most_completed_trick

    def complete_trick(self, most_completed_trick):
        best_trick_line = []
        # 하위 리스트를 허물어 저장할 새로운 리스트 생성
        flattened_best_trick_line = []
        for i in range(len(most_completed_trick)):
            total_line = [sorted(line) for line in combinations(most_completed_trick[i], 2)]
            out_line = [sorted(line) for line in list(combinations(most_completed_trick[i][0:3], 2))]
            in_line = [sorted(line) for line in total_line if line not in out_line]        
            already_line = [sorted(line) for line in self.drawn_lines]
            in_check = [line in already_line for line in in_line]
            out_check = [line in already_line for line in out_line]
            in_count = sum(in_check)
            out_count = sum(out_check)
            

            if out_count == 3:
                if in_count == 3 or in_count == 1:
                    pass
                else:
                    best_trick_line = [c for c in in_line if c not in self.drawn_lines]
                    best_trick_line = [line for line in best_trick_line if self.check_availability(line)]
                    
            else:
                if in_count == 0:
                    best_trick_line = [c for c in out_line if c not in self.drawn_lines]
                    best_trick_line = [line for line in best_trick_line if self.check_availability(line)]
                    best_trick_line = [sorted(line) for line in best_trick_line if not Trick.is_line_is_fatal(self, line, 1, 1)]
        return best_trick_line
    
    # 해당 선분이 트릭에 걸리는 선인지 확인하는 함수
    def is_line_is_fatal(self, selection_line, my_score_condition, other_score_limit):       
        # 내가 획득할 점수
        my_turn_lines = [sorted(line) for line in self.drawn_lines]
        my_score = self.obtained_score(my_turn_lines, selection_line)
        #print(f"my_score -> {my_score}")
            
        # 상대방이 획득 가능한 최고 점수
        other_turn_lines = copy.deepcopy(my_turn_lines)
        #other_turn_lines = my_turn_lines.copy() # 두 리스트의 메모리 공간이 같아지지 않도록 복사본 사용
        other_turn_lines.append(selection_line)
        other_turn_lines = [sorted(line) for line in other_turn_lines]
        #print(f"other_turn_lines -> {other_turn_lines}")
        #print(f"my_turn_lines -> {my_turn_lines}")
        
        total_lines = list(combinations(self.whole_points, 2)) ; total_lines = [sorted(line) for line in total_lines] # 맵에서 그을 수 있는 모든 선분
        other_avail_lines = [sorted(line) for line in total_lines if line not in other_turn_lines]
        other_avail_lines = [sorted(line) for line in other_avail_lines if self.check_availability(line)]
        #print(f"total_lines -> {total_lines}")
        
        other_max_score = 0
        for other_selection_line in other_avail_lines:       
            other_score = self.obtained_score(other_turn_lines, other_selection_line)           
            #print(f"other_selection_line -> {other_selection_line}")
            #print(f"other_score -> {other_score}")
            if other_score > other_max_score:               
                other_max_score = other_score
                #print(f"RENEW!! other_max_score -> {other_max_score}")
        
        # 상대방이 other_score_limit만큼 점수를 획득할 수 있고 내가 획득한 점수는 my_score_condition 보다 작을 때 fatal_line으로 판정              
        if my_score < my_score_condition and other_max_score >= other_score_limit:
            #print(f"other_selection_line -> {other_selection_line}")
            #print("Watch out! It is trick!!")
            #print(f"my_score: {my_score} VS other_max_score: {other_max_score}")
            return True
        else:
            return False
       
    # 트릭에 걸리는 선들을 반환하는 함수
    def find_fatal_lines(self, my_score_condition, other_score_limit):
        total_lines = list(combinations(self.whole_points, 2))
        total_lines = [sorted(line) for line in total_lines]
        avail_lines = [sorted(line) for line in total_lines if line not in self.drawn_lines]
        avail_lines = [sorted(line) for line in avail_lines if self.check_availability(line)]
        
        fatal_lines = []
        for line in avail_lines:            
            fatal_decision = Trick.is_line_is_fatal(self, line, my_score_condition, other_score_limit)            
            if fatal_decision:
                fatal_lines.append(line)
        return fatal_lines
                
                
'''
    여기서부터 class Tree & Node
'''


# minmax는 node에서 돌릴 수 있어야함
# node는 tree와는 별개인가?
# Tree클래스는 1개만 생성한다. tree 내부에 node를 계속 만들어나간다.
# Tree는 machine의 모든 인자를 가지고 있어야함

class Tree:
    max_lines_num = 0

    # class Node:
    #     def __init__(self, sim_score = [0 , 0], sim_drawn_lines = [], sim_whole_points = [], sim_triangles = [], turn = 1):
    #         self.sim_drawn_lines = [copy.deepcopy(sim_drawn_lines)] #node별로 개별 sim_drawnliens가 필요
    #         self.sim_score = sim_score   #sim_score
    #         self.sim_triangles = sim_triangles  #현재 노드에서 그려진 삼각형들
    #         self.sim_whole_points = sim_whole_points
    #         if turn == 1:
    #             self.turn = 0
    #         else:
    #             self.turn = 1
    #         self.children = []  # 그릴 수 있는 선택지들
    #         self.parent = None


    class Node:
        def __init__(self, sim_score=None, sim_drawn_lines=None, sim_whole_points=None, sim_triangles=None, turn=1):
            self.sim_score = [0, 0] if sim_score is None else copy.deepcopy(sim_score)
            self.sim_drawn_lines = copy.deepcopy(sim_drawn_lines)
            self.sim_whole_points = [] if sim_whole_points is None else copy.deepcopy(sim_whole_points)
            self.sim_triangles = [] if sim_triangles is None else copy.deepcopy(sim_triangles)
            self.turn = turn
            self.children = []
            self.parent = None
            
            
        # Score Checking Functions  #각 노드별로 check해야하니까 node클래스 내부함수로
        # line을 통해 score를 변경하고, 득점여부를 return
        def check_triangle(self, line): #line을 입력받았을때, sim_drawn_lines에 있는 선들로 삼각형이 되냐
            self.get_score = False

            point1 = line[0]    #line의 두 점
            point2 = line[1]

            point1_connected = []   #각 점이 연결된 
            point2_connected = []

        
            for l in self.sim_drawn_lines:  # 노드 상의 그려진 라인들
                if l==line: # 자기 자신 제외
                    continue
                if point1 in l:
                    point1_connected.append(l) 
                if point2 in l:
                    point2_connected.append(l)

            if point1_connected and point2_connected: # 최소한 2점 모두 다른 선분과 연결되어 있어야 함
                for line1, line2 in product(point1_connected, point2_connected):
                    
                    # Check if it is a triangle & Skip the triangle has occupied
                    triangle = self.organize_points(list(set(chain(*[line, line1, line2]))))
                    if len(triangle) != 3 or triangle in self.sim_triangles:
                        continue

                    empty = True
                    for point in self.sim_whole_points:
                        if point in triangle:
                            continue
                        if bool(Polygon(triangle).intersection(Point(point))):
                            empty = False

                    if empty:
                        self.sim_triangles.append(triangle)
                        self.sim_score[self.turn]+=1
                        self.get_score = True
            
        def get_turn(self):
            return self.turn

        def get_point(self):
            return (self.sim_score[1] - self.sim_score[0])
        
        # Organization Functions
        def organize_points(self, point_list):
            point_list.sort(key=lambda x: (x[0], x[1]))
            return point_list

        def add_child(self, child): #add_child(Node(이미 그려진 라인))
            child.parent = self
            child.turn = 1 - self.turn
            self.children.append(child)

        def get_level(self):
            level = 0
            p = self.parent
            while p:
                level += 1
                p = p.parent
            return level

        def is_terminal_node(self):
            # 터미널 노드 여부를 판단하는 로직 구현
            # 노드의 라인 수 == 처음에 판단한 총 라인수와 같아질 경우에 종료
            if self.get_level() == Tree.max_lines_num:
                return True
            else:
                return False
        


    def __init__(self, drawn_lines, max_lines_num): #tree class 만들기
        self.root = self.Node(sim_drawn_lines = drawn_lines)
        Tree.max_lines_num = max_lines_num

  
    def print_tree(self, node=None, level=0):
        if node is None:
            node = self.root
        spaces = ' ' * level * 4
        prefix = spaces + "|__ " if node.parent else ""
        #print(prefix + f"{node.sim_drawn_lines}")
        print(prefix + f"Turn: {node.get_turn()}, {node.sim_drawn_lines} (Score: {node.get_point()})")

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
