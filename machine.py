import random
from itertools import combinations
from shapely.geometry import LineString, Point
from shapely.geometry import Polygon

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
        #222#
        
    def find_best_selection(self):
        # 짝수 개의 삼각형 전략을 먼저 시도
        line = self.find_even_triangle_strategy()
        if line:
            return line
        # 가능한 모든 선분 중에서 무작위로 선택
        available = [[point1, point2] for (point1, point2) in list(combinations(self.whole_points, 2)) if self.check_availability([point1, point2])]
        return random.choice(available)
    
    # 선분이 유효한지 확인
    def check_availability(self, line):
        line_string = LineString(line)

        # 선분의 양 끝점이 게임 보드의 점 중 하나여야 함
        condition1 = (line[0] in self.whole_points) and (line[1] in self.whole_points)
        
        # 선분이 중간에 다른 점을 건너뛰지 않아야 함
        condition2 = not any(line_string.intersection(Point(point)) for point in self.whole_points if point not in line)

        # 선분이 다른 선분과 교차하지 않아야 함
        condition3 = all(not line_string.intersection(LineString(l)) for l in self.drawn_lines if set(l) != set(line))

        # 선분이 이미 그려진 선분이 아니어야 함
        condition4 = line not in self.drawn_lines

        return condition1 and condition2 and condition3 and condition4

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