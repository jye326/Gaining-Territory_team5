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
        버전: 트릭
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
        
        '''
        trick 함수에 대한 설명 
        0) 다른 곳에 그어져서 trick이 깨졌는지 여부를 판단할 수 있는 장치 필요함
    
        1) most_completed_trick 구현 
            - trick_line 중에서 그려진 선분 개수 보고 가장 그려진 트릭들의 집합 저장
            - 선분 개수 같으면 append / 더 크면 기존 것 삭제 후 대입 / 작으면 pass / 개수가 4 or 6 pass
            - most_completed_trick 반환
            
        2) complete_trick 구현 
            - 일단 most_completed_trick 중에서 
            - 그리는 순서가 중요함. 앞의 3개의 점 먼저 그리기, 그 다음은 아무거나 상관 없음
        

        3) 나중에 일반화된 trick을 피하는 로직 필요함
        '''

        self.sim_drawnline = []  #  simulation시에 사용할 그려진 라인들
        self.avail_lines_num = 0 #self.count_available() #처음에 하는거 의미 없는듯
        self.sim_score = [0,0]  # 가상의 점수
        self.sim_triangles = []
        #print(self.avail_lines_num)
        #222#
        

        

    def find_best_selection(self):
        # Find available tricks
        avail_trick_point = self.find_trick()
        if avail_trick_point:  # If there are available tricks
            most_completed_trick = self.find_most_completed_trick(avail_trick_point)
            if most_completed_trick:  # If there are most completed tricks
                # Complete the trick
                best_trick_lines = self.complete_trick(most_completed_trick)
                if best_trick_lines:  # If there are best trick lines
                    # Choose a line from best trick lines
                    return random.choice(best_trick_lines)
        # If there are no tricks available, choose a line randomly from the available lines
        available = [[point1, point2] for (point1, point2) in list(combinations(self.whole_points, 2)) if self.check_availability([point1, point2])]
        return random.choice(available)

    
    def find_trick(self):
        just_3 = list(combinations(self.whole_points, 3))
        triangle_point = []                               # point of completing a triangle - (1,1), (1,4), (4,4)
        trick_point = []                                  # point of completing a trick - (1,1), (1,4), (4,4), (2,2)
        avail_trick_point = []
        most_completed_trick = []                       # point of completing the mostt completed trick
        trick_line = []                                 # line needed to complete the trick 
        
        # function of calculating triangle area
        def area(p1, p2, p3):
            area = 0.5 * abs(p1[0]*p2[1] + p2[0]*p3[1] + p3[0]*p1[1] - p1[0]*p3[1] - p3[0]*p2[1] - p2[0]*p1[1])
            return area
        
        # find triangle_point
        i=0
        for p in just_3:
            p_set = list(p)
            p1, p2, p3 = p_set[0], p_set[1], p_set[2]
            area = area(p1, p2, p3)
            if area > 0:               
                triangle_point[i] = p_set
            
        # find trick_point
        i=0
        for p in triangle_point:
            inside_num = 0         
            for q in list(combinations(self.whole_points, 1)):
                if (q in triangle_point) == False:
                    inside_point = []
                    if area(p[0],p[1],p[2]) == area(p[0],p[1],q) + area(p[0],q,p[2]) + area(q,p[1],p[2]):
                        inside_num += 1
                        inside_point = q                  
            if inside_num == 1:
                trick_point[i] = p
                trick_point[i].append(inside_point)
                
        # find trick_line
        i=0
        for p in trick_point:
            trick_line[i] = list(combinations(p, 2))
            
        # validate trick_point
        for j in range(0,len(trick_point)):
            outer_point = list(trick_point[j][0:3])
            total_line = list(combinations(trick_point, 2))
            outside_line = list(combinations(outer_point,2))
            inside_line = [i for i in total_line if i not in outside_line]
            
            out_drawen_check = [outside_line(n) for n in self.drawn_lines]
            in_drawen_check = [inside_line(n) for n in self.drawn_lines]
            out_count = sum(out_drawen_check)
            in_count = sum(in_drawen_check)
            
            avail_candidate = list(trick_point[j])
            
            if out_count == 3:
                if in_count == 0 or 2:
                    avail_trick_point.append(avail_candidate) 
            else:
                if in_count == 0:
                    avail_trick_point.append(avail_candidate)
        return avail_trick_point
    
    def find_most_completed_trick(self, avail_trick_point):
        highest_count = -1
        for j in range(0,len(avail_trick_point)):
            line = list(combinations(avail_trick_point[j], 2))           
            drawen_check = [line(n) for n in self.drawn_lines]           
            count = sum(drawen_check)
            if count > highest_count:
                most_completed_trick = list(avail_trick_point[j])
            if count == highest_count:
                most_completed_trick.append(avail_trick_point) 
        return most_completed_trick
        
    def complete_trick(self, most_completed_trick):
        best_trick_line = []
        for j in range(0, len(most_completed_trick)):
            outer_point = list(most_completed_trick[j][0:3])
            total_line = list(combinations(most_completed_trick, 2))
            outside_line = list(combinations(outer_point,2))
            inside_line = [i for i in total_line if i not in outside_line]
            
            out_drawen_check = [outside_line(n) for n in self.drawn_lines]
            in_drawen_check = [inside_line(n) for n in self.drawn_lines]
            out_count = sum(out_drawen_check)
            in_count = sum(in_drawen_check)
            if out_count < 3:
                candidate = [c for c in inside_line if c not in self.drawn_lines]
                best_trick_line.append(candidate)
            else:
                if in_count == 0 or 2:
                    candidate = [c for c in outside_line if c not in self.drawn_lines]
                    best_trick_line.append(candidate)
        return best_trick_line
                    
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
