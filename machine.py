import random
import copy
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
        
        self.drawn_lines_copy = copy.deepcopy(drawn_lines)

    def find_best_selection(self):
        # Find available tricks
        most_completed_trick = self.find_trick()
        if most_completed_trick:  # If there are available tricks
            trick_line = self.complete_trick(most_completed_trick)
            print(f"most_completed_trick = {most_completed_trick}")
            print(f"trick line is {trick_line}")
            if trick_line:
                available = trick_line
                print(f"available is {available}")
                pass
            else:
                available = [[point1, point2] for (point1, point2) in list(combinations(self.whole_points, 2)) if self.check_availability([point1, point2])]
        return random.choice(available)

    
    def find_trick(self):
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
