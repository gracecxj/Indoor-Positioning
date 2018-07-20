import numpy as np
X_GRID_NUM = 40
Y_GRID_NUM = 30


class Area(object):

    def __init__(self, ver_list):

        self.num_ver = len(ver_list)

        self.ver_list = ver_list
        self.min_lng = min([ver[0] for ver in self.ver_list])
        self.max_lng = max([ver[0] for ver in self.ver_list])
        self.min_lat = min([ver[1] for ver in self.ver_list])
        self.max_lat = max([ver[1] for ver in self.ver_list])

        self.coor_list = Area.lnglat_to_coordinate(ver_list)
        self.min_x = min([ver[0] for ver in self.coor_list])
        self.max_x = max([ver[0] for ver in self.coor_list])
        self.min_y = min([ver[1] for ver in self.coor_list])
        self.max_y = max([ver[1] for ver in self.coor_list])

    @staticmethod
    def lnglat_to_coordinate(ver_list):
        north_west = (55.945139, -3.18781)
        south_east = (55.944600, -3.186537)
        num_grid_x = X_GRID_NUM
        num_grid_y = Y_GRID_NUM
        delta_lng = abs(north_west[1] - south_east[1]) / num_grid_x  # 1e-05 经度
        delta_lat = abs(north_west[0] - south_east[0]) / num_grid_y  # 3e-06 纬度

        ll = []
        for vertex in ver_list:
            x = (vertex[1] - north_west[1]) // delta_lng  # 经度
            y = -(vertex[0] - north_west[0]) // delta_lat  # 纬度
            ll.append((x, y))
        return ll

    # wrong version(boundary conditions haven't been considered)
    # def coor_is_inside(self, x, y):  # x横向grid坐标，y纵向grid坐标
    #     flag = False
    #     if x < self.min_x or x > self.max_x or y < self.min_y or y > self.max_y:
    #         pass
    #     else:
    #         ring = self.coor_list
    #         ring.append(ring[0])
    #         for i in range(self.num_ver):
    #             j = i + 1
    #             # 如果测试点在i，j线段的纵坐标之间，且位于i,j确定的线之下
    #             try:
    #                 if ((y < ring[i][1]) != (y < ring[j][1])) & \
    #                     (x < (ring[j][0] - ring[i][0]) * (y - ring[i][1]) / (ring[j][1] - ring[i][1]) + ring[i][0]):
    #                     flag = not flag
    #             except ZeroDivisionError:
    #                 if y == ring[i][1]:
    #     return flag

    def coor_is_inside(self, x, y):  # x横向grid坐标，y纵向grid坐标
        flag = False
        if x < self.min_x or x > self.max_x or y < self.min_y or y > self.max_y:
            pass
        else:
            ring = self.coor_list
            ring.append(ring[0])
            for i in range(self.num_ver):
                A = ring[i]
                B = ring[i+1]

                # ---- 给定一点P,以及线段AB，判断AB是否应该使P的flag取反（即AB是否应该作用于P的flag） ----
                # Point coincides with the vertices of the polygon
                if ((x == A[0]) & (y == A[1])) | ((x == B[0]) & (y == B[1])):
                    return True

                # 线段两端点是否在射线两侧
                if ((y > A[1]) & (y <= B[1])) | ((y > B[1]) & (y <= A[1])):

                    intersect_x = A[0] + (y - A[1])* (B[0] - A[0])/(B[1] - A[1])

                    # The point on the edge of the polygon
                    if intersect_x == x:
                        return True

                    # The point is on the left side of the intersection
                    # 如果一个点P(x, y)在线段AB的纵坐标之间，且x小于P点发出的右向射线与线段AB的交点的横坐标，那么flag取反
                    if x < intersect_x:
                        flag = not flag

        return flag

    def lnglat_is_inside(self, lng, lat):  # lng经度，lat纬度
        flag = False
        if lng < self.min_lng or lng > self.max_lng or lat < self.min_lat or lat > self.max_lat:
            pass
        else:
            ring = [self.ver_list, self.ver_list[0]]
            for i in range(self.num_ver):
                j = i + 1
                # 如果测试点在i，j线段的纵坐标之间，且位于i,j确定的线之下
                if ((lat < ring[i][1]) != (lat < ring[j][1])) & \
                        (lng < (ring[j][0] - ring[i][0]) * (lat - ring[i][1]) / (ring[j][1] - ring[i][1]) + ring[i][0]):
                    flag = not flag
        return flag


def masking(points):
    '''
    this function take a list of positions as inputs, which construct a polygon.
    index the whole world rectangular space(60*80), and then return the grid index which are inside the polygon.
    :param: points - the vertex(lat, lng) of the masking area
    :return: array - the index of grids that inside a masking area
    '''

    area = Area(ver_list=points)
    array = np.zeros((Y_GRID_NUM, X_GRID_NUM))
    for i in range(X_GRID_NUM):
        for j in range(Y_GRID_NUM):
            if area.coor_is_inside(i, j):
                array[j, i] = 1
    return array

    # define the masking area(using the polygon's vertex)


# vertices = [(55.944949749302125, -3.1877678632736206),
#             (55.944643901703756, -3.1876143068075184),
#             (55.944738904347155, -3.1869940459728237),
#             (55.94483653527933, -3.187024220824241),
#             (55.94482339266817, -3.187117762863636),
#             (55.944765001870415, -3.1870899349451065),
#             (55.94469421933837, -3.1875495985150337),
#             (55.944866012262395, -3.187636099755764),
#             (55.94498992799679, -3.186802938580513),
#             (55.944825457935934, -3.186661116778851),
#             (55.94484179232254, -3.1865846738219257),
#             (55.94509957431598, -3.1867925450205803),
#             (55.94508568079271, -3.1868706643581386),
#             (55.94503949417953, -3.1868478655815125),
#             (55.944909195063346, -3.187657222151756),
#             (55.94496157761375, -3.18768136203289)]
# masking_array = masking(vertices)
# print(masking_array)
