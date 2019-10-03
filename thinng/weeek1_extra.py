import cv2import numpy as npimport random as ramap = cv2.imread("map.png")satellite = cv2.imread("satellite.png")satellite = cv2.resize(satellite, map.shape[0:2])rule_2 = [1, 2, 3, 4, 6, 7, 8, 12, 14, 15, 16, 24, 28, 30, 31, 32, 48, 56, 60, 62, 63, 64, 96, 112, 120, 124, 126, 127,          128, 129, 131, 135, 143, 159, 191, 192, 193, 195, 199, 207, 223, 224, 225, 227, 231, 239, 240, 241, 243, 247,          248, 249, 251, 252, 253, 254]kernel_N = np.array([[1, 1, 1],                     [1, 0, 1],                     [1, 1, 1]], dtype=np.uint8)kernel_sp = np.array([[128, 1, 2],                      [64, 0, 4],                      [32, 16, 8]], dtype=np.uint8)kernel_3 = np.array([[0, 1, 0],                     [0, 0, 2],                     [0, 4, 0]], dtype=np.uint8)kernel_4 = np.array([[0, 0, 0],                     [4, 0, 1],                     [0, 2, 0]], dtype=np.uint8)kernel_3_ = np.array([[0, 1, 0],                     [4, 0, 2],                     [0, 0, 0]], dtype=np.uint8)kernel_4_ = np.array([[0, 1, 0],                     [4, 0, 0],                     [0, 2, 0]], dtype=np.uint8)def extract_road(map):  # using hsv    hsv = cv2.cvtColor(map, cv2.COLOR_BGR2HSV)    h, s, v = cv2.split(hsv)    _, h = cv2.threshold(h, 1, 255, cv2.THRESH_BINARY_INV)    _, s = cv2.threshold(s, 1, 255, cv2.THRESH_BINARY_INV)    mask = cv2.bitwise_and(h, s)    mask = (mask, mask, mask)    mask = np.dstack(mask)    road = cv2.bitwise_and(map, mask)    road = cv2.cvtColor(road, cv2.COLOR_BGR2GRAY)    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))    kernel_y = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))    kernel_x = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))    road = cv2.morphologyEx(road, cv2.MORPH_OPEN, kernel)    road = cv2.morphologyEx(road, cv2.MORPH_CLOSE, kernel_x)    road = cv2.morphologyEx(road, cv2.MORPH_CLOSE, kernel_y)    return roaddef zhang_suen(road):    mask1 = np.zeros(road.shape, np.uint8)    mask2 = np.zeros(road.shape, np.uint8)    # 255 是删除 0 是保留    rule1 = cv2.filter2D(road, -1, kernel_N, borderType=cv2.BORDER_REPLICATE)    mask1 = np.ones(road.shape, np.uint8)*255    loc = np.where(rule1<2)    mask1[loc] = 0    loc = np.where(rule1>6)    mask1[loc] = 0    mask2 = cv2.bitwise_or(mask1, mask2)    mask1 = np.zeros(road.shape, np.uint8)    del rule1    rule2 = cv2.filter2D(road, -1, kernel_sp, borderType=cv2.BORDER_REPLICATE)    for i in rule_2 :        loc = np.where(rule2==i)        mask1[loc] = 255    mask2 = cv2.bitwise_and(mask1, mask2)    mask1 = np.zeros(road.shape, np.uint8)    del rule2    rule3 = cv2.filter2D(road, -1, kernel_3, borderType=cv2.BORDER_REPLICATE)    loc = np.where(rule3!=7)    mask1[loc] = 255    mask2 = cv2.bitwise_and(mask1, mask2)    mask1 = np.zeros(road.shape, np.uint8)    del rule3    rule4 = cv2.filter2D(road, -1, kernel_4, borderType=cv2.BORDER_REPLICATE)    loc = np.where(rule4!=7)    mask1[loc] = 255    mask2 = cv2.bitwise_and(mask1, mask2)    mask1 = np.zeros(road.shape, np.uint8)    del rule4    # 0 是删除， 255是保留    mask = 255-mask2    road = cv2.bitwise_and(mask,road)    mask2 = np.zeros(road.shape, np.uint8)    rule1 = cv2.filter2D(road, -1, kernel_N, borderType=cv2.BORDER_REPLICATE)    mask1 = np.ones(road.shape, np.uint8) * 255    loc = np.where(rule1 < 2)    mask1[loc] = 0    loc = np.where(rule1 > 6)    mask1[loc] = 0    mask2 = cv2.bitwise_or(mask1, mask2)    mask1 = np.zeros(road.shape, np.uint8)    del rule1    rule2 = cv2.filter2D(road, -1, kernel_sp, borderType=cv2.BORDER_REPLICATE)    for i in rule_2:        loc = np.where(rule2 == i)        mask1[loc] = 255    mask2 = cv2.bitwise_and(mask1, mask2)    mask1 = np.zeros(road.shape, np.uint8)*255    del rule2    rule3 = cv2.filter2D(road, -1, kernel_3_, borderType=cv2.BORDER_REPLICATE)    loc = np.where(rule3 != 7)    mask1[loc] = 255    mask2 = cv2.bitwise_and(mask1, mask2)    mask1 = np.zeros(road.shape, np.uint8)    del rule3    rule4 = cv2.filter2D(road, -1, kernel_4_, borderType=cv2.BORDER_REPLICATE)    loc = np.where(rule4 != 7)    mask1[loc] = 255    mask2 = cv2.bitwise_and(mask1, mask2)    mask1 = np.zeros(road.shape, np.uint8)    del rule4    mask = 255-mask2    road = cv2.bitwise_and(road,mask)    return road# 鉴于 zhang 方法之后有细小分支，而且道路多为闭合，可以将单个点滤去def filter_one(road):    mask1 = np.zeros(road.shape, np.uint8)    mask2 = np.zeros(road.shape, np.uint8)    rule1 = cv2.filter2D(road, -1, kernel_N, borderType=cv2.BORDER_REPLICATE)    mask1 = np.ones(road.shape, np.uint8) * 255    loc = np.where(rule1 == 1)    mask1[loc] = 0    mask2 = cv2.bitwise_or(mask1, mask2)    del rule1    mask = mask2    road = cv2.bitwise_and(mask, road)    return maskdef thinning(road):    canvas = road.copy()    _, road = cv2.threshold(road, 0, 1, cv2.THRESH_BINARY)    status = 1    last = 0    while status :        road = zhang_suen(road)        mask = road*255        canvas = cv2.bitwise_and(canvas,mask)        one = np.where(road==1)        new = one[0].shape[0]        print(new)        if last!=0 and abs(new-last)<10:            status = 0        last = new    cv2.imwrite("thin.jpg", canvas)    return road# binary imagedef fing_cross(road):    canvas = np.ones(road.shape, dtype=np.uint8)    gap = [85, 21, 81, 69, 84]    nodes_set = []    result = cv2.filter2D(road, -1, kernel_sp, borderType=cv2.BORDER_REPLICATE)    for i in gap_3:        y_set, x_set = np.where(result==i)        for x,y in zip(x_set,y_set):            nodes_set.append([y,x])            canvas[y,x] = 255    # delete overlap points    index = 0    while index<len(nodes_set) :        j = index + 1        y0 = nodes_set[index][0]        x0 = nodes_set[index][1]        while j<len(nodes_set):            y1 = nodes_set[j][0]            x1 = nodes_set[j][1]            dis = (y0-y1)**2 + (x0-x1)**2            if dis < 100:                del nodes_set[j]            else :                j+=1        index+=1    for y,x in nodes_set:        cv2.circle(map, (x, y), 5, (255, 0, 0), 1)    cv2.imshow("map",map)    cv2.imwrite("nodes.jpg",map)    cv2.imshow("canvas",canvas)    return nodes_setdef findpath(start_nodes, road):    road = road/255    nodes_distance = np.zeros(road.shape,np.uint32)    p = start_nodes[0]    row = road.shape[0]    col = road.shape[1]    process_list = []    path_map = np.zeros((road.shape[0],road.shape[1],2), np.uint16)    dis = 1    init = tuple([p[0], p[1]])    nodes_distance[init] = dis    process_list.append(init)    path_map[init] = p    while len(process_list)!=0:        dis += 1        print(dis)        next = []        for y,x in process_list:            if y-1>=0:                up = tuple([[y-1],[x]])                if (nodes_distance[up] == 0 or nodes_distance[up] > dis) and road[up]==1:                    if nodes_distance[up] == 0:                        next.append((y-1,x))                    nodes_distance[up] = dis                    path_map[up] = [y,x]  # 代表 (x,y-1) 点数据由 （x,y）数据更新            if y+1<=row-1:                down = tuple([[y + 1], [x]])                if (nodes_distance[down] == 0 or nodes_distance[down] > dis) and road[down]==1:                    if nodes_distance[down] == 0 :                        next.append((y+1,x))                    nodes_distance[down] = dis                    path_map[down] = [y,x]            if x-1>=0:                left = tuple([[y], [x-1]])                if (nodes_distance[left] == 0 or nodes_distance[left] > dis) and road[left]==1:                    if nodes_distance[left] == 0:                        next.append((y,x-1))                    nodes_distance[left] = dis                    path_map[left] = [y,x]            if x+1 <= col-1:                right = tuple([[y], [x+1]])                if (nodes_distance[right] == 0 or nodes_distance[right] > dis) and road[right]==1:                    if nodes_distance[right] == 0 :                        next.append((y,x+1))                    nodes_distance[right] = dis                    path_map[right] = [y,x]        process_list = next    nodes_distance_ = np.float32(nodes_distance)    nodes_distance_ = nodes_distance_/np.amax(nodes_distance)*255    nodes_distance_ = nodes_distance_.astype(np.uint8)    nodes_distance_ = cv2.applyColorMap(nodes_distance_,cv2.COLORMAP_JET)    cv2.imshow("distance",nodes_distance_)    return nodes_distance, path_mapdef draw_path(path_map, start_nodes):    index = 1    while index<len(start_nodes):        b = ra.randint(0, 255)        g = ra.randint(0, 255)        r = ra.randint(0, 255)        color = (b,g,r)        y0, x0 = start_nodes[index]        current = tuple([[y0],[x0]])        map[current] = color        satellite[current] = color        a = path_map[current]        y1 = a[0][0]        x1 = a[0][1]        next = tuple([[y1], [x1]])        while current!=next :            current = next            a = path_map[current]            y1 = a[0][0]            x1 = a[0][1]            next = tuple([[y1], [x1]])            map[current] = color            satellite[current] = color        index += 1    cv2.imshow("path",map)    cv2.imwrite("path.jpg", map)    cv2.imshow("satellite",satellite)    cv2.imwrite("satellite.jpg",satellite)    cv2.waitKey(0)    cv2.destroyAllWindows()# 枚举获得可能的情况def rule_3_val():    result = []    for i in range(256):        num = i        bit = [0] * 8        cnt = 7        up = 0        while i != 0 and cnt >= 0:            digit = i % 2            bit[cnt] = digit            i //= 2            cnt -= 1        for ind, j in enumerate(bit):            if ind == len(bit) - 1:                if bit[-1] == 0 and bit[0] == 1:                    up += 1            else:                if bit[ind] == 0 and bit[ind + 1] == 1:                    up += 1        if up > 2 :            result.append(num)    return resultgap_3 = rule_3_val()if __name__ == '__main__':    road_src = extract_road(map)    # cv2.imwrite("train_map_gray2.jpg",road)    road = thinning(road_src)    start_nodes = fing_cross(road)    nodes_distance, path_map = findpath(start_nodes,road_src)    draw_path(path_map,start_nodes)