import copy
import os
import imageio
import numpy as np
from PIL import Image
from numpy import zeros
import re
import ast

'''可视化状态'''


def create_gif(num):
    if num == 1:
        path = './BFS_result'
    elif num == 2:
        path = './DFS_result'
    elif num == 3:
        path = './Astar_result'
    elif num == 4:
        path = './myStar1'
    elif num == 5:
        path = './myStar2'
    filenames = []
    for files in os.listdir(path):
        if files.endswith('tiff'):
            file = os.path.join(path, files)
            filenames.append(file)
    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
    imageio.mimsave("test.gif", images, fps=1)


def Delete_all(num):
    if num == 1:
        file_name = "E:\\Program\\15_Digital_Issues\\15digital\\BFS_result"
    elif num == 2:
        file_name = "E:\\Program\\15_Digital_Issues\\15digital\\DFS_result"
    elif num == 3:
        file_name = "E:\\Program\\15_Digital_Issues\\15digital\\Astar_result"
    elif num == 4:
        file_name = "E:\\Program\\15_Digital_Issues\\15digital\\myStar1"
    elif num == 5:
        file_name = "E:\\Program\\15_Digital_Issues\\15digital\\myStar2"
    for root, dirs, files in os.walk(file_name):
        for name in files:
            if name.endswith(".tiff"):  # 填写规则
                os.remove(os.path.join(root, name))
                print("Delete File: " + os.path.join(root, name))


def PrintMat(myMatrix):  # 打印状态矩阵
    for i in range(4):
        for j in range(4):
            print(myMatrix[i][j], end='')
        print("")
    print("--------")
    return


def Display(myMatrix, num, type):  # 将结果可视化图片
    show_matrix = zeros((400, 400))
    for ii in range(4):
        for jj in range(4):
            temp = int(myMatrix[ii][jj])
            if temp > 0:
                im = Image.open(
                    "E:\\Program\\15_Digital_Issues\\15digital\\Digital_Images\\" + str(temp) + '.png').convert('L')
                show_matrix[100 * ii:100 * ii + 100, 100 * jj:100 * jj + 100] = np.asarray(im)
            else:
                continue
    show_im = Image.fromarray(np.uint8(show_matrix))
    if type == 1:
        path = 'BFS_result/'
    elif type == 2:
        path = 'DFS_result/'
    elif type == 3:
        path = 'Astar_result/'
    elif type == 4:
        path = 'myStar1/'
    elif type == 5:
        path = 'myStar2/'
    show_im.save(path + str(num) + '.tiff')
    # show_im.show()


def move(myMatrix, x1, y1, x2, y2):  # 移动操作
    temp = myMatrix[x1][y1]
    myMatrix[x1][y1] = myMatrix[x2][y2]
    myMatrix[x2][y2] = temp
    return myMatrix


def Check_right(myMatrix):  # 根据逆序数检查该状态是否能到达目标状态
    num = 0
    array = []
    for i in range(0, 4):
        for j in range(0, 4):
            array.append(myMatrix[i][j])
    for i in range(1, 8):
        for j in range(0, i):
            if array[i] != 0:
                if array[j] > array[i]:
                    num += 1
    return num


"""状态类"""


class Node:
    def __init__(self, myMatrix):
        self.array2d = myMatrix
        self.parent = None

    def setParent(self, Parent):
        self.parent = Parent


class BFS:
    def __init__(self, now_state, result):  # 内部变量
        self.start_state = now_state
        self.now_state = now_state
        self.result = result
        self.myOpen = []
        self.myClose = []
        self.number = 0
        self.Depth = 1
        self.type = 1

    def Check_Open(self, myList):  # 判断是否在Open表
        for i in self.myOpen:
            if i.__getattribute__("array2d") == myList.__getattribute__("array2d"):
                return True
        return False

    def Check_Close(self, myList):  # 判断是否在Close表中
        for i in self.myClose:
            if i.array2d == myList.array2d:
                return True
        return False

    def Check_ResultInOpen(self):  # 判断目标节点是否在Open表中
        for i in self.myOpen:
            if i.array2d == self.result.array2d:
                return True
        return False

    def Check_Result(self, num):  # 直接到达Open表中存在的Result节点
        for i in self.myOpen:
            '''可视化'''
            PrintMat(i.array2d)
            num += 1
            Display(i.array2d, num, self.type)
            self.number = self.number + 1
            if i.array2d == self.result.array2d:
                return i.array2d
        return None

    def get_Open(self, myList):  # 从Open中取节点
        for i in self.myOpen:
            if i.array2d == myList.array2d:
                return i
        return None

    def set_Open(self, myList):  # 将某个状态放入Open表中
        if self.Check_Close(myList):
            return
        if not self.Check_Open(myList):
            self.myOpen.append(myList)
            myList.parent = self.now_state
        return

    def set_Children(self):  # 找子节点，并放入Open表中
        global i, j
        flag = False
        for i in range(0, 4):
            for j in range(0, 4):
                if self.now_state.array2d[i][j] == 0:
                    flag = True
                    break
            if flag:
                break
        if i - 1 >= 0:
            temp = move(copy.deepcopy(self.now_state.array2d), i, j, i - 1, j)  # 向上移动
            self.set_Open(Node(temp))
        if i + 1 < 4:
            temp = move(copy.deepcopy(self.now_state.array2d), i, j, i + 1, j)  # 向下移动
            self.set_Open(Node(temp))
        if j - 1 >= 0:
            temp = move(copy.deepcopy(self.now_state.array2d), i, j, i, j - 1)  # 向左移动
            self.set_Open(Node(temp))
        if j + 1 < 4:
            temp = move(copy.deepcopy(self.now_state.array2d), i, j, i, j + 1)  # 向右移动
            self.set_Open(Node(temp))
        return

    def bfs(self):  # bfs搜索操作
        Delete_all(1)
        print("文件夹BFS_result已清空完毕！")
        nn = -1
        reverse_num1 = Check_right(self.start_state.array2d)
        reverse_num2 = Check_right(self.result.array2d)
        if (reverse_num1 - reverse_num2) % 2 == 1:
            print("行不通")
            return False
        self.myOpen.append(self.start_state)
        self.now_state = self.myOpen[0]
        result = self.Check_Result(nn)
        nn += 1
        self.myClose.append(self.now_state)
        self.myOpen.remove(self.now_state)
        self.set_Children()
        result = self.Check_Result(nn)
        nn += 1
        while True:
            self.Depth += 1
            if result:
                print("BFS搜索结果")
                print("需要搜索的节点数", self.number)
                print("深度", self.Depth)
                break
            else:
                x = len(self.myOpen)
                for i in range(0, x):
                    self.now_state = self.myOpen[0]
                    self.set_Children()
                    self.myClose.append(self.now_state)
                    self.myOpen.remove(self.myOpen[0])
                nn += 1
                result = self.Check_Result(nn)


class DFS:
    def __init__(self, now_state, result):  # 内部变量
        self.start_state = now_state
        self.now_state = now_state
        self.result = result
        self.myOpen = []
        self.myClose = []
        self.number = 0
        self.Depth = 1
        self.type = 2

    def Check_Open(self, myList):  # 判断是否在Open表
        for i in self.myOpen:
            if i.__getattribute__("array2d") == myList.__getattribute__("array2d"):
                return True
        return False

    def Check_Close(self, myList):  # 判断是否在Close表中
        for i in self.myClose:
            if i.array2d == myList.array2d:
                return True
        return False

    def Check_ResultInOpen(self):  # 判断目标节点是否在Open表中
        for i in self.myOpen:
            if i.array2d == self.result.array2d:
                return True
        return False

    def Check_Result(self, num):  # 直接到达Open表中存在的Result节点
        length = len(self.myOpen)
        for i in range(length):
            '''可视化'''
            if i == 0:
                PrintMat(self.myOpen[i].array2d)
                Display(self.myOpen[i].array2d, num, self.type)
            self.number = self.number + 1
            if self.myOpen[i].array2d == self.result.array2d:
                return self.myOpen[i].array2d
        return None

    def get_Open(self, myList):  # 从Open中取节点
        for i in self.myOpen:
            if i.array2d == myList.array2d:
                return i
        return None

    def set_Open(self, myList):  # 将某个状态放入Open表中
        if self.Check_Close(myList):
            return
        if not self.Check_Open(myList):
            self.myOpen.insert(0, myList)
        return

    def set_Children(self):  # 找子节点，并放入Open表中
        flag = False
        for i in range(0, 4):
            for j in range(0, 4):
                if self.now_state.array2d[i][j] == 0:
                    flag = True
                    break
            if flag:
                break
        if i - 1 >= 0:
            temp = move(copy.deepcopy(self.now_state.array2d), i, j, i - 1, j)  # 向上移动
            self.set_Open(Node(temp))
        if i + 1 < 4:
            temp = move(copy.deepcopy(self.now_state.array2d), i, j, i + 1, j)  # 向下移动
            self.set_Open(Node(temp))
        if j - 1 >= 0:
            temp = move(copy.deepcopy(self.now_state.array2d), i, j, i, j - 1)  # 向左移动
            self.set_Open(Node(temp))
        if j + 1 < 4:
            temp = move(copy.deepcopy(self.now_state.array2d), i, j, i, j + 1)  # 向右移动
            self.set_Open(Node(temp))
        return

    def DepthAdd(self):  # 增加深度
        self.Depth += 1

    def dfs(self):
        Delete_all(2)
        print("文件夹DFS_result已清空完毕！")
        reverse_num1 = Check_right(self.start_state.array2d)
        reverse_num2 = Check_right(self.result.array2d)
        if (reverse_num1 - reverse_num2) % 2 == 1:
            print("行不通")
            return False

        self.myOpen.insert(0, self.start_state)
        self.now_state = self.myOpen[0]
        self.myClose.append(self.now_state)
        self.myOpen.remove(self.now_state)
        self.set_Children()
        nn = 1
        result = self.Check_Result(nn)

        max_Depth = 5
        Depth_List = []
        xx = 0
        self.Depth += 1
        x = len(self.myOpen)
        for i in range(x - xx + 1):
            Depth_List.append(self.Depth)

        while True:
            xx = x
            if result:
                print("DFS搜索结果")
                print("需要搜索的节点数", self.number)
                print("深度", self.Depth)
                break
            elif xx == 0:
                print("超过搜索深度，搜索失败！")
                break
            else:
                self.DepthAdd()
                if self.Depth == max_Depth:
                    self.now_state = self.myOpen[0]
                    self.myClose.append(self.now_state)
                    self.myOpen.remove(self.myOpen[0])
                    Depth_List.pop()
                    self.Depth = Depth_List[-1]
                    x = len(self.myOpen)
                else:
                    self.now_state = self.myOpen[0]
                    self.myClose.append(self.now_state)
                    self.myOpen.remove(self.myOpen[0])
                    Depth_List.pop()
                    self.set_Children()
                    x = len(self.myOpen)
                    for i in range(x - xx + 1):
                        Depth_List.append(self.Depth)
                nn += 1
                result = self.Check_Result(nn)


class A_Star:
    def __init__(self, now_state, result):  # 内部变量
        self.type = 3
        self.start_state = now_state
        self.now_state = now_state
        self.result = result
        self.myOpen = []
        self.myClose = []
        self.Gn = {}  # 用来存储状态和对应的深度，也就是初始结点到当前结点的路径长度
        self.Fn = {}  # 用来存放状态对应的估价函数值
        self.parent = {}  # 用来存储状态对应的父结点
        # expand中存储的是九宫格中每个位置对应的可以移动的情况
        # 当定位了0的位置就可以得知可以移动的情况
        self.expand = {0: [1, 4], 1: [0, 2, 5], 2: [1, 3, 6], 3: [2, 7],
                       4: [0, 5, 8], 5: [1, 4, 6, 9], 6: [2, 5, 7, 10], 7: [3, 6, 11],
                       8: [4, 9, 12], 9: [5, 8, 10, 13], 10: [6, 9, 11, 14], 11: [7, 10, 15],
                       12: [8, 13], 13: [9, 12, 14], 14: [10, 13, 15], 15: [11, 14]}

    def THEnum(self, node):  # 计算状态对应的逆序数
        Sum = 0
        Node = re.findall(r'\d+', node)
        for i in range(len(Node)-1):
            for j in range(i + 1, len(Node)):
                if Node[i] != 0 and Node[j] != 0 and Node[i] > Node[j]:
                    Sum += 1
        empty_row = Node.index('0')
        empty_row_from_bottom = 4 - empty_row
        if (Sum % 2 == 0 and empty_row_from_bottom % 2 == 1) or \
                (Sum % 2 == 1 and empty_row_from_bottom % 2 == 0):
            return True
        return False

    def Hn(self, node):  # h(n)函数，用于计算估价函数f(n)，这里的h(n)选择的是与目标相比错位的数目
        hn = 0
        Node = re.findall(r'\d+', node)
        Result = re.findall(r'\d+', self.result)
        for i in range(0, 16):
            if Node[i] != Result[i]:
                hn += 1
        return hn

    def Expand(self, node):  # 拓展node状态对应的子结点
        tnode = []
        Node = re.findall(r'\d+', node)
        #print("Node:",node)
        state = Node.index("0")
        elist = self.expand[state]
        j = state
        for i in elist:
            j = state
            if i > j:
                i, j = j, i
            tempnode = Node[:i] + [Node[j]] + Node[i + 1:j] + [Node[i]] + Node[j + 1:]
            #print("tempnode",tempnode)
            tnode.append(str(tempnode))
        return tnode

    def PRINT(self, node):  # 将最后的结果按格式输出
        for i in range(len(node)):
            Node = re.findall(r'\d+', node[i])
            print("step--" + str(i + 1) + "|")
            print("|" + Node[0] + Node[1] + Node[2] + Node[3] + "|")
            print("|" + Node[4] + Node[5] + Node[6] + Node[7] + "|")
            print("|" + Node[8] + Node[9] + Node[10] + Node[11] + "|")
            print("|" + Node[12] + Node[13] + Node[14] + Node[15] + "|")
            mystr = Node
            mymatrix = zeros((4, 4))
            n = 0
            for ii in range(4):
                for jj in range(4):
                    mymatrix[ii][jj] = mystr[n]
                    n += 1
            Display(mymatrix, i, self.type)

    def MIN(self):  # 选择opened表中的最小的估价函数值对应的状态
        ll = {}
        for node in self.myOpen:
            k = self.Fn[node]
            ll[node] = k
        kk = min(ll, key=ll.get)
        return kk

    def astar(self):  # A*搜索
        Delete_all(3)
        print("文件夹Astar_result已清空完毕！")
        if self.start_state == self.result:
            print("初始状态和目标状态一致！")
        # 判断从初始状态是否可以达到目标状态
        if (self.THEnum == False):
            print("该目标状态不可达！")
        else:
            self.parent[self.start_state] = -1  # 初始结点的父结点存储为-1
            self.Gn[self.start_state] = 0  # 初始结点的g(n)为0
            self.Fn[self.start_state] = self.Gn[self.start_state] + self.Hn(self.start_state)  # 计算初始结点的估价函数值
            self.myOpen.append(self.start_state)  # 将初始结点存入opened表
            print(self.start_state)
            while self.myOpen:
                self.now_state = self.MIN()  # 选择估价函数值最小的状态
                del self.Fn[self.now_state]
                self.myOpen.remove(self.now_state)  # 将要遍历的结点取出opened表

                if self.now_state == self.result:
                    break
                if self.now_state not in self.myClose:
                    self.myClose.append(self.now_state)  # 存入closed表
                    Tnode = self.Expand(self.now_state)  # 扩展子结点
                    for node in Tnode:
                        # 如果子结点在opened和closed表中都未出现，则存入opened表
                        # 并求出对应的估价函数值
                        if node not in self.myOpen and node not in self.myClose:
                            self.Gn[node] = self.Gn[self.now_state] + 1
                            self.Fn[node] = self.Gn[node] + self.Hn(node)
                            self.parent[node] = self.now_state
                            self.myOpen.append(node)
                        else:
                            # 若子结点已经在opened表中，则判断估价函数值更小的一个路径
                            # 同时改变parent字典和Fn字典中的值
                            if node in self.myOpen:
                                fn = self.Gn[self.now_state] + 1 + self.Hn(node)
                                if fn < self.Fn[node]:
                                    self.Fn[node] = fn
                                    self.parent[node] = self.now_state

            result = []  # 用来存放路径
            result.append(self.now_state)
            while self.parent[self.now_state] != -1:  # 根据parent字典中存储的父结点提取路径中的结点
                self.now_state = self.parent[self.now_state]
                result.append(self.now_state)
            result.reverse()  # 逆序
            self.PRINT(result)  # 按格式输出结果


class my_Star1:
    def __init__(self, now_state, result):  # 内部变量
        self.type = 4
        self.start_state = now_state
        self.now_state = now_state
        self.result = result
        self.myOpen = []
        self.myClose = []
        self.Gn = {}  # 用来存储状态和对应的深度，也就是初始结点到当前结点的路径长度
        self.Fn = {}  # 用来存放状态对应的估价函数值
        self.parent = {}  # 用来存储状态对应的父结点
        # expand中存储的是九宫格中每个位置对应的可以移动的情况
        # 当定位了0的位置就可以得知可以移动的情况
        self.expand = {0: [1, 4], 1: [0, 2, 5], 2: [1, 3, 6], 3: [2, 7],
                       4: [0, 5, 8], 5: [1, 4, 6, 9], 6: [2, 5, 7, 10], 7: [3, 6, 11],
                       8: [4, 9, 12], 9: [5, 8, 10, 13], 10: [6, 9, 11, 14], 11: [7, 10, 15],
                       12: [8, 13], 13: [9, 12, 14], 14: [10, 13, 15], 15: [11, 14]}

    def THEnum(self, node):  # 计算状态对应的逆序数
        Sum = 0
        Node = re.findall(r'\d+', node)
        for i in range(len(Node) - 1):
            for j in range(i + 1, len(Node)):
                if Node[i] != 0 and Node[j] != 0 and Node[i] > Node[j]:
                    Sum += 1
        empty_row = Node.index('0')
        empty_row_from_bottom = 4 - empty_row
        if (Sum % 2 == 0 and empty_row_from_bottom % 2 == 1) or \
                (Sum % 2 == 1 and empty_row_from_bottom % 2 == 0):
            return True
        return False

    def GetIndex(self, node, num):
        IndexSets = [[0, 0], [0, 1], [0, 2], [0, 3],
                     [1, 0], [1, 1], [1, 2], [1, 3],
                     [2, 0], [2, 1], [2, 2], [2, 3],
                     [3, 0], [3, 1], [3, 2], [3, 3]]
        n = 0
        for i in range(16):
            if node[i] == num:
                n = i
                break
        return IndexSets[n]

    def Distance_1(self, Index_1, Index_2):  # 计算曼哈顿距离
        distance = abs(Index_1[0] - Index_2[0]) + abs(Index_1[1] - Index_2[1])
        return distance

    def Dn(self, node):  # D(n)函数，用于计算估价函数f(n)，这里的D(n)选择的是当前状态与目标状态的曼哈顿距离
        dn = 0
        for i in range(16):
            Index_1 = self.GetIndex(node, i)
            Index_2 = self.GetIndex(self.result, i)
            dn += self.Distance_1(Index_1, Index_2)
        return dn

    def Expand(self, node):  # 拓展node状态对应的子结点
        tnode = []
        Node = re.findall(r'\d+', node)
        state = Node.index("0")
        elist = self.expand[state]
        j = state
        for i in elist:
            j = state
            if i > j:
                i, j = j, i
            tempnode = Node[:i] + [Node[j]] + Node[i + 1:j] + [Node[i]] + Node[j + 1:]
            tnode.append(str(tempnode))
        return tnode

    def PRINT(self, node):  # 将最后的结果按格式输出
        for i in range(len(node)):
            Node = re.findall(r'\d+', node[i])
            print("step--" + str(i + 1) + "|")
            print("|" + Node[0] + Node[1] + Node[2] + Node[3] + "|")
            print("|" + Node[4] + Node[5] + Node[6] + Node[7] + "|")
            print("|" + Node[8] + Node[9] + Node[10] + Node[11] + "|")
            print("|" + Node[12] + Node[13] + Node[14] + Node[15] + "|")
            mystr = Node
            mymatrix = zeros((4, 4))
            n = 0
            for ii in range(4):
                for jj in range(4):
                    mymatrix[ii][jj] = mystr[n]
                    n += 1
            Display(mymatrix, i, self.type)

    def MIN(self):  # 选择opened表中的最小的估价函数值对应的状态
        ll = {}
        for node in self.myOpen:
            k = self.Fn[node]
            ll[node] = k
        kk = min(ll, key=ll.get)
        return kk

    def mystar_1(self):  # A*搜索
        Delete_all(4)
        print("文件夹myStar1已清空完毕！")
        if self.start_state == self.result:
            print("初始状态和目标状态一致！")
        # 判断从初始状态是否可以达到目标状态
        if (self.THEnum == False):
            print("该目标状态不可达！")
        else:
            self.parent[self.start_state] = -1  # 初始结点的父结点存储为-1
            self.Gn[self.start_state] = 0  # 初始结点的g(n)为0
            self.Fn[self.start_state] = self.Gn[self.start_state] + self.Dn(self.start_state)  # 计算初始结点的估价函数值
            self.myOpen.append(self.start_state)  # 将初始结点存入opened表
            print(self.start_state)
            while self.myOpen:
                self.now_state = self.MIN()  # 选择估价函数值最小的状态
                del self.Fn[self.now_state]
                self.myOpen.remove(self.now_state)  # 将要遍历的结点取出opened表

                if self.now_state == self.result:
                    break
                if self.now_state not in self.myClose:
                    self.myClose.append(self.now_state)  # 存入closed表
                    Tnode = self.Expand(self.now_state)  # 扩展子结点
                    for node in Tnode:
                        # 如果子结点在opened和closed表中都未出现，则存入opened表
                        # 并求出对应的估价函数值
                        if node not in self.myOpen and node not in self.myClose:
                            self.Gn[node] = self.Gn[self.now_state] + 1
                            w = self.Dn(node) / (self.Gn[node] + self.Dn(node)) * 2
                            self.Fn[node] = self.Gn[node] + self.Dn(node) * w
                            self.parent[node] = self.now_state
                            self.myOpen.append(node)
                        else:
                            # 若子结点已经在opened表中，则判断估价函数值更小的一个路径
                            # 同时改变parent字典和Fn字典中的值
                            if node in self.myOpen:
                                w = self.Dn(node) / (self.Gn[self.now_state] + 1 + self.Dn(node)) * 2
                                fn = self.Gn[self.now_state] + 1 + self.Dn(node) * w
                                if fn < self.Fn[node]:
                                    self.Fn[node] = fn
                                    self.parent[node] = self.now_state

            result = []  # 用来存放路径
            result.append(self.now_state)
            while self.parent[self.now_state] != -1:  # 根据parent字典中存储的父结点提取路径中的结点
                self.now_state = self.parent[self.now_state]
                result.append(self.now_state)
            result.reverse()  # 逆序
            self.PRINT(result)  # 按格式输出结果


class my_Star2:
    def __init__(self, now_state, result):  # 内部变量
        self.type = 5
        self.start_state = now_state
        self.now_state = now_state
        self.result = result
        self.myOpen = []
        self.myClose = []

        self.start_state_reverse = result
        self.now_state_reverse = result
        self.result_reverse = now_state
        self.myOpen_reverse = []
        self.myClose_reverse = []

        self.Gn = {}  # 用来存储状态和对应的深度，也就是初始结点到当前结点的路径长度
        self.Fn = {}  # 用来存放状态对应的估价函数值
        self.parent1 = {}  # 用来存储状态对应的父结点
        self.parent2 = {}  # 用来存储状态对应的父结点
        # expand中存储的是九宫格中每个位置对应的可以移动的情况
        # 当定位了0的位置就可以得知可以移动的情况
        self.expand = {0: [1, 4], 1: [0, 2, 5], 2: [1, 3, 6], 3: [2, 7],
                       4: [0, 5, 8], 5: [1, 4, 6, 9], 6: [2, 5, 7, 10], 7: [3, 6, 11],
                       8: [4, 9, 12], 9: [5, 8, 10, 13], 10: [6, 9, 11, 14], 11: [7, 10, 15],
                       12: [8, 13], 13: [9, 12, 14], 14: [10, 13, 15], 15: [11, 14]}

    def THEnum(self, node):  # 计算状态对应的逆序数
        Sum = 0
        Node = re.findall(r'\d+', node)
        for i in range(len(Node) - 1):
            for j in range(i + 1, len(Node)):
                if Node[i] != 0 and Node[j] != 0 and Node[i] > Node[j]:
                    Sum += 1
        empty_row = Node.index('0')
        empty_row_from_bottom = 4 - empty_row
        if (Sum % 2 == 0 and empty_row_from_bottom % 2 == 1) or \
                (Sum % 2 == 1 and empty_row_from_bottom % 2 == 0):
            return True
        return False

    def GetIndex(self, node, num):
        IndexSets = [[0, 0], [0, 1], [0, 2], [0, 3],
                     [1, 0], [1, 1], [1, 2], [1, 3],
                     [2, 0], [2, 1], [2, 2], [2, 3],
                     [3, 0], [3, 1], [3, 2], [3, 3]]
        n = 0
        for i in range(16):
            if node[i] == num:
                n = i
                break
        return IndexSets[n]

    def Distance_2(self, Index_1, Index_2):  # 计算欧式距离
        distance = (Index_1[0] - Index_2[0])**2 + (Index_1[1] - Index_2[1])**2
        return distance

    def Dn(self, node):  # D(n)函数，用于计算估价函数f(n)，这里的D(n)选择的是当前状态与目标状态的曼哈顿距离
        dn = 0
        for i in range(16):
            Index_1 = self.GetIndex(node, i)
            Index_2 = self.GetIndex(self.result, i)
            dn += self.Distance_2(Index_1, Index_2)
        return dn

    def Expand(self, node):  # 拓展node状态对应的子结点
        tnode = []
        Node = re.findall(r'\d+', node)
        state = Node.index("0")
        elist = self.expand[state]
        j = state
        for i in elist:
            j = state
            if i > j:
                i, j = j, i
            tempnode = Node[:i] + [Node[j]] + Node[i + 1:j] + [Node[i]] + Node[j + 1:]
            tnode.append(str(tempnode))
        return tnode

    def PRINT(self, node):  # 将最后的结果按格式输出
        for i in range(len(node)):
            Node = re.findall(r'\d+', node[i])
            print("step--" + str(i + 1) + "|")
            print("|" + Node[0] + Node[1] + Node[2] + Node[3] + "|")
            print("|" + Node[4] + Node[5] + Node[6] + Node[7] + "|")
            print("|" + Node[8] + Node[9] + Node[10] + Node[11] + "|")
            print("|" + Node[12] + Node[13] + Node[14] + Node[15] + "|")
            mystr = Node
            mymatrix = zeros((4, 4))
            n = 0
            for ii in range(4):
                for jj in range(4):
                    mymatrix[ii][jj] = mystr[n]
                    n += 1
            Display(mymatrix, i, self.type)

    def MIN(self, num):  # 选择opened表中的最小的估价函数值对应的状态
        ll = {}
        if num == 0:
            for node in self.myOpen:
                k = self.Fn[node]
                ll[node] = k
            kk = min(ll, key=ll.get)
            return kk
        else:
            for node in self.myOpen_reverse:
                k = self.Fn[node]
                ll[node] = k
            kk = min(ll, key=ll.get)
            return kk

    def Check_Open(self):
        for i in range(len(self.myOpen)):
            for j in range(len(self.myOpen_reverse)):
                if self.myOpen[i] == self.myOpen_reverse[j]:
                    return True
        return False

    def Get_SameOpen(self):
        for i in range(len(self.myOpen)):
            for j in range(len(self.myOpen_reverse)):
                if self.myOpen[i] == self.myOpen_reverse[j]:
                    return self.myOpen[i]

    def mystar_2(self):  # A**搜索
        Delete_all(5)
        print("文件夹myStar2已清空完毕！")
        if self.start_state == self.result:
            print("初始状态和目标状态一致！")
        # 判断从初始状态是否可以达到目标状态
        if (self.THEnum == False):
            print("该目标状态不可达！")
        else:
            self.parent1[self.start_state] = -1  # 初始结点的父结点存储为-1
            self.Gn[self.start_state] = 0  # 初始结点的g(n)为0
            self.Fn[self.start_state] = self.Gn[self.start_state] + self.Dn(self.start_state)  # 计算初始结点的估价函数值
            self.myOpen.append(self.start_state)  # 将初始结点存入opened表
            print("normal:")
            print(self.start_state)

            self.parent2[self.start_state_reverse] = -1  # 初始结点的父结点存储为-1
            self.Gn[self.start_state_reverse] = 0  # 初始结点的g(n)为0
            self.Fn[self.start_state_reverse] = self.Gn[self.start_state_reverse] + self.Dn(
                self.start_state_reverse)  # 计算初始结点的估价函数值
            self.myOpen_reverse.append(self.start_state_reverse)  # 将初始结点存入opened表
            print("reverse:")
            print(self.start_state_reverse)

            while ~self.Check_Open():
                self.now_state = self.MIN(0)  # 选择估价函数值最小的状态
                del self.Fn[self.now_state]
                self.myOpen.remove(self.now_state)  # 将要遍历的结点取出opened表

                if self.now_state == self.result:
                    break
                if self.now_state not in self.myClose:
                    self.myClose.append(self.now_state)  # 存入closed表
                    Tnode = self.Expand(self.now_state)  # 扩展子结点
                    for node in Tnode:
                        # 如果子结点在opened和closed表中都未出现，则存入opened表
                        # 并求出对应的估价函数值
                        if node not in self.myOpen and node not in self.myClose:
                            self.Gn[node] = self.Gn[self.now_state] + 1
                            w = self.Dn(node) / (self.Gn[node] + self.Dn(node)) * 2
                            self.Fn[node] = self.Gn[node] + self.Dn(node) * w
                            self.parent1[node] = self.now_state
                            self.myOpen.append(node)
                        else:
                            # 若子结点已经在opened表中，则判断估价函数值更小的一个路径
                            # 同时改变parent字典和Fn字典中的值
                            if node in self.myOpen:
                                w = self.Dn(node) / (self.Gn[self.now_state] + 1 + self.Dn(node)) * 2
                                fn = self.Gn[self.now_state] + 1 + self.Dn(node) * w
                                if fn < self.Fn[node]:
                                    self.Fn[node] = fn
                                    self.parent1[node] = self.now_state

                self.now_state_reverse = self.MIN(1)  # 选择估价函数值最小的状态
                del self.Fn[self.now_state_reverse]
                self.myOpen_reverse.remove(self.now_state_reverse)  # 将要遍历的结点取出opened表

                if self.now_state_reverse == self.result_reverse:
                    break
                if self.now_state_reverse not in self.myClose_reverse:
                    self.myClose_reverse.append(self.now_state_reverse)  # 存入closed表
                    Tnode = self.Expand(self.now_state_reverse)  # 扩展子结点
                    for node in Tnode:
                        # 如果子结点在opened和closed表中都未出现，则存入opened表
                        # 并求出对应的估价函数值
                        if node not in self.myOpen_reverse and node not in self.myClose_reverse:
                            self.Gn[node] = self.Gn[self.now_state_reverse] + 1
                            w = self.Dn(node) / (self.Gn[node] + self.Dn(node)) * 2
                            self.Fn[node] = self.Gn[node] + self.Dn(node) * w
                            self.parent2[node] = self.now_state_reverse
                            self.myOpen_reverse.append(node)
                        else:
                            # 若子结点已经在opened表中，则判断估价函数值更小的一个路径
                            # 同时改变parent字典和Fn字典中的值
                            if node in self.myOpen_reverse:
                                w = self.Dn(node) / (self.Gn[self.now_state_reverse] + 1 + self.Dn(node)) * 2
                                fn = self.Gn[self.now_state_reverse] + 1 + self.Dn(node) * w
                                if fn < self.Fn[node]:
                                    self.Fn[node] = fn
                                    self.parent2[node] = self.now_state_reverse

                if self.Check_Open():
                    break

            result = []  # 用来存放路径
            result.append(self.now_state)
            while self.parent1[self.now_state] != -1:  # 根据parent字典中存储的父结点提取路径中的结点
                self.now_state = self.parent1[self.now_state]
                result.append(self.now_state)
            result.reverse()  # 逆序
            same_state = self.Get_SameOpen()
            result.append(same_state)
            while self.parent2[self.now_state_reverse] != -1:
                self.now_state_reverse = self.parent2[self.now_state_reverse]
                result.append(self.now_state_reverse)
            result.append(self.result)
            self.PRINT(result)  # 按格式输出结果
