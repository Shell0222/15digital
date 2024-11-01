import FunctionPackage
import pyglet

"""输入现在的8数码状态以及15数码的最终结果"""
resultStr = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '0']
#resultmatrix = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 0]]  # 15数码的最终结果
myStr = input("请【从上到下】【从左到右】输入初始状态：（用,分割）").split(',')
myStr = "1,2,3,4,5,6,7,8,0,9,10,11,13,14,15,12".split(',')
print("初始状态为：",myStr)
now_state = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
temp = 0
for i in range(4):
    for j in range(4):
        now_state[i][j] = int(myStr[temp])
        temp += 1
print(now_state)  # 输出初始状态
First_state = now_state  # 保留初始的8数码形态

print("1:BFS search")
print("2:DFS search")
print("3:A* search")
print("4:myStar1 search")
print("5:myStar2 search")
type = input("请输入搜索方式：")
if type == "1":
    """BFS搜索"""
    b = FunctionPackage.BFS(FunctionPackage.Node(now_state), FunctionPackage.Node(resultmatrix))
    print("BFS search:")
    FunctionPackage.Display(now_state, 0, 1)  # 显示现在的8数码状态
    b.bfs()
elif type == "2":
    """DFS搜索"""
    d = FunctionPackage.DFS(FunctionPackage.Node(now_state), FunctionPackage.Node(resultmatrix))
    print("DFS search:")
    FunctionPackage.Display(now_state, 0, 2)  # 显示现在的8数码状态
    d.dfs()
elif type == "3":
    """A*搜索"""
    d = FunctionPackage.A_Star(str(myStr), str(resultStr))
    print("A* search:")
    d.astar()
elif type == "4":
    """myStar1搜索"""
    d = FunctionPackage.my_Star1(str(myStr), str(resultStr))
    print("my* search:")
    d.mystar_1()
elif type == "5":
    """myStar2搜索"""
    d = FunctionPackage.my_Star2(str(myStr), str(resultStr))
    print("my** search:")
    d.mystar_2()
else:
    print("输入搜索方法有误！")

FunctionPackage.create_gif(int(type))
ag_file = "test.gif"
animation = pyglet.resource.animation(ag_file)
sprite = pyglet.sprite.Sprite(animation)
win = pyglet.window.Window(width=sprite.width, height=sprite.height)
green = 0, 1, 0, 1
pyglet.gl.glClearColor(*green)


@win.event
def on_draw():
    win.clear()
    sprite.draw()

