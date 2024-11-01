# 1 安装
请确保正确安装了以下依赖库：<br>
```pip install copy imageio numpy Pillow re ast```

# 2 运行
可在命令行或IDE中运行```python main.py```，然后选择你所需要的算法。<br>
你可以在``main.py``中修改变量``myStr``来设置初始状态，以及修改变量``resultStr``来设置目标状态。

# 3 算法
本项目提供五种解决15数码问题的算法：<br>
1. BFS：宽度优先搜索
2. DFS：深度优先搜索
3. A star：A*搜索算法
4. my star1：增加状态评估函数Dn，用于评价当前状态结点和目标状态结点之间的距离（曼哈顿距离）
5. my star2：增加状态评估函数Dn，用于评价当前状态结点和目标状态结点之间的距离（欧式距离）

# 4 结果展示
可以在``BFS_result``、``DFS_result``、``Astar_result``、``myStar1``、``myStar2``这五个文件夹下查看各个算法运行之后的每一步的结果图。<br>
在test.gif可以展示动态效果图。

# 5 总结
感谢您对本项目的关注！如有建议和问题欢迎提出！
