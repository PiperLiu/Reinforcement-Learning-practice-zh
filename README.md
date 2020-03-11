# 你好，这是刘洪佳的强化学习笔记

*第一次系统学习强化学习，本笔记语言为中文。*

****

### 正在进行的学习内容与计划中的内容

- [ ] 强化学习圣经的第一遍学习 [[details]](#对强化学习圣经的第一遍学习)
- [ ] Deep Reinforcement Learning 的第一遍阅读 [[details]](#深度强化学习第一遍阅读)
- [ ] Approximate Dynamic Programming 的第一遍阅读 [[details]](#近似动态规划的第一遍阅读)

****

### 对强化学习圣经的第一遍学习

**输出是最好的学习，我的学习方法如下：**
- 读书，为了保证进度，我选择阅读中文版书籍[[1-2]](#参考资料)；
- 一般地，每读完一章，我会把其知识体系用自己的语言概括下来，这会引发我的很多思考：完整地将其表述出来，会弥补我读书时没有注意到的问题；
- 结合代码的笔记与心得，以 `.ipynb` 文件形式写在了[./practice/](./practice/)中，没有代码的，以 `.md` 形式写在了[./mathematics/](./mathematics/)中；
- 我会参考他人的笔记与思考，对我帮助很大的有：
- - [github.com/ShangtongZhangn 使用python复现书上案例](https://github.com/ShangtongZhang/reinforcement-learning-an-introduction)；
- - [github.com/brynhayder 对于本书的笔记，对练习题的解答](https://github.com/brynhayder/reinforcement_learning_an_introduction)。

目前已完成：

- [x] 第I部分 表格型求解方法 [学习总结 link](./mathematics/表格型方法总结.md)
- [ ] 第II部分 表格型近似求解方法
- [ ] 第III部分 表格型深入研究

学习笔记目录（所有的`.ipynb`链接已转换到`nbviewer.jupyter.org/github/`）：

##### 第I部分 表格型求解方法

- 摇臂赌博机：
- - 实例代码：[01-Stochastic-Multi-Armed-Bandit.ipynb](https://nbviewer.jupyter.org/github/PiperLiu/Reinforcement-Learning-practice-zh/blob/master/practice/01-Stochastic-Multi-Armed-Bandit.ipynb)
- - 数学公式的讨论：[梯度赌博机算法中，偏好函数更新：梯度上升公式是精确梯度上升的随机近似的证明.md](./mathematics/梯度赌博机算法中，偏好函数更新：梯度上升公式是精确梯度上升的随机近似的证明.md)
- 马尔科夫链与贝尔曼方程：
- - 实例：[02-MDP-and-Bellman-Equation.ipynb](https://nbviewer.jupyter.org/github/PiperLiu/Reinforcement-Learning-practice-zh/blob/master/practice/02-MDP-and-Bellman-Equation.ipynb)
- 动态规划：
- - 实例1：[./practice/03-01-Grid-World.ipynb](https://nbviewer.jupyter.org/github/PiperLiu/Reinforcement-Learning-practice-zh/blob/master/practice/03-01-Grid-World.ipynb)
- - 实例2：[./practice/03-02-Policy-Iteration.ipynb](https://nbviewer.jupyter.org/github/PiperLiu/Reinforcement-Learning-practice-zh/blob/master/practice/03-02-Policy-Iteration.ipynb)
- - 实例3：[./practice/03-03-Value-Iteration-and-Asynchronous-etc.ipynb](https://nbviewer.jupyter.org/github/PiperLiu/Reinforcement-Learning-practice-zh/blob/master/practice/03-03-Value-Iteration-and-Asynchronous-etc.ipynb)
- 蒙特卡洛方法：[./practice/04-Monte-Carlo-Methods.ipynb](https://nbviewer.jupyter.org/github/PiperLiu/Reinforcement-Learning-practice-zh/blob/master/practice/04-Monte-Carlo-Methods.ipynb)
- （单步）时序差分学习：
- - 评估价值部分：[./practice/05-01-Temporal-Difference-Prediction.ipynb](https://nbviewer.jupyter.org/github/PiperLiu/Reinforcement-Learning-practice-zh/blob/master/practice/05-01-Temporal-Difference-Prediction.ipynb)
- - 控制部分：[./practice/05-02-Temporal-Difference-Control.ipynb](https://nbviewer.jupyter.org/github/PiperLiu/Reinforcement-Learning-practice-zh/blob/master/practice/05-02-Temporal-Difference-Control.ipynb)
- n 步自举法：[./practice/06-N-Step-Bootstrapping.ipynb](https://nbviewer.jupyter.org/github/PiperLiu/Reinforcement-Learning-practice-zh/blob/master/practice/06-N-Step-Bootstrapping.ipynb)
- 表格型方法的规划与学习：
- - **书前八章总结：**[./mathematics/表格型方法总结.md](https://nbviewer.jupyter.org/github/PiperLiu/Reinforcement-Learning-practice-zh/blob/master/mathematics/表格型方法总结.md)
- - Dyna-Q 与 优先遍历实例：[./practice/07-01-Maze-Problem-with-DynaQ-and-Priority.ipynb](https://nbviewer.jupyter.org/github/PiperLiu/Reinforcement-Learning-practice-zh/blob/master/practice/07-01-Maze-Problem-with-DynaQ-and-Priority.ipynb)
- - 期望估计与采用估计：[./practice/07-02-Expectation-vs-Sample.ipynb](https://nbviewer.jupyter.org/github/PiperLiu/Reinforcement-Learning-practice-zh/blob/master/practice/07-02-Expectation-vs-Sample.ipynb)
- - 轨迹采样：[./practice/07-03-Trajectory-Sampling.ipynb](https://nbviewer.jupyter.org/github/PiperLiu/Reinforcement-Learning-practice-zh/blob/master/practice/07-03-Trajectory-Sampling.ipynb)

##### 第II部分 表格型近似求解方法

- 第9章：基于函数逼近的同轨策略预测：
- - 心得：[第9章：基于函数逼近的同轨策略预测.md](./mathematics/第9章：基于函数逼近的同轨策略预测.md)
- - 实例（随机游走与粗编码大小）：[./practice/On-policy-Prediction-with-Approximation.ipynb](https://nbviewer.jupyter.org/github/PiperLiu/Reinforcement-Learning-practice-zh/blob/master/practice/On-policy-Prediction-with-Approximation.ipynb)
- 第10章：基于函数逼近的同轨策略控制:
- - 心得：[第10章：基于函数逼近的同轨策略控制.md](./mathematics/第10章：基于函数逼近的同轨策略控制.md)
- - 实例（n步Sarsa控制与平均收益实例）：[./practice/Mountain-Car-Acess-Control.ipynb](https://nbviewer.jupyter.org/github/PiperLiu/Reinforcement-Learning-practice-zh/blob/master/practice/Mountain-Car-Acess-Control.ipynb)

**** 

### 深度强化学习第一遍阅读

听说这本综述不错：

[Li Y. Deep reinforcement learning: An overview[J]. arXiv preprint arXiv:1701.07274, 2017.](./resources/)

如果想看看论文与代码，可以考虑先看：

[https://github.com/ShangtongZhang/DeepRL](https://github.com/ShangtongZhang/DeepRL)

****

### 近似动态规划的第一遍阅读

在管理中，强化学习（近似动态规划）有哪些应用？老师给我推荐了这本书：

[Powell W B. Approximate Dynamic Programming: Solving the curses of dimensionality[M]. John Wiley & Sons, 2007.](./resources/)

****

### 参考资料

- [1] 强化学习（第2版）; [加拿大] Richard S. Sutton, [美国] Andrew G. Barto; 俞凯 译.
- [2] 在上述书籍出版前，有人已经开始了翻译工作：[http://rl.qiwihui.com/](http://rl.qiwihui.com/).
- [3] 英文电子原版在：[http://rl.qiwihui.com/zh_CN/latest/chapter1/introduction.html](http://rl.qiwihui.com/zh_CN/latest/chapter1/introduction.html)，已经下载到本仓库[./resources/Reinforcement Learning - An Introduction 2018.pdf](./resources/)
- [4] 强化学习读书笔记系列;公众号：老薛带你学Python(xue_python)

### 更多平台

"输出是最好的学习方式"——欢迎在其他平台查看我的学习足迹！

<a id="WeiXin"></a>
![](./doc/image/扫码_搜索联合传播样式-微信标准绿版.png)
