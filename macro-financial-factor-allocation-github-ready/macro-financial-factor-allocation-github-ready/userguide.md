# 完整研究程序使用说明

请发送整个项目 ZIP，解压后保留完整文件夹。Notebook、src、data 必须放在一起。建议将文件夹复制到接收者电脑有写入权限的位置。

## 使用 Anaconda 和 Jupyter

在终端或 Anaconda Prompt 中，先进入包含本说明和 environment.yml 的项目目录，然后执行：

```bash
conda env create -f environment.yml
conda activate utsg7394_crossmarket
python start_jupyter.py --check
python start_jupyter.py
```

创建环境需联网；研究计算使用随包数据，可离线运行。打开 Run_Study.ipynb，选择 **Run → Run All Cells**。启动器自动定位自己所在的文件夹，临时内核也接收该路径，避免从用户主目录启动时找不到项目。它不会安装依赖或更改 Anaconda base。

从 Navigator 启动时，选择本项目环境的 Python；若第一格报告找不到项目，将 PROJECT_DIRECTORY 填为接收者电脑的完整解压目录，不能填写 Notebook 文件名。不要复制作者电脑的路径。缺少 docx 时需要的安装包名为 python-docx。Notebook 的可选安装开关默认 False；需要时仅运行安装格，完成后重启内核，把开关改回 False，再运行全部格。

## 重建内容和查看文件

RUN_FULL_ANALYSIS=True 默认从19个原始数据文件重做10个分析阶段，再完成76个参考CSV数值核对及31项科学检查，生成8张图，不生成Word。改为False只检查已有结果并重建图。命令行也可执行：

```bash
python run_study.py
```

- Word报告请自行撰写；运行程序不需要report文件夹。
- 默认命令和Notebook均不生成Word；--skip-report保留兼容，--report-only已禁用。
- 图像：results/figures/figure1.png 至 figure8.png；图8比较三市场增量与权重距离。
- 英德配置明细：results/cross_market/ 中的预测来源、权重、收益、区间及状态支持。
- 表图来源与代码对应：documentation/report_output_map.csv。
- 数据来源及采用／排除记录：documentation/data_source_decisions.csv。
- 本次命令行复现日志：results/command_line_execution.log；完成记录：results/run_metadata.json。以后从Notebook运行时，会另行生成results/notebook_execution.log。

附录已按正文首次提及顺序编排：A术语、B英德跨市场检验、C历史IP敏感性、D规格与事件、E初始关联与RMW均值、F AI声明。正文引用及表号同步更新，并可在Word中跳转。

程序会重新计算结果和图表。报告生成模块仅作为旧版源码保留，当前入口不会调用。数值和方法变化需要完整重算，不能仅改参考CSV使核对通过。

## 结果如何解释

英德部分照搬冻结的配置规则，但保留各国已拟合的Ridge预测模型。英国Macro相对历史均值的Sharpe差约0.060，区间含零，而且低于等权；德国约−0.00029，权重几乎与基准相同。英国状态配置的六个月区块区间为负、十二个月区间含零。它们是已查看数据后的补充检验，不能称为全新独立外部验证。合成多空收益和假设成本也不等于可投资策略。

当前不生成报告的入口已在Linux x86_64／Python 3.12.14独立目录中从无results、无report文件夹的状态完整重跑，主要数值依赖采用指定版本；19个原始数据文件、76张参考表及31项科学检查均通过，生成8张图，不生成Word。随包原有数值表和图表予以保留，并与全新重建结果核对。详见documentation/reproduction_validation.json和results/command_line_execution.log。本次对Notebook的17个代码单元进行了语法检查，未另开Jupyter会话完整执行，也未实测Windows。Anaconda推荐环境仍为Python 3.13。
