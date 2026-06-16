# 基于大模型的复杂新闻摘要提取系统
第10组：冯俊超、李佳宸、刘景升、谢则烜

---

## 一、总述
系统以 **Qwen2.5-7B-Instruct** 为基座模型，采用 QLoRA 技术在魔塔社区开源新闻摘要数据集 **NLPCC 2017**（精筛后共 5695 条样本，按 8:1:1 划分）上进行微调。项目设计了三种微调方案，分别是**基础微调、思维链+DPO微调、任务微调**，并最终构建了一个融合本地微调模型与云端协同计算的智能新闻助手 Agent。

---

## 二、交付物说明
本项目相关的非代码交付物已单独提交（同时也已放置于本代码仓库根目录），具体内容请参阅：
*   **展示 Slides**：请参阅目录下的`Slides.pdf`。
*   **团队分工说明**：请参阅目录下的`Project Roles and Responsibilities.txt`。
*   **数据说明文档**：请参阅根目录下的 `dataset_specification.md`。
*   **评测报告**：请参阅根目录下的 `evaluation_report.pdf`。

**注**：由于 3 种微调过程中产生的训练中间状态记录以及模型权重等文件数量较多且体积较大，无法完整上传到 Github 代码仓库上，本仓库中仅上传了微调以及预测过程中的核心文件。

---

## 三、代码仓库目录结构

```text
. (项目根目录)
├── Data/                                     # 微调以及测评用到的数据集
│   ├── Base/                                 # 基础微调
│   │   ├── Base_test.jsonl                   # 测试集数据
│   │   └── Base_train_val.jsonl              # 训练与验证集数据
│   ├── Cot_Dpo/                              # 思维链+DPO微调
│   │   ├── Cot_Dpo_test.jsonl                
│   │   └── Cot_Dpo_train_val.jsonl           
│   └── Dual_task/                            # 双任务微调
│       ├── Dual_task_test.json               
│       └── Dual_task_train_val.json          
├── Eval/                                     # 评测代码和结果
│   ├── Base/                                 
│   │   ├── Base_eval.py                      # 微调后模型一键评测脚本
│   │   ├── Base_predictions.jsonl            # 微调后模型生成的预测摘要
│   │   └── Base_results.json                 # 微调后模型部分评测指标
│   ├── Cot_Dpo/                              
│   │   ├── Cot_Dpo_eval.py                   
│   │   ├── Cot_Dpo_predictions.jsonl         
│   │   └── Cot_Dpo_results.json              
│   ├── Dual_task/                            
│   │   ├── Dual_task_eval.py                 
│   │   ├── Dual_task_predictions.jsonl       
│   │   └── Dual_task_results.json            
│   └── Raw/                                  # 未微调的基座模型
│       ├── Raw_eval.py                       
│       ├── Raw_predictions.jsonl             
│       └── Raw_results.json                  
├── Infer/                                    # 推理配置参数
│   ├── Base/                                 
│   │   ├── Base_adapter_config.json          # 微调 LoRA 适配器技术细节
│   │   └── Base_infer_config.yaml            # 微调模型推理超参数配置文件
│   ├── Cot_Dpo/                              
│   │   ├── Cot_Dpo_adapter_config.json      
│   │   └── Cot_Dpo_infer_config.yaml         
│   ├── Dual_task/                            
│   │   ├── Dual_task_adapter_config.json     
│   │   └── Dual_task_infer_config.yaml       
│   └── Raw/                                  
│       └── Raw_infer_config.yaml             
├── Train/                                    # 训练记录与结果
│   ├── Base/                                 
│   │   ├── Base_train_results.json           # 训练结果
│   │   ├── Base_trainer_state.json           # 训练 Loss 记录
│   │   ├── Base_training_args.yaml           # 训练超参数配置文件
│   │   ├── Base_training_eval_loss.png       # 验证集 Loss 曲线图
│   │   └── Base_training_loss.png            # 训练集 Loss 曲线图
│   ├── Cot_Dpo/                              
│   │   ├── Cot_Dpo_train_results.json        
│   │   ├── Cot_Dpo_trainer_state.json        
│   │   ├── Cot_Dpo_training_args.yaml        
│   │   ├── Cot_Dpo_training_eval_loss.png    
│   │   └── Cot_Dpo_training_loss.png         
│   └── Dual_task/                            
│       ├── Dual_task_train_results.json      
│       ├── Dual_task_trainer_state.json      
│       ├── Dual_task_training_args.yaml      
│       ├── Dual_task_training_eval_loss.png  
│       └── Dual_task_training_loss.png       
├── .gitignore                                
├── Agent.py                                  # 交互网页构造代码
├── Dataset_specification.md                 # 数据说明文档（独立交付物）
├── Evaluation_report.pdf                     # 评测报告（独立交付物）
├── Project Roles and Responsibilities.txt    # 团队分工说明
├── README.md                                 # 项目主说明文档
├── Requirements.txt                          # Python 依赖环境配置文件
└── Slides.pdf                                # 展示Slides（独立交付物）
```

---

## 四、环境配置

本项目所有的微调和推理测试均在云服务器容器 AutoDL 中完成。

### 1. 运行服务器物理硬件与系统环境
本项目开发、离线评测与 Web 推理部署所使用的实际物理硬件配置如下：
- **操作系统**: Ubuntu 22.04.5 LTS
- **Python 版本**: 3.12.3
- **CPU 型号**: Intel(R) Xeon(R) Gold 6430
- **物理内存 (RAM)**: 1.0 Ti (可用: 867 Gi)
- **GPU 显卡型号**: NVIDIA GeForce RTX 4090 * 1
- **物理显存容量 (VRAM)**: 23.52 GB
- **CUDA 驱动版本**: 13.0

### 核心依赖安装说明
为了避免在配置依赖时意外覆写容器中预装的、已针对 4090 显卡优化的 PyTorch 环境，建议运行以下命令仅安装与核心业务、离线评测相关的第三方依赖包：

```bash
# 避免覆盖安装原有的 PyTorch 环境，仅安装其余评测与应用依赖
pip install -r Requirements.txt
```

---

## 五、模型微调复现步骤 (Train)

### 1. 微调方式与配置交付
本项目的微调实验全部在 **LLaMA-Factory Web UI** 界面中交互式完成。为了确保实验的可复现性，我们已将每次实验的全部超参数设置（包括基座模型选型 `Qwen2.5-7B-Instruct`、微调方法 `QLoRA 4-bit`、学习率、Epoch、Batch Size、梯度累积等核心超参数）完整导出为了标准的 **YAML 配置文件**。

只需根据自身的 LLaMA-Factory 环境，导入相应的配置文件并关联 `Data/` 目录下的数据集即可启动相同的微调流程。

### 2. 训练配置文件路径
- **方案一：基础微调** -> `Train/Base/Base_training_args.yaml`
- **方案二：思维链 + DPO 微调** -> `Train/Cot_Dpo/Cot_Dpo_training_args.yaml`
- **方案三：双任务微调** -> `Train/Dual_task/Dual_task_training_args.yaml`

### 3. 简要复现步骤描述
（1）**加载训练配置**：启动您的 LLaMA-Factory 环境（Web UI 界面或 CLI 工具），通过导入功能加载上述对应的 `_training_args.yaml` 配置文件，系统将自动同步我们实验时的所有超参数设置。

（2）**指定数据路径**：将训练数据集路径指向本仓库中对应的 `.jsonl` 文件（如 `Data/Base/Base_train_val.jsonl`）。

（3）**启动微调**：点击开始训练即可自动进行复现。
*(训练过程中生成的实时 Loss 变化曲线及结果数据均已自动保存在项目 `Train/` 下各方案的子目录中，供直观查阅。)*

### 4. 模型微调训练运行指标记录
以下为我们基于统一硬件环境（单张 RTX 4090 显卡），通过加载训练配置文件，运行得到的真实实验指标记录：

| 微调方案 | 训练周期 | 训练样本量 | 训练总耗时 | 训练吞吐量 | 最终 Loss |  Loss 曲线图 |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **基础微调** | 3 | 5,126 条 | 15042.13 秒<br>(约 4 小时 10 分钟) | 0.909 | 1.0738 | 见 `Train/Base/` 目录下 `.png` |
| **思维链 + DPO** | 1 | 5,126 条 | 7778.98 秒<br>(约 2 小时 10 分钟) | 0.586 | 0.6185 | 见 `Train/Cot_Dpo/` 目录下 `.png` |
| **双任务微调** | 3 | 10,252 条<br>(5,126 × 2) | 52946.84 秒<br>(约 14 小时 42 分钟) | 0.516 | 0.3247 | 见 `Train/Dual_task/` 目录下 `.png` |

**特别说明（双任务微调时间偏差原因）：**

双任务微调在实验进行期间，曾由于外部设备原因（电脑关机/休眠导致训练挂起中断，但系统计时器仍在部分累计运行时间）导致发生过一次非预期中断，随后从断点恢复训练。因此，日志中记录的 52946.84 秒包含了挂起中断期间的非微调计算时间，导致整体总时长记录明显偏高，且算出来的吞吐量（0.516）也被相应拉低。

若排除该干扰，参考相同硬件环境下标准 SFT 方案的真实吞吐量速率（约 0.909 Samples/Sec）进行去噪推估，该方案实际纯训练计算耗时应约为 **8.4 ~ 9.4 小时** 左右。

---

## 六、模型离线预测与摘要生成 (Inference)

在评测前，我们需要加载训练好的 LoRA 适配器，对 569 条离线测试集进行批量预测生成，以获取模型的输出摘要文本。

### 1. 批量预测复现说明
该步骤同样在 **LLaMA-Factory Web UI** 的 **“Evaluate & Predict”（评测与预测）** 面板中交互式完成：
- 加载对应的训练后 LoRA 权重，并导入对应的离线测试集数据（例如基础微调使用 `Data/Base/Base_test.jsonl`）。
- 启动预测后，系统会自动对测试集进行推理，并将生成的摘要文本输出并保存在 `Eval/` 各自的子目录下（如 `Eval/Base/Base_predictions.jsonl`），用于后续的离线评测。

### 2. 测试集批量预测运行指标记录
根据各方案 `Eval/` 子目录下的 `_results.json` 真实记录，我们在单张 RTX 4090 显卡下的测试集批量预测性能指标如下：

| 评估方案 | 推理总耗时 | 推理吞吐量 | 预测结果路径 |
| :--- | :---: | :---: | :--- |
| **未微调基座模型** | 715.52 秒<br>(约 11 分 56 秒) | 0.795 | `Eval/Raw/Raw_predictions.jsonl` |
| **基础微调** | 3116.92 秒<br>(约 51 分 57 秒) | 0.183 | `Eval/Base/Base_predictions.jsonl` |
| **思维链 + DPO 微调** | 665.84 秒<br>(约 11 分 06 秒) | 0.855 | `Eval/Cot_Dpo/Cot_Dpo_predictions.jsonl` |
| **双任务微调** | 937.33 秒<br>(约 15 分 37 秒) | 0.607 | `Eval/Dual_task/Dual_task_predictions.jsonl` |

**特别说明（双任务微调时间偏差原因）：**

基础微调在实验进行期间，也曾由于外部设备原因导致发生过一次非预期中断，随后从断点恢复训练。正常的推理总耗时约为 10 到 15 分钟之间。

---

## 七、离线评测与交互系统部署 (Eval & Agent)

### 1. 一键运行离线评测
本项目采用“合规性硬性约束 + 事实逻辑忠实度 + BERTScore 语义等价性 + ROUGE 词汇精准度”的综合评分体系。

运行以下一键评测脚本，系统会自动读取第六步生成的预测结果（`.jsonl`）并计算核心客观指标：

```bash
python Eval/Raw/Raw_eval.py               # 评测未微调的基座模型
python Eval/Base/Base_eval.py             # 评测基础微调模型
python Eval/Cot_Dpo/Cot_Dpo_eval.py       # 评测 Cot+Dpo 方案
python Eval/Dual_task/Dual_task_eval.py   # 评测双任务微调方案
```
具体的评测结果对比图和误差分析请参阅根目录下的独立交付报告`evaluation_report.pdf`。

### 2. 启动智能新闻助手 Agent

本项目最终交付了一个智能新闻控制台。系统核心采用**端云协同**架构，将本地部署的微调大模型与云端服务深度融合：

启动并运行控制台只需三步：

1.  开启本地模型：确保微调后的本地模型已在 8000 端口挂载启动。
2.  运行前端代码：在终端执行命令 python Agent.py。
3.  浏览器访问：打开网页 http://localhost:6008，点击界面上的“⚡ 同步最新舆情”按钮，系统便会自动抓取实时新闻并调用本地模型生成摘要。在这之后将会再可通过云端大模型对摘要分析生成当日简报，若还有进一步的探讨可以进入Copilot对当日的问题与云端大模型进一步的沟通协作。


#### 系统架构逻辑
1. **数据采集层**：通过多线程并发抓取天行舆情数据，并利用 BeautifulSoup 进行正文去噪与安全数据兜底。
2. **端侧提炼层**：通过本地推理端口（`:8000`）调用微调后的模型 `my-summarizer`，顺序生成严格限制在 60 字内的离线摘要。
3. **云端协同层**：结合本地提炼出的纯净摘要，调用云端 `qwen-plus` 接口进行全局研报提炼并提供智能 Copilot 问答。

#### 部署与启动步骤
1. 确保本地微调模型推理服务（如 vLLM）已启动，并在 `http://localhost:8000` 监听。
2. 在 `Agent.py` 顶部配置好您的 Dashscope 密钥。
3. 在根目录下运行启动命令，随后通过浏览器访问本地地址 `http://localhost:6008`（这里在AutoDL服务器时有其自动生成的对公网的地址映射所以实际不一定是这个地址）：
   ```bash
   python Agent.py