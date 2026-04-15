<p align="center">
  <h1 align="center">🏛️ 敦煌文化遗产智能知识库系统</h1>
  <p align="center">
    <strong>Dunhuang Cultural Heritage AI Knowledge Base</strong>
  </p>
  <p align="center">
    基于 RAG 架构的敦煌学智能问答与语义检索平台
  </p>
</p>

---

## 项目简介

敦煌文化遗产智能知识库系统是一个面向**敦煌学研究**的智能问答平台，旨在解决文化资源分散、传统检索方式无法理解语义关联的核心痛点。

系统收录了 **35 篇** 敦煌藻井相关学术论文，覆盖北朝至元代约 **1000 年** 的艺术演变，用户可通过自然语言对话方式，便捷获取敦煌艺术的结构化知识。每条回答均标注文献来源，确保学术严谨性。

> ⚡ **核心特性**：语义理解检索（非关键词匹配）· 来源可追溯 · 交互式界面 · 可本地部署

---

## 目标用户

| 用户类型 | 使用场景 |
|---------|---------|
| **敦煌学研究者** | 快速检索相关文献片段，辅助论文写作 |
| **艺术史学生** | 通过问答了解敦煌艺术演变脉络 |
| **博物馆/文化机构** | 为公众提供敦煌文化的互动式学习工具 |
| **AI/NLP 开发者** | 参考 RAG 架构的工程实现 |

---

## 功能说明

### 💬 智能问答
- 对话式交互界面，支持多轮追问
- 基于 RAG 架构：检索 → 上下文注入 → LLM 生成
- 每条回答标注来源文献，支持学术引用
- 首页提供引导示例，降低使用门槛

### 🔍 语义检索
- 自然语言输入，理解语义而非仅匹配关键词
- 返回相关度评分与文献片段预览
- 支持自定义返回数量（3/5/10 条）

### 📊 数据看板
- 纹样分类体系可视化（4 大类 15+ 子类）
- 各朝代藻井演变趋势图
- 敦煌矿物颜料色彩分析
- 知识库文献统计分布

### 🔧 技术特性
- **向量模型**：paraphrase-multilingual-MiniLM-L12-v2（384 维，50+ 语言）
- **检索策略**：余弦相似度 Top-K 召回，支持阈值过滤
- **索引缓存**：首次构建后持久化，后续秒级加载
- **配置分离**：YAML 集中管理，环境变量注入 API Key

---

## 技术架构

```
用户提问（自然语言）
        │
        ▼
┌───────────────────────────────────┐
│  1. 向量化（Embedding）            │  ← Sentence Transformers
│  2. 语义检索（Vector Search）      │  ← Cosine Similarity Top-K
│  3. 上下文注入（Prompt Build）     │  ← RAG Context Assembly
│  4. 生成回答（LLM）               │  ← DeepSeek / OpenAI API
└───────────────────────────────────┘
        │
        ▼
  带来源标注的准确回答
```

### 为什么用 RAG 而非纯 LLM？

| 问题 | 纯 LLM | RAG（本项目） |
|------|--------|-------------|
| 幻觉 | ❌ 可能编造不存在的文化细节 | ✅ 基于真实文献回答 |
| 知识更新 | ❌ 训练数据截止后无法更新 | ✅ 添加文献即扩展 |
| 可追溯性 | ❌ 无法验证回答来源 | ✅ 每条回答标注出处 |
| 学术适用性 | ❌ 不适合严肃学术场景 | ✅ 有据可查，支持引用 |

### 技术栈

| 模块 | 技术 | 说明 |
|------|------|------|
| 向量模型 | Sentence Transformers | multilingual-MiniLM-L12-v2 |
| 语义检索 | NumPy + Scikit-learn | 余弦相似度 |
| LLM 接口 | OpenAI SDK | 兼容 DeepSeek / 本地模型 |
| PDF 处理 | pypdf | 批量文本提取 |
| Web 界面 | Streamlit | Python 快速 UI 框架 |
| 可视化 | Plotly | 交互式图表 |
| 配置管理 | PyYAML | 集中式配置 |

---

## 安装步骤

### 环境要求

- Python 3.9+
- pip

### 快速安装

```bash
# 1. 克隆项目
git clone https://github.com/your-username/dunhuang-knowledge-base.git
cd dunhuang-knowledge-base

# 2. 创建虚拟环境（推荐）
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 3. 安装依赖
pip install -r requirements.txt

# 4. 准备数据
# 将 PDF 文献放入 data/raw/ 目录

# 5. 启动系统
python run.py
```

首次运行时，系统会自动：
1. 解析 `data/raw/` 下的所有 PDF 文件
2. 按滑动窗口（500 字/块，50 字重叠）切分文本
3. 生成向量嵌入并缓存到 `data/processed/`
4. 启动 Streamlit 交互界面

> ⏱️ 首次构建约需 3-5 分钟（取决于 PDF 数量和硬件性能），后续启动秒级加载。

### 可选：启用 AI 生成模式

默认使用检索摘要模式（无需 API Key）。如需 AI 生成完整回答：

```bash
# 方式一：环境变量
export DUNHUANG_API_KEY="your-api-key"  # macOS/Linux
set DUNHUANG_API_KEY=your-api-key       # Windows CMD

# 方式二：在 Streamlit 侧边栏输入
# 启动后在界面左下角的 API Key 输入框中填入
```

支持 DeepSeek、OpenAI 及任何 OpenAI 兼容接口。

---

## 使用方法

### 启动

```bash
python run.py
```

浏览器会自动打开 `http://localhost:8501`。

### 界面导航

| 页面 | 功能 |
|------|------|
| 🏠 首页 | 项目介绍、引导示例、数据概览 |
| 💬 智能问答 | 对话式 RAG 问答 |
| 🔍 语义检索 | 文献片段搜索 |
| 📊 数据看板 | 可视化图表 |
| ℹ️ 关于 | 项目背景与技术架构 |

### 快速体验

1. 打开首页，点击任意示例问题
2. 系统自动跳转到智能问答并填入问题
3. 查看回答及来源标注

---

## 项目结构

```
dunhuang-knowledge-base/
├── config/
│   ├── __init__.py           # 配置加载工具
│   └── settings.yaml         # 集中配置文件
├── core/
│   ├── __init__.py
│   ├── pdf_parser.py         # PDF 文本提取
│   ├── chunker.py            # 滑动窗口分块
│   ├── vectorizer.py         # 向量化 & 语义检索
│   └── rag_engine.py         # RAG 检索增强生成
├── ui/
│   ├── __init__.py
│   └── app.py                # Streamlit 主界面
├── utils/
│   ├── __init__.py
│   └── logger.py             # 日志工具
├── data/
│   ├── raw/                  # PDF 文献（需自行放入）
│   └── processed/            # 向量索引缓存（自动生成）
├── docs/                     # 项目文档
├── .gitignore
├── requirements.txt
├── run.py                    # 一键启动入口
└── README.md
```

---

## 项目截图

> 📸 使用时替换为实际截图

| 首页 | 智能问答 |
|------|---------|
| ![首页](docs/screenshots/home.png) | ![智能问答](docs/screenshots/qa.png) |

| 语义检索 | 数据看板 |
|---------|---------|
| ![语义检索](docs/screenshots/search.png) | ![数据看板](docs/screenshots/dashboard.png) |

---

## Roadmap

### ✅ 已完成（v1.0）

- [x] PDF 批量解析与文本提取
- [x] 滑动窗口分块（可配置大小与重叠）
- [x] 语义向量检索引擎（余弦相似度 Top-K）
- [x] RAG 检索增强生成（支持 OpenAI 兼容 API）
- [x] 向量索引持久化与快速加载
- [x] Streamlit 交互界面（问答 / 检索 / 可视化）
- [x] YAML 集中配置管理
- [x] 引导示例与来源标注

### 🔜 规划中（v2.0）

- [ ] 图像检索（CLIP / 以图搜图）
- [ ] 知识图谱构建（从向量升级为结构化图谱）
- [ ] 向量模型升级（bge-m3 / 领域微调）
- [ ] 多用户支持与对话历史持久化
- [ ] Docker 容器化部署
- [ ] 3D 洞窟藻井可视化（Three.js）

---

## 作者

**陈锦彤** — 独立完成系统设计、开发与实现

---

## 许可证

本项目仅供学术研究与学习交流使用。文献版权归原作者所有。

---

<p align="center">
  Built with ❤️ for Dunhuang Cultural Heritage
</p>
