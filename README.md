# 产嗅藻荧光指纹预警与风险调控技术研究项目 — 可视化平台

基于三维荧光光谱（EEM）与机器学习的水源 2-MIB 风险快速评估平台。

| 项目 | 内容 |
|------|------|
| **项目名称** | 产嗅藻荧光指纹预警与风险调控技术研究项目 |
| **委托单位** | 宁波市奉化区水务投资发展有限公司 |
| **外部验证** | 91个现场水样，Spearman ρ=0.889，Pearson r=0.920 |

---

## 文件清单

```
可视化平台/
├── app.py                  ← Streamlit 主程序（自包含，无外部依赖路径）
├── requirements.txt        ← Python 依赖
├── README.md               ← 本文件
├── .streamlit/
│   └── config.toml         ← 界面主题配置
└── data/
    └── external_validation_results_v5.csv  ← 91个外部验证样品结果
```

> **说明**：本平台仅包含外部验证数据（93个现场样品预测结果），不含原始训练数据和荧光数据库。

---

## 方式一：本地运行

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 启动

```bash
streamlit run app.py
```

浏览器自动打开 `http://localhost:8501`，即可使用。

---

## 方式二：部署到 Streamlit Cloud（外网可访问）

### 第一步：创建 GitHub 仓库

1. 登录 https://github.com
2. 点击右上角 "+" → "New repository"
3. 仓库名：`fenghua-water-eem`，选择 **Public**
4. 点击 "Create repository"

### 第二步：上传文件

将本文件夹下所有文件上传到仓库（包括 `.streamlit/` 和 `data/` 文件夹）。

### 第三步：部署

1. 打开 https://share.streamlit.io
2. 用 GitHub 账号登录
3. 点击 "Create app" → 选择仓库 → 分支 `main` → 文件 `app.py`
4. 点击 "Deploy"

约 2~3 分钟后上线，获得永久链接：`https://fenghua-water-eem.streamlit.app`

---

## 平台功能

| 标签页 | 功能 |
|--------|------|
| 模型概览 | 技术架构、三阶段模型说明、KPI指标展示 |
| EEM分析 | 上传日立F-4600 EEM文件 → 自动前处理 → 嗅味/风险预测 |
| 外部验证 | 91个现场样品散点图、按水库/批次分析、交互式异常点剔除 |
| 模型说明 | 三阶段技术路线、训练/验证数据概况 |
| 风险预警 | 三级预警机制（安全/关注/预警）、响应建议 |

---

## 注意事项

- EEM文件格式：日立 F-4600 导出的 Tab 分隔 .TXT 文件
- 建议使用 Chrome 或 Edge 浏览器
- 手机端可访问但推荐电脑操作

---

*V2.0 | 2026年6月*
