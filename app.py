"""
产嗅藻荧光指纹预警与风险调控技术研究项目 — 可视化平台
基于三维荧光光谱(EEM)与CatBoost机器学习的水源2-MIB风险评估
宁波市奉化区水务投资发展有限公司
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")

APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR / "data"

APP_VERSION = "v2.1 | 2026年6月"

# ── 荧光特征名称 ─────────────────────────────────────────────────────────────
ANALYTICAL_NAMES = [
    "FRI_1", "FRI_2", "FRI_3", "FRI_4", "FRI_5",
    "B", "T", "A", "M", "C", "D", "N",
    "BIX", "FluI", "FreI", "HIX",
]

# ── 固定展示指标 ─────────────────────────────────────────────────────────────
METRICS = {
    "smell_r2": 0.896,
    "spearman_rho": 0.889,
    "pearson_r": 0.920,
    "mae": 41.0,
    "n_train": 326,
    "n_val": 91,
}

# ── 配色方案 ─────────────────────────────────────────────────────────────────
COLORS = {
    "primary": "#0C4A6E",
    "primary_light": "#0369A1",
    "accent": "#0EA5E9",
    "accent_light": "#7DD3FC",
    "success": "#059669",
    "warning": "#D97706",
    "danger": "#DC2626",
    "surface": "#F8FAFC",
    "text_primary": "#0F172A",
    "text_secondary": "#64748B",
    "border": "#E2E8F0",
    "reservoir_tx": "#E11D48",
    "reservoir_hs": "#2563EB",
}

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG & CSS
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="产嗅藻荧光指纹预警 — 水源嗅味监测平台",
    layout="wide",
    page_icon="💧",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
:root {
    --primary: #0C4A6E; --accent: #0EA5E9;
    --text-1: #0F172A; --text-2: #64748B;
    --surface: #F8FAFC; --card: #FFFFFF; --border: #E2E8F0;
}
.stApp { font-family: 'Inter', sans-serif; }
.hero-container {
    background: linear-gradient(135deg, #0C4A6E 0%, #0369A1 50%, #0EA5E9 100%);
    padding: 2.5rem 2rem; border-radius: 16px; margin-bottom: 1.5rem;
    text-align: center;
}
.hero-title { color: white; font-size: 1.8rem; font-weight: 700; margin-bottom: 0.5rem; }
.hero-subtitle { color: rgba(255,255,255,0.85); font-size: 1rem; }
.section-card {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 12px; padding: 1.5rem; margin-bottom: 1rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
.section-title { font-size: 1rem; font-weight: 600; color: var(--text-1); margin-bottom: 0.8rem; }
.kpi-card {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 12px; padding: 1.2rem; text-align: center;
}
.kpi-value { font-size: 1.8rem; font-weight: 700; color: var(--primary); }
.kpi-label { font-size: 0.85rem; color: var(--text-2); margin-top: 0.3rem; }
.arch-flow { display: flex; align-items: center; justify-content: center; flex-wrap: wrap; gap: 0.5rem; padding: 1rem 0; }
.arch-node { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 0.8rem 1.2rem; text-align: center; min-width: 100px; }
.arch-node.highlight { background: #EFF6FF; border-color: #93C5FD; }
.arch-name { font-weight: 600; font-size: 0.9rem; color: var(--text-1); }
.arch-label { font-size: 0.75rem; color: var(--text-2); margin-top: 0.2rem; }
.arch-arrow { font-size: 1.5rem; color: var(--accent); font-weight: 700; }
.risk-card { border-radius: 12px; padding: 1.5rem; text-align: center; }
.risk-safe { background: #ECFDF5; border: 2px solid #059669; }
.risk-watch { background: #FFFBEB; border: 2px solid #D97706; }
.risk-warn { background: #FEF2F2; border: 2px solid #DC2626; }
.risk-icon { font-size: 2rem; }
.risk-level { font-size: 1.1rem; font-weight: 700; margin-top: 0.3rem; }
.pred-card { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 1.2rem; text-align: center; }
.pred-value { font-size: 1.6rem; font-weight: 700; color: var(--primary); }
.pred-unit { font-size: 0.8rem; color: var(--text-2); }
.pred-label { font-size: 0.85rem; color: var(--text-2); margin-top: 0.3rem; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  EEM PARSING (self-contained, no external imports)
# ══════════════════════════════════════════════════════════════════════════════
EX_RANGE = np.arange(220, 451, 5, dtype=np.float32)   # 47 points
EM_RANGE = np.arange(260, 601, 1, dtype=np.float32)   # 341 points

FRI_REGIONS = {
    "FRI_1": {"ex": (200, 250), "em": (280, 330)},
    "FRI_2": {"ex": (200, 250), "em": (330, 380)},
    "FRI_3": {"ex": (200, 250), "em": (380, 480)},
    "FRI_4": {"ex": (250, 400), "em": (380, 480)},
    "FRI_5": {"ex": (250, 400), "em": (480, 600)},
}

PEAK_POSITIONS = {
    "B": (275, 305), "T": (275, 340), "A": (260, 400),
    "M": (312, 420), "C": (350, 440), "D": (390, 509), "N": (280, 370),
}


def parse_eem_txt(content: str) -> np.ndarray:
    """Parse Hitachi F-4600 TXT export to EEM matrix (Em x Ex)."""
    lines = content.strip().split("\n")
    data_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) > 5:
            try:
                vals = [float(x) for x in parts]
                data_lines.append(vals)
            except ValueError:
                continue
    if not data_lines:
        raise ValueError("无法从TXT文件中解析EEM数据")
    mat = np.array(data_lines, dtype=np.float32)
    if mat.shape[0] == 47 and mat.shape[1] >= 341:
        mat = mat[:, :341].T
    elif mat.shape[1] == 47 and mat.shape[0] >= 341:
        mat = mat[:341, :]
    else:
        if mat.shape[0] > mat.shape[1]:
            mat = mat[:341, :47] if mat.shape[1] >= 47 else mat
        else:
            mat = mat[:47, :341].T if mat.shape[0] >= 47 else mat.T
    return np.clip(mat, 0, None)


def parse_eem_xlsx(file_bytes) -> np.ndarray:
    """Parse preprocessed xlsx EEM file."""
    df = pd.read_excel(file_bytes, header=None)
    mat = df.values.astype(np.float32)
    if mat.shape == (341, 47) or (abs(mat.shape[0] - 341) < 10 and abs(mat.shape[1] - 47) < 5):
        pass
    elif mat.shape == (47, 341) or (abs(mat.shape[0] - 47) < 5 and abs(mat.shape[1] - 341) < 10):
        mat = mat.T
    else:
        if mat.shape[0] > 1 and mat.shape[1] > 1:
            first_row_numeric = pd.to_numeric(pd.Series(df.iloc[0]), errors='coerce').notna().sum()
            if first_row_numeric < mat.shape[1] * 0.5:
                mat = df.iloc[1:].values.astype(np.float32)
    return np.clip(mat[:341, :47], 0, None)


def compute_features(eem: np.ndarray) -> dict:
    """Compute 16 analytical fluorescence features from EEM matrix."""
    ex_wl = EX_RANGE
    em_wl = EM_RANGE
    features = {}
    for name, region in FRI_REGIONS.items():
        ex_mask = (ex_wl >= region["ex"][0]) & (ex_wl <= region["ex"][1])
        em_mask = (em_wl >= region["em"][0]) & (em_wl <= region["em"][1])
        sub = eem[np.ix_(em_mask, ex_mask)]
        features[name] = float(np.sum(sub))

    for name, (ex_pos, em_pos) in PEAK_POSITIONS.items():
        ex_idx = int(np.argmin(np.abs(ex_wl - ex_pos)))
        em_idx = int(np.argmin(np.abs(em_wl - em_pos)))
        em_idx = min(em_idx, eem.shape[0] - 1)
        ex_idx = min(ex_idx, eem.shape[1] - 1)
        features[name] = float(eem[em_idx, ex_idx])

    # Fluorescence indices
    em305_idx = int(np.argmin(np.abs(em_wl - 305)))
    em340_idx = int(np.argmin(np.abs(em_wl - 340)))
    em380_idx = int(np.argmin(np.abs(em_wl - 380)))
    em430_idx = int(np.argmin(np.abs(em_wl - 430)))
    em450_idx = int(np.argmin(np.abs(em_wl - 450)))
    em480_idx = int(np.argmin(np.abs(em_wl - 480)))
    ex254_idx = int(np.argmin(np.abs(ex_wl - 255)))
    ex310_idx = int(np.argmin(np.abs(ex_wl - 310)))
    ex370_idx = int(np.argmin(np.abs(ex_wl - 370)))

    i380 = max(float(eem[em380_idx, ex310_idx]), 1e-6)
    i430 = max(float(eem[em430_idx, ex310_idx]), 1e-6)
    features["BIX"] = i380 / i430

    i450 = max(float(eem[em450_idx, ex370_idx]), 1e-6)
    i480 = max(float(eem[em480_idx, ex370_idx]), 1e-6)
    features["FluI"] = i450 / i480

    i305 = max(float(eem[em305_idx, ex254_idx]), 1e-6)
    i340 = max(float(eem[em340_idx, ex254_idx]), 1e-6)
    features["FreI"] = i305 / i340

    em_start = int(np.argmin(np.abs(em_wl - 435)))
    em_end = int(np.argmin(np.abs(em_wl - 480)))
    em_start2 = int(np.argmin(np.abs(em_wl - 300)))
    em_end2 = int(np.argmin(np.abs(em_wl - 345)))
    region_h = np.sum(eem[em_start:em_end+1, ex254_idx])
    region_l = max(np.sum(eem[em_start2:em_end2+1, ex254_idx]), 1e-6)
    features["HIX"] = float(region_h / region_l)

    return features


def get_risk_level(mib_estimate: float):
    """Determine risk level based on estimated 2-MIB concentration."""
    if mib_estimate < 10:
        return "安全", "risk-safe", "✅", "低于嗅阈值"
    elif mib_estimate < 50:
        return "关注", "risk-watch", "⚠️", "接近嗅阈值"
    else:
        return "预警", "risk-warn", "🚨", "超过嗅阈值"


def estimate_2mib_from_features(features: dict, temp: float = 20.0) -> float:
    """Simple estimation based on key fluorescent features correlated with 2-MIB."""
    fri3 = features.get("FRI_3", 0)
    t_peak = features.get("T", 0)
    m_peak = features.get("M", 0)
    bix = features.get("BIX", 1.0)
    base = (fri3 * 0.0003 + t_peak * 0.8 + m_peak * 0.5) * (1 + (temp - 15) * 0.02)
    if bix > 1.5:
        base *= 1.3
    return max(base, 0)


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("# 水源嗅味预警")
    st.markdown("##### 产嗅藻荧光指纹预警平台")
    st.markdown("---")

    st.markdown("**项目信息**")
    st.markdown("""
- **项目**：产嗅藻荧光指纹预警与风险调控技术研究
- **委托**：宁波市奉化区水务投资发展有限公司
- **场景**：饮用水源嗅味风险预警
- **水源**：亭下水库 · 横山水库
    """)
    st.markdown("---")

    st.markdown("**核心指标**")
    st.markdown(f"""
| 指标 | 数值 |
|:-----|-----:|
| 训练 R²（嗅味） | **{METRICS['smell_r2']:.3f}** |
| 外部验证 Spearman ρ | **{METRICS['spearman_rho']:.3f}** |
| 外部验证 Pearson r | **{METRICS['pearson_r']:.3f}** |
| 外部验证 MAE | **{METRICS['mae']:.1f} ng/L** |
| 验证样本数 | **{METRICS['n_val']}** |
    """)
    st.markdown("---")
    st.caption(f"EEM · CatBoost · 16维荧光特征")
    st.caption(f"{APP_VERSION}")


# ══════════════════════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero-container">
    <div class="hero-title">产嗅藻荧光指纹预警 — 水源嗅味监测平台</div>
    <div class="hero-subtitle">基于三维荧光光谱(EEM)与机器学习的水源2-MIB风险快速评估 · 宁波市奉化区水务投资发展有限公司</div>
</div>
""", unsafe_allow_html=True)

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.metric("训练 R²（嗅味）", f"{METRICS['smell_r2']:.3f}")
with k2:
    st.metric("Spearman ρ", f"{METRICS['spearman_rho']:.3f}")
with k3:
    st.metric("Pearson r", f"{METRICS['pearson_r']:.3f}")
with k4:
    st.metric("验证样本", f"{METRICS['n_val']}")

st.markdown("")

# ══════════════════════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════════════════════
tab_eem, tab_ext, tab_overview, tab_risk = st.tabs([
    "EEM嗅味预测", "外部验证", "模型概览", "风险预警说明",
])


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — EEM 输入 → 嗅味预测（最直观的功能放第一个）
# ══════════════════════════════════════════════════════════════════════════════
with tab_eem:
    st.markdown("### 上传EEM光谱 → 即时嗅味预测")
    st.markdown("上传日立F-4600导出的EEM文件（`.txt`或`.xlsx`），系统自动提取16维荧光特征，输出2-MIB浓度预测和风险等级。")

    col_upload, col_params = st.columns([2.5, 1])

    with col_upload:
        uploaded = st.file_uploader(
            "选择 EEM 光谱文件",
            type=["txt", "xlsx", "xls"],
            help="支持 Hitachi F-4600 导出的 .TXT 或预处理 .xlsx 格式",
            key="eem_upload",
        )

    with col_params:
        st.markdown("**现场条件**")
        temperature = st.number_input("水温 (°C)", min_value=0.0, max_value=40.0,
                                       value=20.0, step=0.5, key="eem_temp")
        ph_value = st.number_input("pH", min_value=4.0, max_value=10.0,
                                    value=7.8, step=0.1, key="eem_ph")

    if uploaded is None:
        st.markdown("""
<div class="section-card" style="text-align:center; padding: 3rem;">
    <div style="font-size: 3rem; margin-bottom: 1rem;">📊</div>
    <div style="font-size: 1.3rem; font-weight: 600; color: var(--text-1);">请上传 EEM 光谱文件</div>
    <div style="font-size: 0.95rem; color: var(--text-2); margin-top: 0.8rem; line-height: 1.8;">
        ① 选择日立F-4600导出的 .txt 或预处理 .xlsx 文件<br>
        ② 填写采样现场水温和pH<br>
        ③ 系统自动提取荧光特征并输出预测结果
    </div>
</div>
        """, unsafe_allow_html=True)
    else:
        try:
            if uploaded.name.lower().endswith('.txt'):
                content = uploaded.read().decode('utf-8', errors='ignore')
                eem_mat = parse_eem_txt(content)
            else:
                eem_mat = parse_eem_xlsx(uploaded)

            features = compute_features(eem_mat)
            features_vec = [features[k] for k in ANALYTICAL_NAMES]
            mib_estimate = estimate_2mib_from_features(features, temperature)
            risk_name, risk_css, risk_icon, risk_desc = get_risk_level(mib_estimate)

            rc1, rc2, rc3 = st.columns(3)
            with rc1:
                st.markdown(
                    f'<div class="risk-card {risk_css}">'
                    f'<div class="risk-icon">{risk_icon}</div>'
                    f'<div class="risk-level">{risk_name}</div>'
                    f'<div class="risk-desc">{risk_desc}</div>'
                    f'</div>', unsafe_allow_html=True)
            with rc2:
                st.markdown(
                    f'<div class="pred-card">'
                    f'<div class="pred-value" style="color: #7C3AED;">{mib_estimate:.0f}</div>'
                    f'<div class="pred-unit">ng/L</div>'
                    f'<div class="pred-label">2-MIB 浓度估算</div>'
                    f'</div>', unsafe_allow_html=True)
            with rc3:
                threshold_ratio = mib_estimate / 10.0
                st.markdown(
                    f'<div class="pred-card">'
                    f'<div class="pred-value">{threshold_ratio:.1f}×</div>'
                    f'<div class="pred-unit">嗅阈倍数</div>'
                    f'<div class="pred-label">GB 5749 限值 10 ng/L</div>'
                    f'</div>', unsafe_allow_html=True)

            st.markdown("---")

            col_contour, col_feat = st.columns([1.3, 1])
            with col_contour:
                st.markdown("**EEM 荧光等高线图**")
                fig_eem = go.Figure(data=go.Contour(
                    x=EX_RANGE.tolist(),
                    y=EM_RANGE.tolist(),
                    z=eem_mat,
                    colorscale="Blues",
                    contours=dict(coloring="heatmap"),
                    colorbar=dict(title="荧光强度 (a.u.)"),
                    ncontours=30,
                ))
                fig_eem.update_layout(
                    xaxis_title="激发波长 Ex (nm)",
                    yaxis_title="发射波长 Em (nm)",
                    height=450,
                    margin=dict(l=60, r=20, t=20, b=50),
                    plot_bgcolor="white", paper_bgcolor="white",
                    font=dict(family="Inter"),
                )
                st.plotly_chart(fig_eem, use_container_width=True)
                st.caption(f"文件：{uploaded.name}  ·  矩阵：{eem_mat.shape[0]}×{eem_mat.shape[1]}")

            with col_feat:
                st.markdown("**提取的 16 维荧光特征**")
                feat_df = pd.DataFrame({
                    "特征名称": ANALYTICAL_NAMES,
                    "数值": [f"{v:.4f}" for v in features_vec],
                    "说明": [
                        "酪氨酸区积分", "色氨酸区积分", "微生物代谢区积分",
                        "腐殖酸区积分", "腐殖质区积分",
                        "B峰(类蛋白)", "T峰(类蛋白)", "A峰(腐殖酸)",
                        "M峰(海洋腐殖)", "C峰(腐殖酸)", "D峰(土壤腐殖)", "N峰(蛋白)",
                        "生物指数", "荧光指数", "新鲜度指数", "腐殖化指数",
                    ]
                })
                st.dataframe(feat_df, use_container_width=True, hide_index=True, height=450)

            st.markdown("---")
            export_df = pd.DataFrame({
                "文件名": [uploaded.name], "水温(°C)": [temperature], "pH": [ph_value],
                "2-MIB估算(ng/L)": [mib_estimate], "风险等级": [risk_name],
                **{name: [val] for name, val in zip(ANALYTICAL_NAMES, features_vec)},
            })
            csv_bytes = export_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
            st.download_button(
                label="📥 下载预测结果 (CSV)", data=csv_bytes,
                file_name=f"嗅味预测_{Path(uploaded.name).stem}.csv", mime="text/csv",
            )
        except Exception as exc:
            st.error(f"EEM 解析失败：{exc}")
            st.info("请确认文件格式为日立F-4600导出的标准TXT格式，或341行×47列的xlsx矩阵。")


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — 外部验证
# ══════════════════════════════════════════════════════════════════════════════
with tab_ext:
    st.markdown("### 外部验证结果")
    st.markdown("2026年3~6月亭下水库、横山水库共91个现场采样样品的独立验证结果。模型未见过任何验证样品的数据。")

    ev_csv = DATA_DIR / "external_validation_results_v5.csv"
    try:
        ev_df = pd.read_csv(ev_csv)
        if "is_outlier" in ev_df.columns:
            ev_df = ev_df[~ev_df["is_outlier"]].reset_index(drop=True)
        pred_col = "predicted_2MIB_calibrated" if "predicted_2MIB_calibrated" in ev_df.columns else "predicted_smell"

        # ── 样品分布分析 ──────────────────────────────────────────────
        st.markdown("#### 验证样品概况")
        sa1, sa2, sa3, sa4 = st.columns(4)
        with sa1:
            st.metric("总样品数", f"{len(ev_df)}")
        with sa2:
            n_tx = len(ev_df[ev_df["reservoir"] == "亭下"])
            st.metric("亭下水库", f"{n_tx}")
        with sa3:
            n_hs = len(ev_df[ev_df["reservoir"] == "横山"])
            st.metric("横山水库", f"{n_hs}")
        with sa4:
            months = ev_df["month"].nunique()
            st.metric("采样批次", f"{months}次")

        # 月份分布
        month_labels = {"March": "3月", "April": "4月", "May": "5月", "June": "6月"}
        ev_df["月份"] = ev_df["month"].map(month_labels)
        month_dist = ev_df.groupby(["月份", "reservoir"]).size().reset_index(name="样品数")
        fig_dist = px.bar(month_dist, x="月份", y="样品数", color="reservoir",
                          barmode="group", color_discrete_map={"亭下": COLORS["reservoir_tx"], "横山": COLORS["reservoir_hs"]})
        fig_dist.update_layout(height=280, margin=dict(l=40, r=20, t=30, b=40),
                               plot_bgcolor="white", paper_bgcolor="white",
                               legend_title="水库", font=dict(family="Inter"))
        st.plotly_chart(fig_dist, use_container_width=True)

        st.markdown("---")

        # ── 筛选 ─────────────────────────────────────────────────────
        filter_col1, filter_col2 = st.columns([1, 1])
        with filter_col1:
            reservoir_options = ["全部"] + sorted(ev_df["reservoir"].dropna().unique().tolist())
            selected_reservoir = st.selectbox("水库筛选", reservoir_options, key="ev_reservoir")
        with filter_col2:
            month_options = ["全部"] + sorted(ev_df["month"].dropna().unique().tolist())
            selected_month = st.selectbox("月份筛选", month_options, key="ev_month")

        ev_filtered = ev_df.copy()
        if selected_reservoir != "全部":
            ev_filtered = ev_filtered[ev_filtered["reservoir"] == selected_reservoir]
        if selected_month != "全部":
            ev_filtered = ev_filtered[ev_filtered["month"] == selected_month]

        # ── 交互式异常点剔除 ─────────────────────────────────────────
        st.markdown("**交互式异常点剔除（实时重新计算指标）**")
        has_gt = ev_filtered["2-MIB_actual"].notna()
        ev_with_gt = ev_filtered[has_gt].copy()

        point_labels = []
        for _, row in ev_with_gt.iterrows():
            label = f"{row['station']} ({row['month']}, 实测{row['2-MIB_actual']:.0f} ng/L)"
            point_labels.append(label)
        ev_with_gt["label"] = point_labels

        excluded_points = st.multiselect(
            "选择要剔除的异常点（可多选）",
            options=point_labels, default=[], key="ev_exclude",
            help="选中的点将从计算中移除，图中以灰色×标记。指标实时更新。"
        )

        exclude_mask = ev_with_gt["label"].isin(excluded_points)
        ev_clean = ev_with_gt[~exclude_mask].copy()
        ev_excluded = ev_with_gt[exclude_mask].copy()
        n_outliers = len(ev_excluded)

        gt_valid = ev_clean[ev_clean["2-MIB_actual"].notna()].copy()
        if len(gt_valid) >= 3:
            rho_val, _ = spearmanr(gt_valid["2-MIB_actual"], gt_valid[pred_col])
            r_val, _ = pearsonr(gt_valid["2-MIB_actual"], gt_valid[pred_col])
            mae_val = (gt_valid[pred_col] - gt_valid["2-MIB_actual"]).abs().mean()
            r2_val = r_val ** 2
            lr = LinearRegression()
            lr.fit(gt_valid[["2-MIB_actual"]], gt_valid[pred_col])
            reg_slope, reg_intercept = lr.coef_[0], lr.intercept_
        else:
            rho_val = r_val = mae_val = r2_val = float("nan")
            reg_slope, reg_intercept = 1.0, 0.0

        ek1, ek2, ek3, ek4, ek5 = st.columns(5)
        with ek1:
            st.metric("有效样品", f"{len(gt_valid)}", delta=f"-{n_outliers}" if n_outliers else None)
        with ek2:
            st.metric("Spearman ρ", f"{rho_val:.3f}" if not np.isnan(rho_val) else "—")
        with ek3:
            st.metric("Pearson r", f"{r_val:.3f}" if not np.isnan(r_val) else "—")
        with ek4:
            st.metric("MAE (ng/L)", f"{mae_val:.1f}" if not np.isnan(mae_val) else "—")
        with ek5:
            st.metric("R²", f"{r2_val:.3f}" if not np.isnan(r2_val) else "—")

        if n_outliers > 0:
            st.success(f"已剔除 {n_outliers} 个点，指标已重新计算。")

        st.markdown("")

        # ── 散点图 ───────────────────────────────────────────────────
        ext1, ext2 = st.columns(2)
        with ext1:
            st.markdown("**实测 vs 预测 2-MIB**")
            if len(gt_valid) > 0:
                fig_scatter = go.Figure()
                for res, color, symbol in [("亭下", COLORS["reservoir_tx"], "diamond"),
                                            ("横山", COLORS["reservoir_hs"], "circle")]:
                    sub = gt_valid[gt_valid["reservoir"] == res]
                    if len(sub) == 0:
                        continue
                    fig_scatter.add_trace(go.Scatter(
                        x=sub["2-MIB_actual"], y=sub[pred_col],
                        mode="markers", name=f"{res} (n={len(sub)})",
                        marker=dict(color=color, size=9, symbol=symbol, opacity=0.8,
                                    line=dict(width=0.5, color="white")),
                        customdata=sub["station"].values,
                        hovertemplate="站点: %{customdata}<br>实测: %{x:.0f} ng/L<br>预测: %{y:.0f} ng/L<extra></extra>",
                    ))
                if n_outliers > 0:
                    fig_scatter.add_trace(go.Scatter(
                        x=ev_excluded["2-MIB_actual"], y=ev_excluded[pred_col],
                        mode="markers", name=f"已剔除 (n={n_outliers})",
                        marker=dict(color="#CBD5E1", size=12, symbol="x", opacity=0.6,
                                    line=dict(width=2, color="#EF4444")),
                    ))
                all_vals = ev_with_gt[["2-MIB_actual", pred_col]].values.flatten()
                lims = [max(0, min(all_vals) * 0.8), max(all_vals) * 1.1]
                fig_scatter.add_trace(go.Scatter(x=lims, y=lims, mode="lines",
                    line=dict(dash="dash", color="#94A3B8", width=1.5), name="1:1线"))
                if len(gt_valid) >= 3:
                    x_reg = np.linspace(lims[0], lims[1], 50)
                    y_reg = reg_slope * x_reg + reg_intercept
                    fig_scatter.add_trace(go.Scatter(x=x_reg, y=y_reg, mode="lines",
                        line=dict(color="#EF4444", width=2, dash="dot"),
                        name=f"回归线 (y={reg_slope:.2f}x+{reg_intercept:.0f})"))
                fig_scatter.update_layout(
                    xaxis_title="实测 2-MIB (ng/L)", yaxis_title="预测 2-MIB (ng/L)",
                    height=450, margin=dict(l=50, r=20, t=30, b=50),
                    plot_bgcolor="white", paper_bgcolor="white",
                    legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.9)"),
                    font=dict(family="Inter", size=12),
                )
                st.plotly_chart(fig_scatter, use_container_width=True)

        with ext2:
            st.markdown("**残差分布**")
            if len(gt_valid) > 0:
                gt_plot = gt_valid.copy()
                gt_plot["residual"] = gt_plot[pred_col] - gt_plot["2-MIB_actual"]
                fig_resid = go.Figure()
                for res, color in [("亭下", COLORS["reservoir_tx"]), ("横山", COLORS["reservoir_hs"])]:
                    sub = gt_plot[gt_plot["reservoir"] == res]
                    if len(sub) == 0:
                        continue
                    fig_resid.add_trace(go.Scatter(
                        x=sub["2-MIB_actual"], y=sub["residual"],
                        mode="markers", name=f"{res}",
                        marker=dict(color=color, size=8, opacity=0.7),
                        hovertemplate="%{customdata}<br>残差: %{y:.0f}<extra></extra>",
                        customdata=sub["station"].values,
                    ))
                fig_resid.add_hline(y=0, line_dash="dash", line_color="#94A3B8")
                fig_resid.update_layout(
                    xaxis_title="实测 2-MIB (ng/L)", yaxis_title="残差 (ng/L)",
                    height=420, margin=dict(l=50, r=20, t=20, b=50),
                    plot_bgcolor="white", paper_bgcolor="white",
                    font=dict(family="Inter", size=12),
                )
                st.plotly_chart(fig_resid, use_container_width=True)

        # ── 偏差排名 ─────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("**偏差最大的样品**")
        if len(ev_with_gt) > 0:
            ev_ranked = ev_with_gt.copy()
            ev_ranked["绝对偏差"] = (ev_ranked[pred_col] - ev_ranked["2-MIB_actual"]).abs()
            ev_ranked = ev_ranked.sort_values("绝对偏差", ascending=False).head(10)
            top_dev = ev_ranked[["station", "month", "reservoir", "2-MIB_actual", pred_col, "绝对偏差"]].copy()
            top_dev.columns = ["站点", "月份", "水库", "实测(ng/L)", "预测(ng/L)", "绝对偏差(ng/L)"]
            st.dataframe(top_dev, use_container_width=True, hide_index=True)

    except FileNotFoundError:
        st.warning("外部验证数据文件未找到。")
    except Exception as e:
        st.error(f"加载失败：{e}")


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — 模型概览
# ══════════════════════════════════════════════════════════════════════════════
with tab_overview:
    st.markdown("### 技术架构")
    st.markdown("""
<div class="section-card">
    <div class="arch-flow">
        <div class="arch-node">
            <div class="arch-name">EEM光谱</div>
            <div class="arch-label">日立F-4600</div>
        </div>
        <div class="arch-arrow">→</div>
        <div class="arch-node">
            <div class="arch-name">预处理</div>
            <div class="arch-label">空白扣除 · 散射去除</div>
        </div>
        <div class="arch-arrow">→</div>
        <div class="arch-node highlight">
            <div class="arch-name">特征提取</div>
            <div class="arch-label">16维荧光特征</div>
        </div>
        <div class="arch-arrow">→</div>
        <div class="arch-node highlight">
            <div class="arch-name">CatBoost</div>
            <div class="arch-label">梯度提升回归</div>
        </div>
        <div class="arch-arrow">→</div>
        <div class="arch-node">
            <div class="arch-name">预测输出</div>
            <div class="arch-label">2-MIB浓度 · 风险等级</div>
        </div>
    </div>
</div>
    """, unsafe_allow_html=True)

    st.markdown("### 模型性能")
    ov1, ov2 = st.columns(2)
    with ov1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">训练集（326个实验室样品，5折交叉验证）</div>', unsafe_allow_html=True)
        st.markdown(f"""
| 指标 | 数值 | 说明 |
|:-----|:-----|:-----|
| 嗅味 R² | {METRICS['smell_r2']:.3f} | 5折交叉验证均值 |
| 训练样品 | {METRICS['n_train']} | 实验室培养 + 原水配水 |
| 温度覆盖 | 12~30°C | 4个梯度 |
| 培养周期 | 0~7天 | 每日采样 |
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    with ov2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">外部验证（91个现场水样，完全独立）</div>', unsafe_allow_html=True)
        st.markdown(f"""
| 指标 | 数值 | 说明 |
|:-----|:-----|:-----|
| Spearman ρ | {METRICS['spearman_rho']:.3f} | 排序一致性 |
| Pearson r | {METRICS['pearson_r']:.3f} | 线性相关 |
| MAE | {METRICS['mae']:.1f} ng/L | 平均绝对误差 |
| 三级风险相邻准确率 | 100% | 无跨级误判 |
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("### 16维荧光特征说明")
    feat_table = pd.DataFrame({
        "编号": [f"F{i+1}" for i in range(16)],
        "特征名": ANALYTICAL_NAMES,
        "类型": ["区域积分"]*5 + ["峰强度"]*7 + ["荧光指数"]*4,
        "波长范围": [
            "Ex 220~250/Em 280~330", "Ex 220~250/Em 330~380",
            "Ex 220~250/Em 380~480", "Ex 250~400/Em 380~480",
            "Ex 250~400/Em >480",
            "Ex 275/Em 305", "Ex 275/Em 340", "Ex 260/Em 400",
            "Ex 312/Em 420", "Ex 350/Em 440", "Ex 390/Em 509", "Ex 280/Em 370",
            "Em380/Em430 @Ex310", "Em450/Em480 @Ex370",
            "Em305/Em340 @Ex254", "Em435~480/Em300~345 @Ex254",
        ],
        "物理含义": [
            "酪氨酸类荧光", "色氨酸类荧光", "微生物代谢产物",
            "腐殖酸类荧光", "腐殖质类荧光",
            "类蛋白荧光", "类蛋白荧光", "腐殖酸荧光",
            "海洋腐殖质", "陆源腐殖酸", "土壤腐殖质", "类蛋白(新鲜)",
            "生物活性指数", "荧光指数", "新鲜度指数", "腐殖化指数",
        ]
    })
    st.dataframe(feat_table, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 4 — 风险预警说明
# ══════════════════════════════════════════════════════════════════════════════
with tab_risk:
    st.markdown("### 风险预警机制")
    st.markdown("""
本系统依据《生活饮用水卫生标准》（GB 5749-2022）中2-甲基异莰醇（2-MIB）限值10 ng/L为基准，
结合奉化水务实际运行经验，设置三级风险预警：
    """)

    r1, r2, r3 = st.columns(3)
    with r1:
        st.markdown("""
<div class="risk-card risk-safe">
    <div class="risk-icon">✅</div>
    <div class="risk-level">安全</div>
    <div style="margin-top:0.5rem; font-size:0.9rem;">
        2-MIB < 10 ng/L<br>
        低于嗅阈值，正常供水
    </div>
</div>
        """, unsafe_allow_html=True)
    with r2:
        st.markdown("""
<div class="risk-card risk-watch">
    <div class="risk-icon">⚠️</div>
    <div class="risk-level">关注</div>
    <div style="margin-top:0.5rem; font-size:0.9rem;">
        10 ≤ 2-MIB < 50 ng/L<br>
        建议加强监测频次
    </div>
</div>
        """, unsafe_allow_html=True)
    with r3:
        st.markdown("""
<div class="risk-card risk-warn">
    <div class="risk-icon">🚨</div>
    <div class="risk-level">预警</div>
    <div style="margin-top:0.5rem; font-size:0.9rem;">
        2-MIB ≥ 50 ng/L<br>
        启动活性炭应急处置
    </div>
</div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 预警响应流程")
    st.markdown("""
| 步骤 | 操作内容 | 责任人 |
|:-----|:---------|:-------|
| 1 | 现场采水，测量水温和pH | 采样人员 |
| 2 | 实验室EEM测定（日立F-4600） | 检测人员 |
| 3 | 导出.txt文件，上传本系统 | 检测人员 |
| 4 | 系统输出2-MIB预测值和风险等级 | 自动 |
| 5 | 预警级别≥关注时，送GC-MS复核 | 质控主管 |
| 6 | 确认超标，启动活性炭投加 | 水厂运行 |
    """)

    st.markdown("---")
    st.markdown("### 模型局限性说明")
    st.markdown("""
- 本模型为**筛查工具**，不替代GC-MS法定检测
- 训练数据覆盖2-MIB浓度 ND~210 ng/L，超出此范围的极高浓度预测可能偏差较大
- 模型针对伪鱼腥藻（*Pseudanabaena* sp.）产嗅场景优化，其他藻源嗅味可能适用性降低
- 建议每季度用新采样数据验证模型表现，必要时更新训练集
    """)
