# ============================================================
# 金属制品业出厂价格波动智能预警系统
# 手动输入 + 自动获取 + 情景模拟 + 历史回溯
# ============================================================
import streamlit as st
import pandas as pd
import numpy as np
from core2 import full_analysis, generate_report, generate_report_llm
from event_detect import detect_current_events

from core2 import df_history
from data_fetch import auto_fetch_data

# ============================================================
# 页面配置
# ============================================================
st.set_page_config(
    page_title="金属制品业出厂价格波动预警",
    page_icon="📈",
    layout="wide"
)

st.title("📈 金属制品业出厂价格波动智能预警系统")
st.markdown("输入当前月份的 9 个指标，系统将预测未来 1-2 个月的产品出厂价格波动情况。")
st.markdown("---")

# ============================================================
# 侧边栏：手动输入 + 自动获取
# ============================================================
st.sidebar.header("📊 输入当前月指标")

# ---------- 月份输入 ----------
current_month = st.sidebar.text_input("当前月份（如 2026-03）", value="2026-03")

# ---------- 自动获取按钮 ----------
if st.sidebar.button("🔄 自动获取数据（约1分钟）", use_container_width=True):
    with st.spinner(f"正在爬取 {current_month} 的数据..."):
        auto_data, failed = auto_fetch_data(current_month)
        st.session_state['auto_data'] = auto_data
        st.session_state['auto_failed'] = failed
        st.session_state['auto_month'] = current_month

        success = sum(1 for v in auto_data.values() if v is not None)
        if success == 9:
            st.sidebar.success("✅ 全部 9 个指标获取成功")
        else:
            st.sidebar.warning(f"⚠️ 成功 {success}/9，其余需手动填")

# ---------- 读取已爬取的值 ----------
auto_data = st.session_state.get('auto_data', {})
auto_failed = st.session_state.get('auto_failed', [])

# 如果用户改了月份，清空旧的爬取数据
if st.session_state.get('auto_month') and st.session_state['auto_month'] != current_month:
    auto_data = {}
    auto_failed = []

def dval(key, fallback):
    """有爬到的值就用，没有就用默认值"""
    v = auto_data.get(key)
    if v is not None:
        return float(v)
    return fallback

def warn_label(label, key):
    """爬取失败的指标后面加 ⚠️"""
    if key in auto_failed:
        return f"{label} ⚠️"
    return label

# ---------- 9个输入框（保留手动输入） ----------
st.sidebar.markdown("---")

X_input = {
    'X1布伦特原油价格（每桶）': st.sidebar.number_input(
        warn_label("布伦特原油（美元/桶）", 'X1布伦特原油价格（每桶）'),
        value=dval('X1布伦特原油价格（每桶）', 85.0), step=1.0),
    'X2铜价格': st.sidebar.number_input(
        warn_label("铜价格（元/吨）", 'X2铜价格'),
        value=dval('X2铜价格', 72000.0), step=100.0),
    'X4铁矿石价格': st.sidebar.number_input(
        warn_label("铁矿石价格（元/吨）", 'X4铁矿石价格'),
        value=dval('X4铁矿石价格', 850.0), step=10.0),
    'X5焦煤价格': st.sidebar.number_input(
        warn_label("焦煤价格（元/吨）", 'X5焦煤价格'),
        value=dval('X5焦煤价格', 1600.0), step=10.0),
    'X7工业增加值': st.sidebar.number_input(
        warn_label("工业增加值同比(%)", 'X7工业增加值'),
        value=dval('X7工业增加值', 5.8), step=0.1),
    'X8 USD_CNY月度平均汇率': st.sidebar.number_input(
        warn_label("美元兑人民币汇率", 'X8 USD_CNY月度平均汇率'),
        value=dval('X8 USD_CNY月度平均汇率', 708.0), step=1.0),
    'X9 BDI': st.sidebar.number_input(
        warn_label("BDI 指数", 'X9 BDI'),
        value=dval('X9 BDI', 1600.0), step=10.0),
    'X10 PMI（制造业-指数）': st.sidebar.number_input(
        warn_label("制造业 PMI", 'X10 PMI（制造业-指数）'),
        value=dval('X10 PMI（制造业-指数）', 50.8), step=0.1),
    'X11 PMI（制造业-同比增长）': st.sidebar.number_input(
        warn_label("PMI 同比增长", 'X11 PMI（制造业-同比增长）'),
        value=dval('X11 PMI（制造业-同比增长）', 2.0), step=1.0),
}

# ---------- 状态面板 ----------
if auto_data:
    st.sidebar.markdown("---")
    st.sidebar.markdown("**数据获取状态**")
    success_count = sum(1 for v in auto_data.values() if v is not None)
    st.sidebar.markdown(f"✅ 成功：**{success_count}/9**")
    if auto_failed:
        st.sidebar.markdown(f"⚠️ 需手动填：**{len(auto_failed)} 个**")

st.sidebar.markdown("---")
st.sidebar.markdown("**使用方式**：")
st.sidebar.markdown("① 输入月份 → ② 点自动获取 → ③ 缺的手动填 → ④ 开始分析")

# ============================================================
# Tab 布局：实时分析 / 情景模拟 / 历史回溯
# ============================================================
tab1, tab2, tab3 = st.tabs(["📈 实时分析", "🎛️ 情景模拟", "📉 历史回溯"])
# ============================================================
# 重置回调（必须在创建滑块之前定义）
# ============================================================
def reset_sim_values():
    for col_name, val in X_input.items():
        st.session_state[f"sim_{col_name}"] = float(val)

# ============================================================
# Tab 1：实时分析
# ============================================================
with tab1:
    # ---- 顶部提示：报告来源 ----
    st.info("""
    📌 **报告生成说明**
    - 系统会**优先使用 AI 大模型**生成报告
    - 如果 AI 服务不可用（网络/额度问题），**自动降级为模板报告**
    - 报告底部会标明本次报告的实际来源
    """)

    if st.button("🚀 开始分析", type="primary", use_container_width=True):
        with st.spinner("模型分析中..."):
            analysis = full_analysis(X_input, current_month)

            # 先试 AI 报告
            llm_report = generate_report_llm(analysis, current_month)
            if llm_report:
                report = llm_report
                report_source = "🤖 AI 生成"
            else:
                report = generate_report(analysis)
                report_source = "📄 标准模板"

        # ---- 报告来源提示（显眼） ----
        if report_source == "🤖 AI 生成":
            st.success(f"✅ 本次报告来源：{report_source}")
        else:
            st.warning(f"⚠️ 本次报告来源：{report_source}（AI 服务不可用，已降级）")

        # ---- 风险卡片 ----
        col1, col2 = st.columns(2)
        for col, key in [(col1, '下1月'), (col2, '下2月')]:
            with col:
                r = analysis['risk'][key]
                st.subheader(f"📅 {key}价格波动预测")
                level_color = {'高': '🔴', '中': '🟡', '低': '🟢'}
                st.markdown(f"### 价格波动等级：{level_color[r['risk_level']]} **{r['risk_level']}**")
                st.metric("预测 PPI", f"{r['predicted_PPI']:.2f}")
                st.markdown("**各等级概率**")
                st.progress(r['prob_low'], text=f"低波动 {r['prob_low']:.1%}")
                st.progress(r['prob_mid'], text=f"中波动 {r['prob_mid']:.1%}")
                st.progress(r['prob_high'], text=f"高波动 {r['prob_high']:.1%}")

        st.markdown("---")

        # ---- 事件归因 ----
        st.subheader("📰 事件归因")

        st.markdown("#### 🟢 实时预测事件（根据当前输入值触发）")
        if analysis.get('current_events'):
            current_df = pd.DataFrame(analysis['current_events'])
            cols_to_show = ['time', 'event_type', 'direction', 'magnitude']
            cols_to_show = [c for c in cols_to_show if c in current_df.columns]
            display_df = current_df[cols_to_show].copy()
            display_df.columns = ['时间', '事件类型', '方向', '强度']
            st.dataframe(display_df, use_container_width=True, hide_index=True)
        else:
            st.warning("⚠️ 根据当前输入值，未触发任何结构性事件。")

        st.markdown("---")

        st.markdown("#### ⚪ 历史事件（近12个月已发生）")
        if analysis.get('hist_events'):
            hist_df = pd.DataFrame(analysis['hist_events'])
            cols_to_show = ['time', 'event_type', 'direction', 'magnitude']
            cols_to_show = [c for c in cols_to_show if c in hist_df.columns]
            display_df = hist_df[cols_to_show].copy()
            display_df.columns = ['时间', '事件类型', '方向', '强度']
            st.dataframe(display_df, use_container_width=True, hide_index=True)
        else:
            st.caption("近12个月未发现系统性历史事件。")

        st.markdown("---")
        st.markdown("#### 📈 综合事件趋势")
        st.info(analysis['event_trend'])

        st.markdown("---")

        # ---- 完整报告 ----
        st.subheader("📄 出厂价格波动预测报告")
        st.markdown(report)
        st.caption(f"报告来源：{report_source}")

        st.download_button(
            label="📥 下载报告（Markdown）",
            data=report,
            file_name=f"出厂价格预警报告_{current_month}.md",
            mime="text/markdown",
            use_container_width=True
        )
    else:
        st.info("👈 请在左侧输入当前月的指标（或点「自动获取」），然后点击「开始分析」。")


# ============================================================
# Tab 2：情景模拟
# ============================================================
with tab2:
    st.subheader("🎛️ 情景模拟器")
    st.markdown("拖动滑块调整指标，实时看未来风险变化。")
    st.markdown("---")

    DISPLAY_NAMES = {
        'X1布伦特原油价格（每桶）': '布伦特原油（美元/桶）',
        'X2铜价格': '铜价格（元/吨）',
        'X4铁矿石价格': '铁矿石价格（元/吨）',
        'X5焦煤价格': '焦煤价格（元/吨）',
        'X7工业增加值': '工业增加值同比（%）',
        'X8 USD_CNY月度平均汇率': '美元兑人民币汇率',
        'X9 BDI': 'BDI 指数',
        'X10 PMI（制造业-指数）': '制造业 PMI',
        'X11 PMI（制造业-同比增长）': 'PMI 同比增长',
    }

    col_left, col_right = st.columns([1, 1.2])

    with col_left:
        st.markdown("##### 调整指标")
        X_sim = {}

        for col_name in X_input.keys():
            key = f"sim_{col_name}"
            dv = float(X_input[col_name])

            if '原油' in col_name:
                min_v = min(20.0, dv * 0.5)
                max_v = max(200.0, dv * 2)
                step = 1.0
            elif '铜' in col_name:
                min_v = min(30000.0, dv * 0.5)
                max_v = max(100000.0, dv * 2)
                step = 500.0
            elif '铁矿石' in col_name:
                min_v = min(300.0, dv * 0.5)
                max_v = max(1500.0, dv * 2)
                step = 10.0
            elif '焦煤' in col_name:
                min_v = min(500.0, dv * 0.5)
                max_v = max(3500.0, dv * 2)
                step = 50.0
            elif '工业增加值' in col_name:
                min_v = -20.0
                max_v = 40.0
                step = 0.5
            elif '汇率' in col_name:
                min_v = 600.0
                max_v = 750.0
                step = 1.0
            elif 'BDI' in col_name:
                min_v = min(200.0, dv * 0.5)
                max_v = max(5000.0, dv * 2)
                step = 50.0
            elif 'PMI' in col_name and '同比' not in col_name:
                min_v = 35.0
                max_v = 60.0
                step = 0.1
            elif 'PMI' in col_name:
                min_v = -30.0
                max_v = 30.0
                step = 0.1
            else:
                min_v = dv * 0.5
                max_v = dv * 1.5
                step = 1.0

            # 确保 session_state 里的值在范围内
            if key not in st.session_state:
                st.session_state[key] = dv
            else:
                if st.session_state[key] < min_v:
                    st.session_state[key] = min_v
                elif st.session_state[key] > max_v:
                    st.session_state[key] = max_v

            X_sim[col_name] = st.slider(
                DISPLAY_NAMES.get(col_name, col_name),
                min_value=float(min_v),
                max_value=float(max_v),
                step=float(step),
                key=key
            )

        # 重置按钮（用 on_click 回调）
        st.button(
            "🔄 重置为当前输入值",
            use_container_width=True,
            on_click=reset_sim_values
        )

    with col_right:
        st.markdown("##### 实时预测结果")

        sim_analysis = full_analysis(X_sim, current_month)
        sim_risk = sim_analysis['risk']

        c1, c2 = st.columns(2)
        for c, key in [(c1, '下1月'), (c2, '下2月')]:
            with c:
                r = sim_risk[key]
                st.markdown(f"**{key}**")
                if r['risk_level'] == '高':
                    st.error(f"🔴 高风险\n\nPPI {r['predicted_PPI']:.2f}")
                elif r['risk_level'] == '中':
                    st.warning(f"🟡 中风险\n\nPPI {r['predicted_PPI']:.2f}")
                else:
                    st.success(f"🟢 低风险\n\nPPI {r['predicted_PPI']:.2f}")

                st.progress(r['prob_high'], text=f"高风险概率 {r['prob_high']:.1%}")

        st.markdown("---")
        st.markdown("##### 与当前值的对比")

        changes = []
        for col_name in X_input.keys():
            current = float(X_input[col_name])
            sim = X_sim[col_name]
            if current != 0:
                pct = (sim - current) / current * 100
                if abs(pct) > 1:
                    changes.append({
                        '指标': DISPLAY_NAMES.get(col_name, col_name),
                        '原值': round(current, 2),
                        '模拟值': round(sim, 2),
                        '变化': f"{pct:+.1f}%"
                    })

        if changes:
            st.dataframe(pd.DataFrame(changes), use_container_width=True, hide_index=True)
        else:
            st.caption("未调整任何指标。")

# ============================================================
# Tab 3：历史回溯
# ============================================================
with tab3:
    st.subheader("📉 历史预测回溯")
    st.markdown("用过去12个月的数据，跑一遍预测，和真实PPI对比。")
    st.markdown("---")

    df_valid = df_history.dropna(subset=['Y_pred_target']).reset_index(drop=True)

    if len(df_valid) > 14:
        backtest_data = []

        for i in range(len(df_valid) - 13, len(df_valid) - 1):
            row = df_valid.iloc[i]
            next_row = df_valid.iloc[i + 1]

            X_bt = {}
            for col in ['X1布伦特原油价格（每桶）', 'X2铜价格', 'X4铁矿石价格',
                        'X5焦煤价格', 'X7工业增加值', 'X8 USD_CNY月度平均汇率',
                        'X9 BDI', 'X10 PMI（制造业-指数）', 'X11 PMI（制造业-同比增长）']:
                if col in row.index:
                    X_bt[col] = float(row[col])

            month_bt = row['year_month'].strftime('%Y-%m')

            try:
                analysis_bt = full_analysis(X_bt, month_bt)
                pred_ppi = analysis_bt['risk']['下1月']['predicted_PPI']
                risk_level = analysis_bt['risk']['下1月']['risk_level']

                backtest_data.append({
                    '月份': month_bt,
                    '真实PPI': float(next_row['Y_pred_target']),
                    '预测PPI': pred_ppi,
                    '误差': pred_ppi - float(next_row['Y_pred_target']),
                    '风险等级': risk_level,
                })
            except Exception as ex:
                st.warning(f"月份 {month_bt} 回溯失败: {ex}")

        if backtest_data:
            bt_df = pd.DataFrame(backtest_data)

            chart_data = bt_df[['月份', '真实PPI', '预测PPI']].set_index('月份')
            st.line_chart(chart_data)

            mae = bt_df['误差'].abs().mean()
            col_a, col_b = st.columns(2)
            col_a.metric("平均绝对误差 (MAE)", f"{mae:.3f}")
            col_b.metric("回溯样本数", f"{len(bt_df)} 个月")

            with st.expander("查看详细数据"):
                st.dataframe(bt_df, use_container_width=True, hide_index=True)
        else:
            st.info("回溯数据为空。")
    else:
        st.info("数据库样本不足，无法回溯（需要至少14个月）。")
