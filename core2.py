import pandas as pd
import numpy as np
import joblib
import os
from openai import OpenAI
from dotenv import load_dotenv

# 从event_detect导入实时事件识别函数
from event_detect import detect_current_events, apply_event_rules, build_event_json

# ========== 加载历史数据集，用于实时事件识别 ==========
df_history = pd.read_excel('数据库.xlsx', sheet_name='总数据集')
df_history['year_month'] = pd.to_datetime(df_history['year_month'], format='%y-%b')
df_history = df_history.sort_values('year_month').reset_index(drop=True)
df_history['X7工业增加值'] = df_history.groupby(df_history['year_month'].dt.year)['X7工业增加值'].transform(
    lambda x: x.ffill().bfill()
)
# =====================================================

# ============================================================
# 加载模型与事件库
# ============================================================
saved = joblib.load('metal_cost_model_2month_xgb.pkl')
model_t1 = saved['model_t1']
model_t2 = saved['model_t2']
features = saved['features']

event_db = pd.read_csv('event_db.csv')


# ============================================================
# 风险预测
# ============================================================
def predict_risk(X_dict):
    """输入当前X，输出下1月+下2月的波动预测"""
    X_now = pd.DataFrame([X_dict])

    Xa = model_t1['scaler'].transform(X_now[features].values)
    prob_a = model_t1['clf'].predict_proba(Xa)[0]
    mag_a = model_t1['reg'].predict(Xa)[0]

    Xb = model_t2['scaler'].transform(X_now[features].values)
    prob_b = model_t2['clf'].predict_proba(Xb)[0]
    mag_b = model_t2['reg'].predict(Xb)[0]

    def risk_level(p):
        return '高' if p > 0.5 else ('中' if p > 0.3 else '低')

    return {
        '下1月': {
            'prob_low': float(prob_a[0]), 'prob_mid': float(prob_a[1]), 'prob_high': float(prob_a[2]),
            'predicted_PPI': float(mag_a),
            'risk_level': risk_level(prob_a[2])
        },
        '下2月': {
            'prob_low': float(prob_b[0]), 'prob_mid': float(prob_b[1]), 'prob_high': float(prob_b[2]),
            'predicted_PPI': float(mag_b),
            'risk_level': risk_level(prob_b[2])
        }
    }


# ============================================================
# 事件查询
# ============================================================
def get_recent_events(current_month, months=12):
    """查最近N个月的离线历史事件"""
    current = pd.to_datetime(current_month)
    start = current - pd.DateOffset(months=months - 1)
    start_str = start.strftime('%Y-%m')
    end_str = current.strftime('%Y-%m')

    mask = (event_db['time'] >= start_str) & (event_db['time'] <= end_str)
    return event_db[mask].to_dict('records')


def analyze_event_trend(recent_events):
    """判断事件趋势"""
    if not recent_events:
        return "近期无系统性事件，产品价格波动主要来自常规波动。"

    types = [e['event_type'] for e in recent_events]
    if len(set(types)) == 1 and len(types) >= 2:
        return f"「{types[0]}」事件已连续出现{len(types)}个月，需重点关注。"
    else:
        return f"近期出现{len(set(types))}类事件：{'、'.join(set(types))}。"


# ============================================================
# 完整分析（历史事件 + 实时预测事件 双展示）
# ============================================================
def full_analysis(X_dict, current_month):
    """
    完整分析：价格波动预测 + 历史事件(近12个月) + 实时预测事件
    """
    risk = predict_risk(X_dict)

    # 1. 读取离线历史事件（近12个月）
    hist_events = get_recent_events(current_month, months=12)

    # 2. 实时识别当前月份事件（根据你输入的当前值动态判断）
    current_events, event_details = detect_current_events(X_dict, current_month, df_history)

    # 3. 分别计算趋势
    current_trend = analyze_event_trend(current_events)
    hist_trend = analyze_event_trend(hist_events)

    # 综合趋势：优先展示实时，其次展示历史
    if current_events:
        trend = f"【实时事件】{current_trend}"
    elif hist_events:
        trend = f"【近12个月历史事件】{hist_trend}"
    else:
        trend = "近12个月无系统性事件，产品价格波动主要来自常规波动。"

    return {
        'current_month': current_month,
        'risk': risk,
        'current_events': current_events,
        'hist_events': hist_events,
        'event_trend': trend,
        'current_details': event_details
    }


# ============================================================
# 规则版报告生成
# ============================================================
def generate_report(analysis):
    """把分析结果拼成Markdown报告"""
    r = analysis['risk']
    r1 = r['下1月']
    r2 = r['下2月']

    events = analysis.get('current_events', []) + analysis.get('hist_events', [])
    if not events:
        events = analysis.get('recent_events', [])

    trend = analysis['event_trend']

    report = f"# 金属制品业出厂价格波动预警报告\n\n"
    report += f"**分析基准月**：{analysis['current_month']}\n\n"

    report += "## 一、价格波动预警结论\n\n"
    report += f"### 下1个月（价格波动等级：**{r1['risk_level']}**）\n\n"
    report += f"- 预测金属制品业出厂PPI：{r1['predicted_PPI']:.2f}\n"
    report += f"- 低波动概率：{r1['prob_low']:.1%}\n"
    report += f"- 中波动概率：{r1['prob_mid']:.1%}\n"
    report += f"- 高波动概率：{r1['prob_high']:.1%}\n\n"

    report += f"### 下2个月（价格波动等级：**{r2['risk_level']}**）\n\n"
    report += f"- 预测金属制品业出厂PPI：{r2['predicted_PPI']:.2f}\n"
    report += f"- 低波动概率：{r2['prob_low']:.1%}\n"
    report += f"- 中波动概率：{r2['prob_mid']:.1%}\n"
    report += f"- 高波动概率：{r2['prob_high']:.1%}\n\n"

    report += "## 二、宏观事件归因\n\n"
    report += f"{trend}\n\n"
    if events:
        for e in events:
            report += f"- **{e['time']}**：{e['event_type']}（{e['direction']}，强度{e['magnitude']}）\n"
        report += "\n"

    report += "## 三、经营决策建议\n\n"

    report += "### 📢 销售部门\n"
    if r1['risk_level'] == '高':
        report += "- 预期出厂价上行，**优先锁定产品销售长单**\n"
        report += "- 暂缓低价签单，等待价格上行\n"
    elif r1['risk_level'] == '中':
        report += "- 价格存在上行可能性，**审慎签订低价长期合同**\n"
        report += "- 签中短期合同，保留调价条款\n"
    else:
        report += "- 价格波动风险较低，**维持常规接单节奏**\n"

    report += "\n### 🏭 生产计划\n"
    if r1['risk_level'] == '高':
        report += "- 适度加大产能，提前排产\n"
        report += "- 增加成品库存，迎接价格上涨\n"
    elif r1['risk_level'] == '中':
        report += "- 动态调整生产计划，关注订单变化\n"
    else:
        report += "- 按订单生产，不提前备货\n"

    report += "\n### 💼 公司管理层\n"
    if r1['risk_level'] == '高':
        report += "- 关注需求端承接能力，防止涨价导致客户流失\n"
        report += "- 监控原材料成本传导是否顺畅\n"
    elif r1['risk_level'] == '中':
        report += "- 持续跟踪宏观指标，做好中长期经营规划\n"
    else:
        report += "- 维持现有经营策略，关注市场变化\n"

    return report


# ============================================================
# 大模型报告生成（DeepSeek）—— 【安全初始化客户端】
# ============================================================
# 1. 尝试加载本地 .env 文件
try:
    load_dotenv()
except Exception:
    pass

# 2. 尝试从 Streamlit Cloud 的 Secrets 里读取 Key
try:
    import streamlit as st
    if "DEEPSEEK_API_KEY" in st.secrets:
        os.environ["DEEPSEEK_API_KEY"] = st.secrets["DEEPSEEK_API_KEY"]
except Exception:
    pass

# 3. 检查 Key 是否配置成功
if not os.getenv("DEEPSEEK_API_KEY"):
    print("⚠️ 警告：未检测到 DEEPSEEK_API_KEY，大模型功能将无法使用。请检查 .env 文件或 Streamlit Secrets 配置。")

# 4. 初始化客户端（全局只初始化一次，不要写在函数里）
_client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)


# ============================================================
# 大模型报告生成函数（只定义一次）
# ============================================================
def generate_report_llm(analysis, current_month):
    """用DeepSeek生成报告，失败返回None"""
    r = analysis['risk']
    r1 = r['下1月']
    r2 = r['下2月']

    # ⚠️ 兼容新版的事件结构
    events = analysis.get('current_events', []) + analysis.get('hist_events', [])
    if not events:
        events = analysis.get('recent_events', [])

    events_text = "\n".join([
        f"- {e['time']}: {e['event_type']} ({e['direction']})"
        for e in events
    ]) if events else "近期无系统性事件"

    system_prompt = """你是金属制品行业市场分析师。
请基于提供的出厂价格预测数据，为企业内部三个部门（销售、生产、管理层）
撰写一份简洁专业的出厂价格波动预测简报。

要求：
1. 总体判断（涨/跌/震荡 + 幅度）
2. 驱动因素分析
3. 给三个部门的具体行动建议（要可执行）
4. 不超过400字
"""

    user_prompt = f"""分析基准月：{current_month}

【出厂价格预测】
- 下1月：{r1['predicted_PPI']:.2f}，风险{r1['risk_level']}
- 下2月：{r2['predicted_PPI']:.2f}，风险{r2['risk_level']}

【概率分布】
- 下1月：低{r1['prob_low']:.1%}，中{r1['prob_mid']:.1%}，高{r1['prob_high']:.1%}

【近期事件】
{events_text}

请写一份出厂价格波动预测报告。
"""

    try:
        resp = _client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=800
        )
        return resp.choices[0].message.content
    except Exception as e:
        print(f"DeepSeek 报告生成失败: {e}")
        return None