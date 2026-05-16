import streamlit as st
import baostock as bs
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ===== 页面配置 =====
st.set_page_config(
    page_title="股票波动分析系统",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== 样式 =====
st.markdown("""
<style>
  .main { background: #0f0f0f; }
  .metric-card {
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-radius: 10px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
  }
  .metric-label { font-size: 12px; color: #888; margin-bottom: 4px; }
  .metric-value { font-size: 24px; font-weight: 600; }
  .green { color: #4caf72; }
  .red { color: #e05555; }
  .amber { color: #d4a843; }
  .signal-buy {
    background: #1a3326;
    border: 1px solid #4caf72;
    color: #4caf72;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 600;
  }
  .signal-sell {
    background: #321616;
    border: 1px solid #e05555;
    color: #e05555;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 600;
  }
  .signal-watch {
    background: #2a2010;
    border: 1px solid #d4a843;
    color: #d4a843;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 13px;
  }
  .signal-hold {
    background: #1e1e1e;
    border: 1px solid #444;
    color: #888;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 13px;
  }
  div[data-testid="stMetric"] {
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-radius: 10px;
    padding: 12px 16px;
  }
</style>
""", unsafe_allow_html=True)

# ===== baostock 工具函数 =====
@st.cache_resource
def get_bs_session():
    lg = bs.login()
    return lg.error_code == "0"

def to_bs_code(code: str) -> str:
    code = code.strip()
    if "." in code:
        return code
    if code.startswith("6") or code.startswith("9"):
        return f"sh.{code}"
    return f"sz.{code}"

def get_end_date():
    now = datetime.now()
    if now.hour < 16:
        return (now - timedelta(days=1)).strftime("%Y-%m-%d")
    return now.strftime("%Y-%m-%d")

@st.cache_data(ttl=3600)
def fetch_price_data(bs_code: str, days: int = 365) -> pd.DataFrame:
    get_bs_session()
    end_date = get_end_date()
    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    rs = bs.query_history_k_data_plus(
        bs_code,
        "date,open,high,low,close,volume,peTTM,pbMRQ,dividendYieldRatio",
        start_date=start_date,
        end_date=end_date,
        frequency="d",
        adjustflag="2"
    )
    rows = []
    while rs.error_code == "0" and rs.next():
        rows.append(rs.get_row_data())
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=rs.fields)
    for col in ["open","high","low","close","volume","peTTM","pbMRQ","dividendYieldRatio"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["close"]).reset_index(drop=True)

@st.cache_data(ttl=3600)
def fetch_fundamentals(bs_code: str) -> dict:
    get_bs_session()
    result = {}
    try:
        rs = bs.query_profit_data(code=bs_code, year=2024, quarter=4)
        if rs.error_code == "0" and rs.next():
            d = dict(zip(rs.fields, rs.get_row_data()))
            result["roe"] = round(float(d.get("roeAvg") or 0) * 100, 2)
            result["gross_margin"] = round(float(d.get("grossProfitMargin") or 0) * 100, 2)
    except: pass
    try:
        rs = bs.query_growth_data(code=bs_code, year=2024, quarter=4)
        if rs.error_code == "0" and rs.next():
            d = dict(zip(rs.fields, rs.get_row_data()))
            result["revenue_growth"] = round(float(d.get("YOYRevenue") or 0), 2)
            result["profit_growth"] = round(float(d.get("YOYNI") or 0), 2)
    except: pass
    try:
        rs = bs.query_balance_data(code=bs_code, year=2024, quarter=4)
        if rs.error_code == "0" and rs.next():
            d = dict(zip(rs.fields, rs.get_row_data()))
            result["debt_ratio"] = round(float(d.get("debtToAsset") or 0) * 100, 2)
    except: pass
    return result

@st.cache_data(ttl=3600)
def fetch_stock_name(bs_code: str) -> str:
    get_bs_session()
    try:
        rs = bs.query_stock_basic(code=bs_code)
        if rs.error_code == "0" and rs.next():
            d = dict(zip(rs.fields, rs.get_row_data()))
            return d.get("code_name", bs_code)
    except: pass
    return bs_code.split(".")[-1]

# ===== 分析函数 =====
def calc_support_resistance(close, high, low):
    price_min = float(low.min())
    price_max = float(high.max())
    counts, edges = np.histogram(close, bins=50, range=(price_min, price_max))
    centers = (edges[:-1] + edges[1:]) / 2
    cur = float(close.iloc[-1])
    sorted_idx = np.argsort(counts)[::-1]
    sup_candidates = [centers[i] for i in sorted_idx[:15] if centers[i] < cur]
    res_candidates = [centers[i] for i in sorted_idx[:15] if centers[i] > cur]
    support = round(max(min(sup_candidates) if sup_candidates else price_min, float(low.tail(90).min()) * 0.98), 2)
    resistance = round(min(max(res_candidates) if res_candidates else price_max, float(high.tail(90).max()) * 1.02), 2)
    return support, resistance

def calc_indicators(close):
    # RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rsi = float((100 - 100 / (1 + gain / loss.replace(0, 1e-9))).iloc[-1])

    # MACD
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9).mean()
    macd = (dif - dea) * 2

    # 布林带
    ma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    boll_upper = ma20 + 2 * std20
    boll_lower = ma20 - 2 * std20

    # 波动周期
    ma = close.rolling(20).mean().dropna().reset_index(drop=True)
    c = close.iloc[len(close)-len(ma):].reset_index(drop=True)
    crossings = int((c > ma).astype(int).diff().abs().sum())
    cycle = max(7, min(int(round(len(c) / max(crossings/2, 1))), 90))
    vol_pct = round(((close.max() - close.min()) / close.min()) * 100, 2)

    return {
        "rsi": round(rsi, 1),
        "dif": round(float(dif.iloc[-1]), 3),
        "dea": round(float(dea.iloc[-1]), 3),
        "macd": round(float(macd.iloc[-1]), 3),
        "boll_upper": round(float(boll_upper.iloc[-1]), 2),
        "boll_mid": round(float(ma20.iloc[-1]), 2),
        "boll_lower": round(float(boll_lower.iloc[-1]), 2),
        "cycle": cycle,
        "vol_pct": vol_pct,
        "dif_series": dif,
        "dea_series": dea,
        "macd_series": macd,
        "boll_upper_series": boll_upper,
        "boll_mid_series": ma20,
        "boll_lower_series": boll_lower,
    }

def calc_score(data: dict, weights: dict) -> tuple:
    scores = {}
    pe = data.get("pe", 99) or 99
    pb = data.get("pb", 99) or 99
    roe = data.get("roe", 0) or 0
    gm = data.get("gross_margin", 0) or 0
    rev = data.get("revenue_growth", 0) or 0
    pft = data.get("profit_growth", 0) or 0
    debt = data.get("debt_ratio", 80) or 80
    div = data.get("dividend_yield", 0) or 0
    vol = data.get("vol_pct", 50) or 50
    cycle = data.get("cycle", 30) or 30
    rsi = data.get("rsi", 50) or 50

    raw = {
        "估值": (15 if pe<15 else 11 if pe<25 else 6 if pe<35 else 2) + (8 if pb<1.5 else 6 if pb<3 else 3 if pb<5 else 1),
        "盈利": (12 if roe>25 else 9 if roe>18 else 6 if roe>12 else 3 if roe>8 else 1) + (8 if gm>60 else 6 if gm>40 else 4 if gm>25 else 2),
        "成长": (8 if rev>20 else 6 if rev>10 else 3 if rev>0 else 1) + (7 if pft>20 else 5 if pft>10 else 2 if pft>0 else 1),
        "财务": (9 if debt<25 else 7 if debt<40 else 4 if debt<60 else 2) + (6 if div>4 else 4 if div>2 else 2 if div>0 else 1),
        "波动": (10 if vol<10 else 7 if vol<20 else 5 if vol<30 else 2) + (8 if 20<=cycle<=45 else 5 if cycle<60 else 2) + (5 if 35<=rsi<=65 else 3),
    }
    maxes = {"估值": 23, "盈利": 20, "成长": 15, "财务": 15, "波动": 23}
    w_map = {"估值": weights["w_val"], "盈利": weights["w_prof"], "成长": weights["w_grow"], "财务": weights["w_fin"], "波动": weights["w_wav"]}

    total_w = sum(w_map.values())
    final = sum((raw[k]/maxes[k]) * w_map[k] / total_w * 100 for k in raw)
    for k in raw:
        scores[k] = {"got": raw[k], "max": maxes[k], "pct": round(raw[k]/maxes[k]*100)}
    return round(final, 1), scores

def get_signal(cur, support, resistance):
    if cur <= support * 1.02:
        return "立即买入", "buy"
    elif cur >= resistance * 0.98:
        return "立即卖出", "sell"
    elif cur <= support * 1.07:
        return "接近买点", "watch_buy"
    elif cur >= resistance * 0.93:
        return "接近卖点", "watch_sell"
    return "观望", "hold"

def get_pattern(vol_pct, cycle):
    if vol_pct < 12 and 15 <= cycle <= 50:
        return "优质震荡", "#4caf72"
    elif vol_pct < 25 and cycle <= 60:
        return "规律波动", "#d4a843"
    elif vol_pct >= 35:
        return "高波动", "#e05555"
    return "一般震荡", "#d4a843"

@st.cache_data(ttl=3600)
def full_analyze(code: str) -> dict:
    bs_code = to_bs_code(code)
    df = fetch_price_data(bs_code)
    if df.empty or len(df) < 5:
        return {"error": f"股票代码 {code} 数据不足，请确认代码正确"}
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    cur = round(float(close.iloc[-1]), 2)
    prev = round(float(close.iloc[-2]), 2) if len(close) > 1 else cur
    chg = round((cur - prev) / prev * 100, 2)
    support, resistance = calc_support_resistance(close, high, low)
    ind = calc_indicators(close)
    pe = round(float(df["peTTM"].dropna().iloc[-1]), 2) if not df["peTTM"].dropna().empty else 0
    pb = round(float(df["pbMRQ"].dropna().iloc[-1]), 2) if not df["pbMRQ"].dropna().empty else 0
    div = round(float(df["dividendYieldRatio"].dropna().iloc[-1]), 2) if not df["dividendYieldRatio"].dropna().empty else 0
    fin = fetch_fundamentals(bs_code)
    name = fetch_stock_name(bs_code)
    return {
        "code": code, "bs_code": bs_code, "name": name,
        "cur": cur, "prev": prev, "chg": chg,
        "support": support, "resistance": resistance,
        "pe": pe, "pb": pb, "dividend_yield": div,
        **fin, **ind, "df": df,
        "update_time": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }

# ===== K线图 =====
def draw_kline(df, support, resistance, ind):
    df60 = df.tail(60).copy()
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.6, 0.2, 0.2],
        vertical_spacing=0.02
    )
    # K线
    fig.add_trace(go.Candlestick(
        x=df60["date"], open=df60["open"], high=df60["high"],
        low=df60["low"], close=df60["close"],
        increasing_line_color="#e05555", decreasing_line_color="#4caf72",
        increasing_fillcolor="#e05555", decreasing_fillcolor="#4caf72",
        name="K线"
    ), row=1, col=1)
    # 布林带
    idx = df60.index
    fig.add_trace(go.Scatter(x=df60["date"], y=ind["boll_upper_series"].reindex(idx), line=dict(color="#5b8fd4", width=1, dash="dot"), name="布林上轨", showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=df60["date"], y=ind["boll_mid_series"].reindex(idx), line=dict(color="#888", width=1), name="MA20", showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=df60["date"], y=ind["boll_lower_series"].reindex(idx), line=dict(color="#5b8fd4", width=1, dash="dot"), name="布林下轨", showlegend=False), row=1, col=1)
    # 支撑/压力
    fig.add_hline(y=support, line_color="#4caf72", line_dash="dash", line_width=1.5, annotation_text=f"支撑 ¥{support}", annotation_position="right", row=1, col=1)
    fig.add_hline(y=resistance, line_color="#e05555", line_dash="dash", line_width=1.5, annotation_text=f"压力 ¥{resistance}", annotation_position="right", row=1, col=1)
    # 成交量
    colors = ["#e05555" if c >= o else "#4caf72" for c, o in zip(df60["close"], df60["open"])]
    fig.add_trace(go.Bar(x=df60["date"], y=df60["volume"], marker_color=colors, name="成交量", showlegend=False), row=2, col=1)
    # MACD
    macd_s = ind["macd_series"].reindex(idx)
    macd_colors = ["#e05555" if v >= 0 else "#4caf72" for v in macd_s]
    fig.add_trace(go.Bar(x=df60["date"], y=macd_s, marker_color=macd_colors, name="MACD", showlegend=False), row=3, col=1)
    fig.add_trace(go.Scatter(x=df60["date"], y=ind["dif_series"].reindex(idx), line=dict(color="#5b8fd4", width=1), name="DIF", showlegend=False), row=3, col=1)
    fig.add_trace(go.Scatter(x=df60["date"], y=ind["dea_series"].reindex(idx), line=dict(color="#d4a843", width=1), name="DEA", showlegend=False), row=3, col=1)

    fig.update_layout(
        paper_bgcolor="#0f0f0f", plot_bgcolor="#0f0f0f",
        font=dict(color="#888", size=11),
        margin=dict(l=10, r=80, t=10, b=10),
        xaxis_rangeslider_visible=False,
        height=520,
        showlegend=False,
    )
    for i in range(1, 4):
        fig.update_xaxes(gridcolor="#1e1e1e", row=i, col=1)
        fig.update_yaxes(gridcolor="#1e1e1e", row=i, col=1)
    return fig

# ===== 侧边栏 =====
with st.sidebar:
    st.markdown("## 📈 股票波动分析")
    st.markdown("---")
    page = st.radio("功能", ["🔍 单股分析", "📋 批量监控"], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("### ⚖️ 评分权重配置")
    w_val  = st.slider("估值权重",  0.0, 1.0, 0.20, 0.05)
    w_prof = st.slider("盈利权重",  0.0, 1.0, 0.20, 0.05)
    w_grow = st.slider("成长权重",  0.0, 1.0, 0.15, 0.05)
    w_fin  = st.slider("财务健康",  0.0, 1.0, 0.15, 0.05)
    w_wav  = st.slider("波动规律",  0.0, 1.0, 0.30, 0.05)
    weights = {"w_val": w_val, "w_prof": w_prof, "w_grow": w_grow, "w_fin": w_fin, "w_wav": w_wav}
    st.caption(f"权重合计: {sum(weights.values()):.2f}")
    st.markdown("---")
    st.caption(f"数据来源: BaoStock\n更新时间: {get_end_date()}")

# ===== 单股分析页面 =====
if "单股" in page:
    st.markdown("# 🔍 单股波动分析")
    col_input, col_btn = st.columns([4, 1])
    with col_input:
        code_input = st.text_input("股票代码", placeholder="如 600519、000858", label_visibility="collapsed")
    with col_btn:
        analyze_btn = st.button("分 析", use_container_width=True, type="primary")

    quick_cols = st.columns(6)
    quick_stocks = [("600519","茅台"), ("600900","长江电力"), ("000333","美的"), ("600036","招行"), ("601318","平安"), ("000858","五粮液")]
    for i, (qcode, qname) in enumerate(quick_stocks):
        with quick_cols[i]:
            if st.button(f"{qcode}\n{qname}", use_container_width=True):
                code_input = qcode
                analyze_btn = True

    if analyze_btn and code_input:
        with st.spinner(f"正在分析 {code_input}..."):
            data = full_analyze(code_input.strip())

        if "error" in data:
            st.error(data["error"])
        else:
            score, score_details = calc_score(data, weights)
            signal_text, signal_type = get_signal(data["cur"], data["support"], data["resistance"])
            pattern, pattern_color = get_pattern(data["vol_pct"], data["cycle"])
            dist_sup = round((data["cur"] - data["support"]) / data["support"] * 100, 1)
            dist_res = round((data["resistance"] - data["cur"]) / data["cur"] * 100, 1)

            # 头部
            st.markdown(f"### {data['name']} · {data['code']}")
            signal_class = {"buy":"signal-buy","sell":"signal-sell","watch_buy":"signal-watch","watch_sell":"signal-watch","hold":"signal-hold"}[signal_type]
            
            col1, col2, col3, col4, col5 = st.columns(5)
            chg_color = "red" if data["chg"] >= 0 else "green"
            col1.metric("当前价", f"¥{data['cur']}", f"{'+' if data['chg']>=0 else ''}{data['chg']}%")
            col2.metric("综合评分", f"{score} 分")
            col3.metric("支撑位（买入参考）", f"¥{data['support']}", f"距离 {dist_sup:+.1f}%")
            col4.metric("压力位（卖出参考）", f"¥{data['resistance']}", f"距离 +{dist_res:.1f}%")
            col5.metric("波动周期", f"{data['cycle']} 天", f"振幅 {data['vol_pct']}%")

            # 结论
            st.markdown(f"""
            <div style="background:#1a1a1a;border:1px solid #2a2a2a;border-radius:10px;padding:1rem 1.5rem;margin:1rem 0;display:flex;align-items:center;gap:2rem">
                <div style="font-size:36px;font-weight:700;color:{pattern_color}">{score}</div>
                <div style="width:1px;height:40px;background:#2a2a2a"></div>
                <div>
                    <div style="font-size:18px;font-weight:600;color:{pattern_color}">{pattern}</div>
                    <div style="font-size:13px;color:#888;margin-top:4px">{signal_text} · 更新于 {data['update_time']}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # K线图
            fig = draw_kline(data["df"], data["support"], data["resistance"], data)
            st.plotly_chart(fig, use_container_width=True)

            # 评分 + 技术指标
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("#### 📊 评分明细")
                for name, v in score_details.items():
                    pct = v["pct"]
                    bar_color = "#4caf72" if pct >= 70 else "#d4a843" if pct >= 45 else "#e05555"
                    st.markdown(f"""
                    <div style="margin-bottom:10px">
                        <div style="display:flex;justify-content:space-between;font-size:12px;color:#888;margin-bottom:4px">
                            <span>{name}</span><span>{v['got']}/{v['max']}</span>
                        </div>
                        <div style="background:#2a2a2a;border-radius:3px;height:5px">
                            <div style="width:{pct}%;background:{bar_color};height:5px;border-radius:3px"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            with col_b:
                st.markdown("#### 📡 技术指标")
                rsi_color = "#4caf72" if data["rsi"] < 35 else "#e05555" if data["rsi"] > 70 else "#e8e6e0"
                rsi_desc = "超卖" if data["rsi"] < 35 else "超买" if data["rsi"] > 70 else "中性"
                macd_color = "#4caf72" if data["macd"] > 0 else "#e05555"
                tc1, tc2 = st.columns(2)
                tc1.metric("RSI(14)", data["rsi"], rsi_desc)
                tc2.metric("MACD", data["macd"], "金叉" if data["dif"] > data["dea"] else "死叉")
                tc3, tc4 = st.columns(2)
                tc3.metric("布林下轨", f"¥{data['boll_lower']}")
                tc4.metric("布林上轨", f"¥{data['boll_upper']}")
                tc5, tc6 = st.columns(2)
                tc5.metric("PE(TTM)", data["pe"] or "—")
                tc6.metric("PB(MRQ)", data["pb"] or "—")

            # 基本面
            st.markdown("#### 🏦 基本面数据")
            fc = st.columns(4)
            fc[0].metric("ROE", f"{data.get('roe', 0) or '—'}{'%' if data.get('roe') else ''}")
            fc[1].metric("毛利率", f"{data.get('gross_margin', 0) or '—'}{'%' if data.get('gross_margin') else ''}")
            fc[2].metric("营收增速", f"{data.get('revenue_growth', 0) or '—'}{'%' if data.get('revenue_growth') else ''}")
            fc[3].metric("资产负债率", f"{data.get('debt_ratio', 0) or '—'}{'%' if data.get('debt_ratio') else ''}")

# ===== 批量监控页面 =====
else:
    st.markdown("# 📋 批量监控")

    # 添加股票
    with st.expander("➕ 添加监控股票", expanded=True):
        ac1, ac2, ac3, ac4, ac5 = st.columns([2,2,2,2,1])
        with ac1: add_code = st.text_input("股票代码", key="add_code", placeholder="600519")
        with ac2: add_buy = st.number_input("买入价", min_value=0.0, step=0.01, key="add_buy", format="%.2f")
        with ac3: add_sell = st.number_input("卖出价", min_value=0.0, step=0.01, key="add_sell", format="%.2f")
        with ac4: add_cur = st.number_input("当前价（可选）", min_value=0.0, step=0.01, key="add_cur", format="%.2f")
        with ac5:
            st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
            add_btn = st.button("添加", use_container_width=True, type="primary")

        if add_btn and add_code:
            if "watchlist" not in st.session_state:
                st.session_state.watchlist = []
            if not any(s["code"] == add_code.strip() for s in st.session_state.watchlist):
                st.session_state.watchlist.append({
                    "code": add_code.strip(),
                    "buy": add_buy,
                    "sell": add_sell,
                    "cur": add_cur or None,
                })
                st.success(f"✅ 已添加 {add_code}")
                st.rerun()
            else:
                st.warning("该股票已在监控列表中")

    # 快捷添加默认股票
    if "watchlist" not in st.session_state:
        st.session_state.watchlist = []

    if not st.session_state.watchlist:
        if st.button("📋 加载示例股票"):
            st.session_state.watchlist = [
                {"code": "600900", "buy": 25.5, "sell": 27.3, "cur": None},
                {"code": "600036", "buy": 37.0, "sell": 40.0, "cur": None},
                {"code": "000333", "buy": 72.0, "sell": 82.0, "cur": None},
            ]
            st.rerun()

    if st.session_state.watchlist:
        refresh = st.button("🔄 刷新所有数据", type="primary")

        rows = []
        for item in st.session_state.watchlist:
            with st.spinner(f"获取 {item['code']}..."):
                data = full_analyze(item["code"])
            if "error" in data:
                rows.append({"代码": item["code"], "名称": "查询失败", "当前价": "—", "买入线": item["buy"], "卖出线": item["sell"], "信号": "错误", "评分": 0, "波动": "—", "周期": "—"})
                continue
            cur = data["cur"]
            buy = item["buy"] or data["support"]
            sell = item["sell"] or data["resistance"]
            score, _ = calc_score(data, weights)
            sig, sig_type = get_signal(cur, buy, sell)
            pattern, _ = get_pattern(data["vol_pct"], data["cycle"])
            chg_str = f"{'+' if data['chg']>=0 else ''}{data['chg']}%"
            rows.append({
                "代码": item["code"],
                "名称": data["name"],
                "当前价": f"¥{cur} ({chg_str})",
                "买入线": f"¥{buy}",
                "卖出线": f"¥{sell}",
                "信号": sig,
                "评分": score,
                "振幅": f"{data['vol_pct']}%",
                "周期": f"{data['cycle']}天",
                "类型": pattern,
            })

        if rows:
            # 信号汇总
            buy_count = sum(1 for r in rows if "买入" in r["信号"])
            sell_count = sum(1 for r in rows if "卖出" in r["信号"])
            sm1, sm2, sm3, sm4 = st.columns(4)
            sm1.metric("监控总数", len(rows))
            sm2.metric("买入信号", buy_count)
            sm3.metric("卖出信号", sell_count)
            sm4.metric("平均评分", round(sum(r["评分"] for r in rows) / len(rows), 1))

            st.markdown("---")

            # 按信号排序展示
            signal_order = {"立即买入": 0, "立即卖出": 1, "接近买点": 2, "接近卖点": 3, "观望": 4, "错误": 5}
            rows_sorted = sorted(rows, key=lambda r: signal_order.get(r["信号"], 9))

            for r in rows_sorted:
                sig_color = {"立即买入": "#4caf72", "立即卖出": "#e05555", "接近买点": "#d4a843", "接近卖点": "#d4a843"}.get(r["信号"], "#888")
                with st.container():
                    c1,c2,c3,c4,c5,c6,c7,c8 = st.columns([1.5,2,2,2,2,1.5,1.5,1])
                    c1.markdown(f"**{r['代码']}**")
                    c2.markdown(r["名称"])
                    c3.markdown(r["当前价"])
                    c4.markdown(f"<span style='color:#4caf72'>{r['买入线']}</span>", unsafe_allow_html=True)
                    c5.markdown(f"<span style='color:#e05555'>{r['卖出线']}</span>", unsafe_allow_html=True)
                    c6.markdown(f"<span style='color:{sig_color};font-weight:600'>{r['信号']}</span>", unsafe_allow_html=True)
                    c7.markdown(f"**{r['评分']}** 分")
                    if c8.button("删除", key=f"del_{r['代码']}"):
                        st.session_state.watchlist = [s for s in st.session_state.watchlist if s["code"] != r["代码"]]
                        st.rerun()
                    st.divider()
