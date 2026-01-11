import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
import numpy as np

# 設定頁面配置
st.set_page_config(
    page_title="網路行銷分析儀表板",
    page_icon="📊",
    layout="wide"
)

# 標題
st.title("📊 網路行銷數據分析儀表板")
st.markdown("### AI 協作行銷分析工具")

# 側邊欄
st.sidebar.header("資料來源設定")
data_option = st.sidebar.radio(
    "選擇資料來源：",
    ["使用示範資料", "上傳自己的資料"]
)

# 生成示範資料的函數
def generate_demo_data():
    dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
    data = {
        '日期': dates,
        '訪客數': np.random.randint(100, 500, 30),
        '頁面瀏覽數': np.random.randint(300, 1500, 30),
        '轉換次數': np.random.randint(5, 50, 30),
        '跳出率': np.random.uniform(30, 70, 30)
    }
    df = pd.DataFrame(data)
    df['轉換率'] = (df['轉換次數'] / df['訪客數'] * 100).round(2)
    return df

# 載入資料
if data_option == "使用示範資料":
    df = generate_demo_data()
    st.sidebar.success("✅ 已載入示範資料（最近30天）")
# 日期範圍篩選
st.sidebar.markdown("---")
st.sidebar.subheader("📅 日期範圍篩選")
min_date = df['日期'].min().date()
max_date = df['日期'].max().date()

date_range = st.sidebar.date_input(
    "選擇分析日期範圍：",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# 根據選擇的日期篩選資料
if len(date_range) == 2:
    start_date, end_date = date_range
    df = df[(df['日期'].dt.date >= start_date) & (df['日期'].dt.date <= end_date)]
    st.sidebar.info(f"已篩選：{len(df)} 天的資料")
else:
    uploaded_file = st.sidebar.file_uploader("上傳 CSV 或 Excel 檔案", type=['csv', 'xlsx'])
    if uploaded_file:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        st.sidebar.success("✅ 檔案上傳成功！")
    else:
        st.info("👈 請從左側上傳資料檔案，或使用示範資料")
        st.stop()

# 顯示資料概覽
st.subheader("📋 資料概覽")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("總訪客數", f"{df['訪客數'].sum():,}")
with col2:
    st.metric("總頁面瀏覽數", f"{df['頁面瀏覽數'].sum():,}")
with col3:
    st.metric("總轉換次數", f"{df['轉換次數'].sum():,}")
with col4:
    avg_conversion = (df['轉換次數'].sum() / df['訪客數'].sum() * 100)
    st.metric("平均轉換率", f"{avg_conversion:.2f}%")

# 趨勢圖表
st.subheader("📈 流量趨勢分析")

# 訪客數趨勢
fig_visitors = px.line(
    df, 
    x='日期', 
    y='訪客數',
    title='每日訪客數趨勢',
    markers=True
)
fig_visitors.update_layout(height=400)
st.plotly_chart(fig_visitors, use_container_width=True)

# 雙軸圖表
col1, col2 = st.columns(2)

with col1:
    fig_conversion = px.bar(
        df,
        x='日期',
        y='轉換次數',
        title='每日轉換次數'
    )
    fig_conversion.update_layout(height=350)
    st.plotly_chart(fig_conversion, use_container_width=True)

with col2:
    fig_bounce = px.line(
        df,
        x='日期',
        y='跳出率',
        title='跳出率變化',
        markers=True
    )
    fig_bounce.update_layout(height=350)
    st.plotly_chart(fig_bounce, use_container_width=True)

# 顯示原始資料
with st.expander("🔍 查看原始資料"):
    st.dataframe(df, use_container_width=True)

# AI 預測功能
st.subheader("🤖 AI 流量預測")
st.markdown("使用簡單線性迴歸預測未來7天流量趨勢")

# 預測未來7天
from sklearn.linear_model import LinearRegression

# 準備訓練資料
X = np.arange(len(df)).reshape(-1, 1)
y = df['訪客數'].values

# 訓練模型
model = LinearRegression()  
model.fit(X, y)

# 預測未來7天
future_days = 7
future_X = np.arange(len(df), len(df) + future_days).reshape(-1, 1)
predictions = model.predict(future_X)

# 建立預測資料框
future_dates = pd.date_range(start=df['日期'].max() + timedelta(days=1), periods=future_days, freq='D')
forecast_df = pd.DataFrame({
    '日期': future_dates,
    '預測訪客數': predictions.astype(int)
})

# 合併歷史與預測資料
historical_df = df[['日期', '訪客數']].copy()
historical_df['類型'] = '歷史資料'
forecast_plot_df = forecast_df.copy()
forecast_plot_df['訪客數'] = forecast_plot_df['預測訪客數']
forecast_plot_df['類型'] = 'AI預測'

combined_df = pd.concat([
    historical_df[['日期', '訪客數', '類型']], 
    forecast_plot_df[['日期', '訪客數', '類型']]
])

# 繪製預測圖表
fig_forecast = px.line(
    combined_df,
    x='日期',
    y='訪客數',
    color='類型',
    title='訪客數歷史與 AI 預測',
    markers=True,
    color_discrete_map={'歷史資料': '#636EFA', 'AI預測': '#EF553B'}
)
fig_forecast.update_layout(height=400)
st.plotly_chart(fig_forecast, use_container_width=True)

# 顯示預測數據表
col1, col2 = st.columns([2, 1])
with col1:
    st.dataframe(forecast_df, use_container_width=True)
with col2:
    avg_forecast = int(predictions.mean())
    st.metric("未來7天預測平均訪客數", f"{avg_forecast:,}")
    
    # 趨勢判斷
    trend = "上升 📈" if predictions[-1] > predictions[0] else "下降 📉"
    st.metric("預測趨勢", trend)

# 頁尾
st.markdown("---")
st.markdown("💡 **AI 協作說明**：此儀表板透過與 Claude AI 協作開發完成")