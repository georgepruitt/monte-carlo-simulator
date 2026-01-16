import streamlit as st
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt

# --- Page Config ---
st.set_page_config(page_title="Trading Flight Simulator", layout="wide")

st.title("🚀 Trading Flight Simulator")
st.markdown("""
This tool uses **Monte Carlo Simulation** to stress-test your trading strategy. 
Paste your trades and discover your true Risk of Ruin and Scaling potential.
""")

# --- Sidebar Inputs ---
st.sidebar.header("Simulation Controls")
base_equity = st.sidebar.number_input("Base Start Equity ($)", value=10000, step=1000)
ruin_level = st.sidebar.number_input("Margin/Ruin Level ($)", value=3000, step=500)
n_trades = st.sidebar.number_input("Trades per Year", value=439, step=1)

st.sidebar.markdown("---")
raw_text = st.sidebar.text_area("Paste Trades (Column of P/L)", height=300, help="Paste a list of trade profits/losses. Symbols and headers are ignored.")

# --- Data Parser ---
def clean_data(text):
    cleaned = []
    for line in text.splitlines():
        parts = line.replace(',', '').split()
        for part in parts:
            if not part or any(x in part.lower() for x in ["pnl", "trade", "amount", "profit"]) or part == "1":
                continue
            is_neg = "(" in part or "-" in part
            val = re.sub(r'[^\d.]', '', part)
            if val:
                num = float(val)
                cleaned.append(-num if is_neg else num)
    return cleaned

# --- Main Logic ---
if st.sidebar.button("RUN FLIGHT SIMULATOR"):
    trades = clean_data(raw_text)
    
    if not trades:
        st.error("Please paste some trade data first!")
    else:
        # Progress bar for web UX
        progress_bar = st.progress(0)
        
        increment = base_equity / 4
        table_data = []
        base_paths = []
        final_equities = []
        base_dds = []

        for step in range(11):
            current_start = base_equity + (step * increment)
            ruins, profits, dds = 0, [], []
            
            for _ in range(1000):
                sim = np.random.choice(trades, size=int(n_trades), replace=True)
                path = [current_start]
                peak, mdd, ruined = current_start, 0, False
                for t in sim:
                    new_val = path[-1] + t
                    path.append(new_val)
                    if new_val > peak: peak = new_val
                    drawdown = (peak - new_val) / peak if peak > 0 else 0
                    if drawdown > mdd: mdd = drawdown
                    if new_val < ruin_level: ruined = True
                
                if ruined: ruins += 1
                profits.append(path[-1] - current_start)
                dds.append(mdd)
                
                if step == 0: 
                    base_paths.append(path)
                    final_equities.append(path[-1])
                    base_dds.append(mdd)
            
            progress_bar.progress((step + 1) / 11)

            med_p, m_dd = np.median(profits), np.median(dds)
            worst_case = np.percentile(profits, 1)
            
            table_data.append({
                "Start Equity": f"${current_start:,.0f}",
                "Risk of Ruin": f"{int(ruins/10)}%",
                "Med Drawdown": f"{m_dd*100:.1f}%",
                "Med $ Profit": f"${med_p:,.0f}",
                "Med Return %": f"{int((med_p/current_start)*100)}%",
                "Worst Case (1%)": f"${worst_case:,.0f}",
                "Ret/DD": round((med_p/current_start)/m_dd, 2) if m_dd > 0 else 0,
                "Prob > 0": f"{int(sum(1 for p in profits if p > 0)/10)}%"
            })

        # --- Display Table ---
        st.subheader("📊 Risk & Scaling Analysis")
        df = pd.DataFrame(table_data)
        st.table(df)

        # --- Charts ---
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📈 The Journey")
            fig1, ax1 = plt.subplots()
            p5, p50, p95 = np.percentile(base_paths, [5, 50, 95], axis=0)
            ax1.fill_between(range(int(n_trades)+1), p5, p95, color='royalblue', alpha=0.15, label='90% Range')
            ax1.plot(range(int(n_trades)+1), p50, color='darkblue', lw=2, label='Median Path')
            ax1.axhline(ruin_level, color='red', ls='--', label='Ruin Level')
            ax1.set_ylabel("Account Balance ($)")
            ax1.legend()
            st.pyplot(fig1)

        with col2:
            st.subheader("🎯 The Destination")
            fig2, ax2 = plt.subplots()
            ax2.hist(final_equities, bins=40, color='forestgreen', alpha=0.6, edgecolor='white')
            ax2.axvline(base_equity, color='black', ls='--', label='Break-Even')
            ax2.set_xlabel("Year-End Balance ($)")
            st.pyplot(fig2)

        st.subheader("⚖️ Efficiency Cloud (Pain vs. Gain)")
        fig3, ax3 = plt.subplots(figsize=(10, 4))
        net_profits = [e - base_equity for e in final_equities]
        ax3.scatter(base_dds, net_profits, alpha=0.2, s=8, c='purple')
        ax3.axvline(np.median(base_dds), color='black', ls=':', alpha=0.7)
        ax3.axhline(np.median(net_profits), color='black', ls=':', alpha=0.7)
        ax3.set_xlabel("Max Drawdown (%)")
        ax3.set_ylabel("Net Profit ($)")
        st.pyplot(fig3)

else:
    st.info("👈 Enter your parameters and paste your trades in the sidebar, then click 'Run Flight Simulator'.")
