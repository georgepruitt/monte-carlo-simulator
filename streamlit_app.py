import streamlit as st
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt

# --- Page Config (This sets the browser tab title) ---
st.set_page_config(page_title="Monte Carlo Trade Flight Simulator", layout="wide")

# --- Main Page Title ---
st.title("🚀 Monte Carlo Trade Flight Simulator")
st.markdown(
    """
This tool uses **Monte Carlo Simulation** to stress-test your trading strategy.
Paste your trades and discover your true Risk of Ruin and Scaling potential.

**Update:** You no longer need to know "Trades per Year". The simulator automatically uses
**the number of trades you paste** as the simulation length.
"""
)

# --- Sidebar Inputs ---
st.sidebar.header("Simulation Controls")
base_equity = st.sidebar.number_input("Base Start Equity ($)", value=25000, step=1000)
ruin_level = st.sidebar.number_input("Margin/Ruin Level ($)", value=5000, step=500)

st.sidebar.markdown("---")
raw_text = st.sidebar.text_area(
    "Paste Trades (Column of P/L)",
    height=300,
    help=(
        "Paste a list of trade profits/losses. If your paste includes a trade number column (1, 2, 3, ...), "
        "the simulator will ignore that and use the P/L column. Symbols and headers are ignored."
    ),
)

# --- Data Parser ---
_NUM_TOKEN = re.compile(r"\(?-?\$?\d{1,3}(?:,\d{3})*(?:\.\d+)?\)?|\(?-?\$?\d+(?:\.\d+)?\)?")


def _token_to_float(tok: str) -> float | None:
    """Convert a token like '(1,234.56)' or '-$1234' or '1234' into float."""
    if not tok:
        return None

    t = tok.strip()

    # Skip headers / obvious non-values (belt + suspenders)
    tl = t.lower()
    if any(x in tl for x in ("pnl", "trade", "amount", "profit", "loss", "symbol")):
        return None

    is_neg = ("(" in t and ")" in t) or ("-" in t)
    # Keep digits + decimal only
    val = re.sub(r"[^\d.]", "", t)
    if not val or val == ".":
        return None

    try:
        num = float(val)
    except ValueError:
        return None

    return -num if is_neg else num


def clean_data(text: str) -> list[float]:
    """
    Robustly extract a column of P/L values from pasted text.

    Handles:
      - TradeStation exports where each row might be:  '1   $123.45'  (trade # + P/L)
      - Parentheses negatives: '(123.45)'
      - Commas, currency symbols
      - Random extra columns (dates, symbols, etc.)

    Strategy:
      - For each line, extract numeric-looking tokens.
      - If the first token is a small integer (often a trade # like 1,2,3,...) and there is another number,
        drop that first token.
      - Use the **rightmost remaining numeric token** as the P/L for that line.
    """
    cleaned: list[float] = []

    for line in (text or "").splitlines():
        s = line.strip()
        if not s:
            continue

        # If the line is just an integer (common when copying a Trade # column), skip it.
        # This directly prevents the "454 trades becomes 908" problem.
        if s.isdigit():
            continue

        # Fast header skip
        ll = line.lower()
        if any(h in ll for h in ("p/l", "pnl", "profit", "loss", "net", "strategy", "symbol")):
            continue

        tokens = _NUM_TOKEN.findall(line)
        if not tokens:
            continue

        # Convert all tokens we can
        values: list[tuple[str, float]] = []
        for tok in tokens:
            v = _token_to_float(tok)
            if v is not None:
                values.append((tok, v))

        if not values:
            continue

        # If line looks like: trade_index + pnl (e.g., '1  250.00'), drop the index.
        if len(values) >= 2:
            first_tok, first_val = values[0]
            # Heuristic: pure integer token, small-ish magnitude, and no sign/parentheses
            is_integer_token = re.fullmatch(r"\d+", first_tok.strip()) is not None
            looks_like_index = is_integer_token and 0 <= first_val <= 100000
            if looks_like_index:
                # In particular, your prior bugbear: skip the leading '1'
                values = values[1:]

        if not values:
            continue

        # Use the rightmost numeric as P/L
        cleaned.append(values[-1][1])

    return cleaned


# --- Main Logic ---
if st.sidebar.button("RUN FLIGHT SIMULATOR"):
    trades = clean_data(raw_text)

    if not trades:
        st.error("Please paste some trade P/L data first!")
    else:
        n_trades = int(len(trades))
        st.sidebar.success(f"Detected {n_trades} trades (simulation length = {n_trades}).")

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
                sim = np.random.choice(trades, size=n_trades, replace=True)
                path = [current_start]
                peak, mdd, ruined = current_start, 0.0, False

                for t in sim:
                    new_val = path[-1] + t
                    path.append(new_val)

                    if new_val > peak:
                        peak = new_val

                    drawdown = (peak - new_val) / peak if peak > 0 else 0
                    if drawdown > mdd:
                        mdd = drawdown

                    if new_val < ruin_level:
                        ruined = True

                if ruined:
                    ruins += 1

                profits.append(path[-1] - current_start)
                dds.append(mdd)

                if step == 0:
                    base_paths.append(path)
                    final_equities.append(path[-1])
                    base_dds.append(mdd)

            progress_bar.progress((step + 1) / 11)

            # --- Calculations for Display ---
            med_p, m_dd = np.median(profits), np.median(dds)
            worst_case = np.percentile(profits, 1)
            med_ret_pct = (med_p / current_start) * 100

            table_data.append(
                {
                    "Start Equity": f"${current_start:,.0f}",
                    "Risk of Ruin %": f"{int(ruins/10)}%",
                    "Median Drawdown": f"{m_dd*100:.1f}%",
                    "Median Profit ($ / % Return)": f"${med_p:,.0f} / {int(med_ret_pct)}%",
                    "Worst Case (1st %-tile)": f"${worst_case:,.0f}",
                    "Efficiency (Ret/DD)": round((med_p/current_start)/m_dd, 2) if m_dd > 0 else 0,
                    "Prob > 0": f"{int(sum(1 for p in profits if p > 0)/10)}%",
                }
            )

        # --- Display Table ---
        st.subheader("📊 Risk & Scaling Analysis")
        df = pd.DataFrame(table_data)
        st.table(df)

        # --- Download Button ---
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download Results as CSV",
            data=csv,
            file_name="flight_results.csv",
            mime="text/csv",
        )

        # --- Charts ---
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📈 The Journey")
            fig1, ax1 = plt.subplots()
            p5, p50, p95 = np.percentile(base_paths, [5, 50, 95], axis=0)
            ax1.fill_between(
                range(n_trades + 1),
                p5,
                p95,
                color="royalblue",
                alpha=0.15,
                label="90% Range",
            )
            ax1.plot(range(n_trades + 1), p50, color="darkblue", lw=2, label="Median Path")
            ax1.axhline(ruin_level, color="red", ls="--", label="Ruin Level")
            ax1.set_xlabel("Trade #")
            ax1.set_ylabel("Account Balance ($)")
            ax1.legend()
            st.pyplot(fig1)

        with col2:
            st.subheader("🎯 The Destination")
            fig2, ax2 = plt.subplots()
            ax2.hist(final_equities, bins=40, color="forestgreen", alpha=0.6, edgecolor="white")
            ax2.axvline(base_equity, color="black", ls="--", label="Break-Even")
            ax2.set_xlabel("Ending Balance ($)")
            ax2.legend()
            st.pyplot(fig2)

        st.subheader("⚖️ Efficiency Cloud (Pain vs. Gain)")
        fig3, ax3 = plt.subplots(figsize=(10, 4))
        net_profits = [e - base_equity for e in final_equities]
        ax3.scatter(base_dds, net_profits, alpha=0.2, s=8, c="purple")
        ax3.axvline(np.median(base_dds), color="black", ls=":", alpha=0.7)
        ax3.axhline(np.median(net_profits), color="black", ls=":", alpha=0.7)
        ax3.set_xlabel("Max Drawdown (fraction)")
        ax3.set_ylabel("Net Profit ($)")
        st.pyplot(fig3)

else:
    st.info("👈 Enter your parameters and paste your trades in the sidebar, then click 'Run Flight Simulator'.")

# --- Documentation Section ---
st.markdown("---")
st.header("📖 How to Interpret Your Flight Results")

with st.expander("🔍 Understanding the Metrics"):
    st.markdown(
        """
    ### **The Risk of Ruin %**
    This is the most critical number. It represents the percentage of simulations that hit your **'Margin/Ruin Level'** at any point during the run.
    * **Quant Tip:** If this is above 5%, you are likely over-leveraged for your current account size.

    ### **Median Drawdown**
    The 'middle' drawdown experienced across 1,000 lives. While your strategy might have a historical drawdown of 15%, the Monte Carlo might show a median of 22%. This is the 'Reality Gap'—preparing you for the sequence of trades you *haven't* seen yet.

    ### **Worst Case (1st Percentile)**
    This shows the profit/loss of the unluckiest 1% of simulated runs. If your 'Worst Case' is -$20,000 and your Start Equity is only $15,000, the math says you have a structural risk of total loss.

    ### **Efficiency (Ret/DD Ratio)**
    Also known as the **MAR Ratio**. This measures your return per unit of pain.
    * **> 1.0:** Excellent. You are making more than the risk you are taking.
    * **< 0.5:** High friction. You are enduring a lot of 'pain' for every dollar of 'gain.'
    """
    )

with st.expander("📈 Reading the Visuals"):
    st.markdown(
        """
    ### **The 'Broom' Chart (The Journey)**
    The dark blue line is your **Median Path**. The shaded area is the **90% Confidence Interval**. If the bottom of that shaded area touches your red Ruin line, your strategy is 'flying too low to the ground.'

    ### **The Destination (Histogram)**
    This shows the distribution of ending balances. A 'fat' tail to the left means your strategy has 'Left Tail Risk' (large, infrequent losses). A 'tight' cluster means your results are highly predictable.
    """
    )

st.sidebar.markdown("---")
st.sidebar.caption(
    """
**Disclaimer:** This simulator is for educational purposes only. Past performance (the trades you paste) is not a guarantee of future results.
Monte Carlo analysis is a mathematical model and cannot account for 'Black Swan' events or changes in market regime.
"""
)
