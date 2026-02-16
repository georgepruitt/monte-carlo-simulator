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
**the number of trades you paste** as the simulation length (a run of *N trades*).

**Return definition:** The “% Return” shown is **profit as a percent of starting equity over the pasted trade list**
(i.e., over *N trades*). For convenience we treat that as a “1-year” run in the table.
"""
)

# --- Sidebar Inputs ---
st.sidebar.header("Simulation Controls")
base_equity = st.sidebar.number_input("Base Start Equity ($)", value=25000, step=1000)
ruin_level = st.sidebar.number_input("Margin/Ruin Level ($)", value=5000, step=500)

st.sidebar.markdown("---")
raw_text = st.sidebar.text_area(
    "Paste Trades (Shares/Ctrts + Profit/Loss or just P/L)",
    height=300,
    help=(
        "Paste a list of trade profits/losses. If your paste includes a Shares/Ctrts column (often a repeated '1'), "
        "the simulator ignores those integer-only lines and uses the Profit/Loss values. Parentheses like ($2,000.00) "
        "are treated as negative."
    ),
)

# --- Data Parser ---
_NUM_TOKEN = re.compile(
    r"\(?-?\$?\d{1,3}(?:,\d{3})*(?:\.\d+)?\)?|\(?-?\$?\d+(?:\.\d+)?\)?"
)


def _token_to_float(tok: str):
    """Convert a token like '(1,234.56)' or '-$1234' or '1234' into float."""
    if not tok:
        return None

    t = tok.strip()

    # Skip headers / obvious non-values
    tl = t.lower()
    if any(x in tl for x in ("pnl", "trade", "amount", "profit", "loss", "symbol", "shares", "ctrts")):
        return None

    is_neg = ("(" in t and ")" in t) or ("-" in t)
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
    Extract a column of P/L values from pasted text.

    Supports common TradeStation paste format:
        Shares/Ctrts - Profit/Loss
        1
        ($845.00)
        1
        $870.00
        ...

    Rules:
      - If a line is ONLY an integer ("1", "2", ...), treat it as Shares/Ctrts or an index and skip it.
      - Otherwise, extract numeric tokens.
      - If a line contains multiple numeric tokens, drop a leading integer index if present.
      - Use the rightmost remaining number as the line's P/L.
    """
    cleaned: list[float] = []

    for line in (text or "").splitlines():
        s = line.strip()
        if not s:
            continue

        # Two-column paste often becomes: integer-only line, then P/L line. Skip pure integer lines.
        if s.isdigit():
            continue

        # Fast header skip
        ll = s.lower()
        if any(h in ll for h in ("p/l", "pnl", "profit/loss", "profit", "loss", "net", "strategy", "symbol", "shares", "ctrts")):
            continue

        tokens = _NUM_TOKEN.findall(s)
        if not tokens:
            continue

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
            is_integer_token = re.fullmatch(r"\d+", first_tok.strip()) is not None
            if is_integer_token and 0 <= first_val <= 100000:
                values = values[1:]

        if not values:
            continue

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

        # --- Parse sanity check (helps users confirm negatives are being detected) ---
        neg_ct = sum(1 for x in trades if x < 0)
        pos_ct = sum(1 for x in trades if x > 0)
        zero_ct = len(trades) - neg_ct - pos_ct
        min_t = float(min(trades))
        max_t = float(max(trades))
        med_t = float(np.median(trades))
        avg_t = float(np.mean(trades))
        sum_t = float(np.sum(trades))
        exp_pct_of_start = (sum_t / float(base_equity)) * 100 if float(base_equity) != 0 else 0.0

        st.caption(
            f"Parsed trades: {len(trades)}  |  Winners: {pos_ct}  Losers: {neg_ct}  Zeros: {zero_ct}  |  "
            f"Avg trade: ${avg_t:,.2f}  |  Sum P/L: ${sum_t:,.2f} ({exp_pct_of_start:.1f}% of start)  |  "
            f"Median trade: ${med_t:,.2f}  |  Worst: ${min_t:,.2f}  Best: ${max_t:,.2f}"
        )
        if neg_ct == 0:
            st.warning(
                "No losing trades were detected. If your losers are formatted like ($2,000.00), "
                "this usually means the paste did not include the losing rows (or the wrong column was pasted)."
            )

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
            med_p = float(np.median(profits))
            m_dd = float(np.median(dds))
            worst_case = float(np.percentile(profits, 1))

            # Return over the simulation horizon (N trades)
            ret_pct = (med_p / current_start) * 100

            table_data.append(
                {
                    "Start Equity": f"${current_start:,.0f}",
                    "Risk of Ruin %": f"{int(ruins/10)}%",
                    "Median Drawdown": f"{m_dd*100:.1f}%",
                    "Median Profit ($ / % of Start)": f"${med_p:,.0f} / {ret_pct:.1f}%",
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
            ax1.set_xlabel("Trade # (simulation step)")
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
        ax3.set_xlabel("Max Drawdown (fraction of peak)")
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
    This shows the profit/loss of the unluckiest 1% of simulated runs.

    ### **Efficiency (Ret/DD Ratio)**
    This is (Return as a fraction of start equity) divided by (Drawdown as a fraction of peak). It is a “return per unit of pain” metric over the pasted trade list.

    ### **Median Profit ($ / % of Start)**
    This is the median profit across 1,000 simulations expressed in dollars and as a percent of starting equity over the pasted trade list (N trades).
    """
    )

with st.expander("📈 Reading the Visuals"):
    st.markdown(
        """
    ### **The 'Broom' Chart (The Journey)**
    The dark blue line is your **Median Path**. The shaded area is the **90% Confidence Interval**. If the bottom of that shaded area touches your red Ruin line, your strategy is 'flying too low to the ground.'

    ### **The Destination (Histogram)**
    This shows the distribution of ending balances.
    """
    )

st.sidebar.markdown("---")
st.sidebar.caption(
    """
**Disclaimer:** This simulator is for educational purposes only. Past performance (the trades you paste) is not a guarantee of future results.
Monte Carlo analysis is a mathematical model and cannot account for 'Black Swan' events or changes in market regime.
"""
)
