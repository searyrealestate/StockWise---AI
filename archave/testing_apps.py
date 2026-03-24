import streamlit as st
import pandas as pd
from advanced_testing_features import AdvancedAlgorithmTester
from datetime import date
from algo_testing_script import AlgorithmTester


class AdvancedAlgorithmTester(AlgorithmTester):

    st.set_page_config(layout="wide", page_title="Algorithm Testing Suite", initial_sidebar_state="expanded")

    # Custom Styling
    st.markdown("""
        <style>
        .pass { color: limegreen; font-weight: bold; }
        .fail { color: red; font-weight: bold; }
        .partial { color: orange; font-weight: bold; }
        .metric-box { background-color: #262730; padding: 10px; border-radius: 10px; }
        </style>
    """, unsafe_allow_html=True)

    st.sidebar.title("🧪 Test Configuration")
    stock_input = st.sidebar.text_input("Stock Symbols (comma-separated)", value="AAPL,GOOGL,MSFT,NVDA")
    start_date = st.sidebar.date_input("Start Date", value=date(2024, 1, 1))
    end_date = st.sidebar.date_input("End Date", value=date(2024, 12, 31))
    frequency = st.sidebar.selectbox("Frequency", ["weekly", "daily"])
    holding_days = st.sidebar.slider("Holding Days", min_value=1, max_value=30, value=7)

    start_test = st.sidebar.button("🚀 Run Test")

    if start_test:
        st.info("Running test... please wait ⏳")
        symbols = [s.strip().upper() for s in stock_input.split(",") if s.strip()]

        tester = AdvancedAlgorithmTester(debug=False)
        results = tester.run_comprehensive_test(symbols, start_date.isoformat(), end_date.isoformat(), frequency,
                                                holding_days)
        df = pd.DataFrame(results)

        if not df.empty:
            st.success("✅ Test Complete!")
            st.subheader("📊 Test Summary")

            success_df = df[df["status"] == "SUCCESS"]
            pass_count = len(success_df[success_df["test_result"] == "PASS"])
            fail_count = len(success_df[success_df["test_result"] == "FAIL"])
            partial_count = len(success_df[success_df["test_result"] == "PARTIAL_PASS"])

            st.markdown(f"""
            <div class="metric-box">
            ✅ <span class="pass">PASS:</span> {pass_count}  
            🔶 <span class="partial">PARTIAL:</span> {partial_count}  
            ❌ <span class="fail">FAIL:</span> {fail_count}  
            </div>
            """, unsafe_allow_html=True)

            st.write("📈 Detailed Results")
            df_display = df[
                ["symbol", "test_date", "action", "confidence", "predicted_profit_pct", "actual_profit_pct", "test_result"]]
            df_display["test_result"] = df_display["test_result"].map({
                "PASS": '<span class="pass">PASS</span>',
                "PARTIAL_PASS": '<span class="partial">PARTIAL</span>',
                "FAIL": '<span class="fail">FAIL</span>'
            })
            st.write(df_display.to_html(escape=False), unsafe_allow_html=True)

            st.subheader("📤 Download Reports")
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download CSV", data=csv, file_name="algorithm_test_results.csv", mime="text/csv")

            # Chart Section
            st.subheader("📉 Charts")
            try:
                tester.results = results
                tester.generate_performance_charts(save_charts=False)
            except Exception as e:
                st.error(f"⚠️ Could not generate charts: {e}")
