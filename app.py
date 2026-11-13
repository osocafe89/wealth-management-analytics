import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import re
import io
import warnings

warnings.filterwarnings('ignore')

# --- BRANDING & PAGE CONFIG ---
st.set_page_config(page_title="Wealth Management Analytics", layout="wide", page_icon="📊")

# Set global font to Montserrat (Falls back to Arial/sans-serif if not installed)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Montserrat', 'Arial', 'sans-serif']

# Define Brand Colors
brand_primary_dark = '#001B44'
brand_primary = '#00457C'
brand_secondary = '#3167B8'
brand_neutral = '#E4E4E8'

brand_palette = [
    brand_primary_dark, brand_primary, brand_secondary,
    '#5A89CB', '#85A9D6', '#B0C8E2', '#D3E0F0', brand_neutral
]
brand_cmap = LinearSegmentedColormap.from_list("BrandGradient", [brand_neutral, brand_secondary, brand_primary_dark])

if 'seaborn-v0_8-whitegrid' in plt.style.available:
    plt.style.use('seaborn-v0_8-whitegrid')
else:
    plt.style.use('default')


# --- HELPER FUNCTION: DOWNLOAD BUTTON ---
def create_download_button(fig, filename):
    """Converts a matplotlib figure to a PNG image in memory for downloading."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
    buf.seek(0)
    st.download_button(
        label="📥 Download Chart",
        data=buf,
        file_name=filename,
        mime="image/png"
    )


# --- UI HEADER ---
st.title("📊 Wealth Management Analytics")
st.markdown("Interactive operational reporting, text themes, and illustrative trend projections.")
st.caption("Portfolio demonstration using fictional data. Trend projections are simple linear extrapolations, not validated financial predictions.")

# --- DATA SOURCE ---
from pathlib import Path
DEMO_DIR = Path(__file__).parent / "demo_data"
with st.sidebar:
    st.header("📂 1. Data Source")
    source = st.radio("Choose data", ["Fictional sample reports", "Upload Excel reports"], index=0)
    if source == "Fictional sample reports":
        st.success("Fictional sample reports are loaded by default.")
        issues_path = DEMO_DIR / "sample_issues.csv"
        losses_path = DEMO_DIR / "sample_losses.csv"
        st.download_button("Download sample Issues CSV", issues_path.read_bytes(), file_name="sample_issues.csv")
        st.download_button("Download sample Losses CSV", losses_path.read_bytes(), file_name="sample_losses.csv")
        file_issues = issues_path
        file_losses = losses_path
    else:
        st.warning("Do not upload real client information or confidential business data to this public demo.")
        file_issues = st.file_uploader("Upload Main Issues Report", type=["xlsx"])
        file_losses = st.file_uploader("Upload Losses Report", type=["xlsx"])
    st.markdown("---")


# --- CORE PIPELINE ---
@st.cache_data
def process_data(df, report_name, col_date, col_advisor, col_category, col_text, col_account):
    # Ensure proper missing value handling
    df = df.copy()
    df[col_text] = df[col_text].fillna('')
    df[col_category] = df[col_category].fillna('Other')
    df[col_advisor] = df[col_advisor].fillna('Unknown Advisor')
    df[col_account] = df[col_account].fillna('Unknown Account')

    # Parse Dates to allow filtering
    df[col_date] = pd.to_datetime(df[col_date], errors='coerce')
    df = df.dropna(subset=[col_date])
    df['YearMonth'] = df[col_date].dt.to_period('M')

    total_volume = len(df)

    # ---------------------------------------------------------
    # HISTORICAL BUSINESS INSIGHTS
    # ---------------------------------------------------------
    monthly_counts = df.groupby('YearMonth').size()
    hist_avg_monthly = int(monthly_counts.mean()) if not monthly_counts.empty else 0
    peak_month = monthly_counts.idxmax().strftime('%B %Y') if not monthly_counts.empty else "N/A"
    peak_month_vol = monthly_counts.max() if not monthly_counts.empty else 0

    top_category = df[col_category].mode()[0] if not df.empty else "N/A"
    top_cat_vol = len(df[df[col_category] == top_category])
    top_cat_pct = (top_cat_vol / total_volume * 100) if total_volume > 0 else 0

    top_3_reps = df[col_advisor].value_counts().head(3)
    top_3_reps_vol = top_3_reps.sum()
    top_3_reps_pct = (top_3_reps_vol / total_volume * 100) if total_volume > 0 else 0

    def get_first_name(name):
        name_str = str(name).strip()
        return name_str.split(',')[1].strip() if ',' in name_str else (name_str.split()[0] if name_str else '')

    # Matrix logic: Keep top 10 for the stacked bar chart so it's readable
    top_10_advisors = df[col_advisor].value_counts().head(10).index.tolist()
    top_10_advisors_alpha = sorted(top_10_advisors, key=get_first_name)

    df_top_advisors = df[df[col_advisor].isin(top_10_advisors)].copy()
    if not df_top_advisors.empty:
        advisor_category_matrix = pd.crosstab(df_top_advisors[col_advisor], df_top_advisors[col_category]).loc[
            top_10_advisors_alpha]
    else:
        advisor_category_matrix = pd.DataFrame()

    # Dropdown logic: Get ALL reps for the individual scorecard
    all_advisors = df[col_advisor].dropna().unique().tolist()
    all_advisors_alpha = sorted(all_advisors, key=get_first_name)
    all_rep_monthly_trend = df.groupby([col_advisor, 'YearMonth']).size().unstack(fill_value=0)

    # NLP Keywords
    def clean_text_basic(text):
        if pd.isna(text): return ''
        return re.sub(r'[^a-z\s]', '', str(text).lower())

    df['Clean_Text'] = df[col_text].apply(clean_text_basic)
    vectorizer = CountVectorizer(stop_words='english', ngram_range=(1, 2), max_features=8,
                                 token_pattern=r'(?u)\b[a-zA-Z]{3,}\b')
    try:
        X_keywords = vectorizer.fit_transform(df['Clean_Text'])
        top_keywords = vectorizer.get_feature_names_out()
        keyword_df = pd.DataFrame(X_keywords.toarray(), columns=[f"kw_{kw.replace(' ', '_')}" for kw in top_keywords],
                                  index=df.index)
        df_ml = pd.concat([df.reset_index(drop=True), keyword_df.reset_index(drop=True)], axis=1)
    except ValueError:
        top_keywords = []
        df_ml = df

    # ---------------------------------------------------------
    # PREDICTIVE ML INSIGHTS (Objective Reporting)
    # ---------------------------------------------------------
    category_monthly_trend = df_ml.groupby(['YearMonth', col_category]).size().unstack(fill_value=0)

    kw_columns = [col for col in df_ml.columns if col.startswith('kw_')]
    if kw_columns:
        keyword_monthly_trend = df_ml.groupby('YearMonth')[kw_columns].sum()
        keyword_monthly_trend.columns = [col.replace('kw_', '').replace('_', ' ').title() for col in kw_columns]
    else:
        keyword_monthly_trend = pd.DataFrame()

    ml_model = LinearRegression()
    X_time = np.arange(len(category_monthly_trend)).reshape(-1, 1)
    future_months = 3
    X_future = np.arange(len(X_time) + future_months).reshape(-1, 1)

    original_dates = category_monthly_trend.index.astype(str).tolist()
    last_date = category_monthly_trend.index[-1] if not category_monthly_trend.empty else None
    all_dates = original_dates + [str(last_date + i) for i in range(1, future_months + 1)] if last_date else []

    forecast_insights = []
    forecast_plot_data = {}

    for col in category_monthly_trend.columns:
        y_actuals = category_monthly_trend[col].values
        if len(y_actuals) < 3: continue

        ml_model.fit(X_time, y_actuals)
        slope = ml_model.coef_[0]
        y_forecast_full = np.maximum(ml_model.predict(X_future), 0)

        forecast_plot_data[col] = {
            'y_actuals': y_actuals,
            'y_forecast_full': y_forecast_full
        }

        projected_quarter_vol = int(np.sum(y_forecast_full[-3:]))
        recent_quarter_vol = int(np.sum(y_actuals[-3:]))
        net_change = projected_quarter_vol - recent_quarter_vol

        # Objectively report trend status without suggesting actions
        if slope > 0.5 and recent_quarter_vol > (np.mean(y_actuals) * 3 * 1.2):
            status, obs = "🔴 CRITICAL", "Volume is accelerating rapidly above historical averages."
        elif slope > 0.1:
            status, obs = "🟡 WATCHLIST", "Volume demonstrates steady upward growth."
        elif slope < -0.5:
            status, obs = "🟢 IMPROVING", "Volume trend is actively declining."
        else:
            status, obs = "⚪ STABLE", "Volume remains within normal historical variance."

        forecast_insights.append({
            'category': col, 'slope': slope, 'projected_vol': projected_quarter_vol,
            'net_change': net_change, 'status': status, 'observation': obs
        })

    forecast_insights.sort(key=lambda x: x['slope'], reverse=True)

    # Account Clustering
    df_accounts = df_ml[df_ml[col_account] != 'Unknown Account'].copy()
    account_category_matrix = df_accounts.groupby([col_account, col_category]).size().unstack(fill_value=0)
    scaler = StandardScaler()

    if len(account_category_matrix) >= 4 and len(account_category_matrix.drop_duplicates()) >= 4:
        scaled_accounts = scaler.fit_transform(account_category_matrix)
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        account_category_matrix['Cluster'] = kmeans.fit_predict(scaled_accounts)
        cluster_profiles = account_category_matrix.groupby('Cluster').mean()
    else:
        cluster_profiles = pd.DataFrame()

    top_10_accounts = df_accounts[col_account].value_counts().head(10).index.tolist()
    top_account_matrix = account_category_matrix.loc[
        top_10_accounts] if not account_category_matrix.empty else pd.DataFrame()
    if 'Cluster' in top_account_matrix.columns:
        top_account_matrix = top_account_matrix.drop(columns=['Cluster'])

    # Extract email follow up string for Issues report
    top_outcome_str = ""
    if report_name == "Issues" and 'Follow up email sent' in df_ml.columns:
        df_ml['Clean_Follow_Up'] = df_ml['Follow up email sent'].apply(clean_text_basic)
        outcome_vectorizer = CountVectorizer(stop_words='english', ngram_range=(1, 2), max_features=5)
        try:
            outcome_vectorizer.fit(df_ml['Clean_Follow_Up'])
            top_outcome_str = ", ".join([kw.title() for kw in outcome_vectorizer.get_feature_names_out()])
        except ValueError:
            top_outcome_str = "Insufficient data"

    return {
        "report_name": report_name,
        "hist_total": total_volume,
        "hist_avg_monthly": hist_avg_monthly,
        "hist_peak": peak_month,
        "hist_peak_vol": peak_month_vol,
        "hist_top_cat": top_category,
        "hist_top_cat_pct": top_cat_pct,
        "hist_top_3_reps_pct": top_3_reps_pct,
        "top_keywords_str": ", ".join([kw.replace('_', ' ').title() for kw in top_keywords[:3]]),
        "top_outcome_str": top_outcome_str,
        "insights": forecast_insights,
        "cluster_profiles": cluster_profiles,
        "advisor_category_matrix": advisor_category_matrix,
        "keyword_monthly_trend": keyword_monthly_trend,
        "top_account_matrix": top_account_matrix,
        "all_rep_monthly_trend": all_rep_monthly_trend,
        "all_advisors_alpha": all_advisors_alpha,
        "forecast_plot_data": forecast_plot_data,
        "original_dates": original_dates,
        "all_dates": all_dates,
        "total_months": len(category_monthly_trend)
    }


# --- RENDERING DASHBOARD VISUALS ---
def render_dashboard(data):
    st.markdown("---")

    if data['advisor_category_matrix'].empty:
        st.warning("Not enough data in the selected timeframe to generate charts.")
        return

    col1, col2 = st.columns(2)

    # PLOT 1: Top 10 Reps (Bar Chart)
    with col1:
        st.subheader(f"{data['report_name']} by Category: Highest Volume Reps")
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        data['advisor_category_matrix'].plot(kind='bar', stacked=True, color=brand_palette, ax=ax1)
        ax1.set_ylabel("Reported Volume", color=brand_primary)
        ax1.set_xlabel("Rep/Advisor", color=brand_primary)
        ax1.grid(axis='y', linestyle='--', alpha=0.6, color=brand_neutral)
        plt.xticks(rotation=45, ha='right')
        plt.legend(title="Category", bbox_to_anchor=(1.05, 1), loc='upper left')
        st.pyplot(fig1)
        create_download_button(fig1, f"{data['report_name']}_Top_Reps.png")

    # PLOT 5: Top 10 Accounts
    with col2:
        st.subheader(f"{data['report_name']} by Category: Highest Volume Accounts")
        fig5, ax5 = plt.subplots(figsize=(8, 5))
        if not data['top_account_matrix'].empty:
            data['top_account_matrix'].plot(kind='bar', stacked=True, color=brand_palette, ax=ax5)
            ax5.set_ylabel("Reported Volume", color=brand_primary)
            ax5.set_xlabel("Account Number", color=brand_primary)
            ax5.grid(axis='y', linestyle='--', alpha=0.6, color=brand_neutral)
            plt.xticks(rotation=45, ha='right')
            plt.legend(title="Category", bbox_to_anchor=(1.05, 1), loc='upper left')
            st.pyplot(fig5)
            create_download_button(fig5, f"{data['report_name']}_Top_Accounts.png")
        else:
            st.info("Insufficient account data.")

    # PLOT 3: ML Forecasts
    st.markdown("---")
    st.subheader("Category Volume Trend Extrapolation (Next 3 Months)")
    if data['total_months'] < 3:
        st.warning("⚠️ Please select a date range of at least 3 months to generate trend extrapolations.")
    else:
        forecast_cats = list(data['forecast_plot_data'].keys())
        num_cats = len(forecast_cats)
        cols_count = 2
        rows_count = int(np.ceil(num_cats / cols_count))

        if rows_count > 0:
            fig3, axes3 = plt.subplots(nrows=rows_count, ncols=cols_count, figsize=(14, 4 * rows_count))
            axes3_flat = np.atleast_1d(axes3).flatten()

            for idx, col in enumerate(forecast_cats):
                ax = axes3_flat[idx]
                fdata = data['forecast_plot_data'][col]
                ax.plot(data['original_dates'], fdata['y_actuals'], marker='s', linewidth=2, label="Historical Actual",
                        color=brand_primary_dark)
                ax.plot(data['all_dates'], fdata['y_forecast_full'], linestyle='--', linewidth=2, label="Linear Trend",
                        color=brand_secondary)
                ax.set_title(col, fontweight='bold', color=brand_primary_dark)
                ax.tick_params(axis='x', rotation=45)
                ax.grid(True, linestyle='--', alpha=0.6, color=brand_neutral)
                if idx == 0: ax.legend(loc='upper left')

            for i in range(num_cats, len(axes3_flat)):
                fig3.delaxes(axes3_flat[i])
            plt.tight_layout()
            st.pyplot(fig3)
            create_download_button(fig3, f"{data['report_name']}_ML_Forecast.png")

    # PLOT 4: Keyword Trends
    st.markdown("---")
    st.subheader("Extracted NLP Keyword Trends")
    kw_trend = data['keyword_monthly_trend']
    if not kw_trend.empty:
        fig2, axes2 = plt.subplots(nrows=4, ncols=2, figsize=(14, 12))
        axes2_flat = axes2.flatten()
        keyword_dates = kw_trend.index.astype(str).tolist()

        for idx, column in enumerate(kw_trend.columns):
            if idx >= 8: break
            ax = axes2_flat[idx]
            ax.plot(keyword_dates, kw_trend[column], marker='o', linewidth=2, color=brand_secondary)
            ax.set_title(column, fontweight='bold', color=brand_primary_dark)
            ax.grid(True, linestyle='--', alpha=0.6, color=brand_neutral)
            ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        st.pyplot(fig2)
        create_download_button(fig2, f"{data['report_name']}_Keyword_Trends.png")

    # PLOT 6: Individual Rep Trends (INTERACTIVE FOR ALL REPS)
    st.markdown("---")
    colA, colB = st.columns([1, 2])
    with colA:
        st.subheader("Individual Rep Scorecard")
        st.write("Select a Rep to view their specific trendline.")
        selected_rep = st.selectbox(f"Select Rep/Advisor ({data['report_name']})", data['all_advisors_alpha'])

    with colB:
        if selected_rep and selected_rep in data['all_rep_monthly_trend'].index:
            monthly_data = data['all_rep_monthly_trend'].loc[selected_rep]
            months_str = monthly_data.index.astype(str).tolist()
            issue_counts = monthly_data.values

            fig6 = plt.figure(figsize=(10, 4))
            gs = fig6.add_gridspec(1, 2, width_ratios=[3, 1])
            ax_line = fig6.add_subplot(gs[0])
            ax_table = fig6.add_subplot(gs[1])

            ax_line.plot(months_str, issue_counts, marker='o', linewidth=2, color=brand_primary)
            ax_line.set_title(f"Trend: {selected_rep}", fontweight='bold', color=brand_primary_dark)
            ax_line.set_ylabel("Volume", color=brand_primary)
            ax_line.tick_params(axis='x', rotation=45)
            ax_line.grid(True, linestyle='--', alpha=0.6, color=brand_neutral)

            ax_table.axis('tight')
            ax_table.axis('off')
            table_data = [[m, c] for m, c in zip(months_str, issue_counts)]
            table = ax_table.table(cellText=table_data, colLabels=["Month", "Total"], loc='center', cellLoc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 1.5)
            for (row, col), cell in table.get_celld().items():
                if row == 0:
                    cell.set_text_props(weight='bold', color='#000000')
                    cell.set_facecolor(brand_neutral)
            plt.tight_layout()
            st.pyplot(fig6)
            safe_name = "".join([c for c in selected_rep if c.isalpha() or c.isdigit()]).rstrip()
            create_download_button(fig6, f"{data['report_name']}_Trend_{safe_name}.png")


# --- APP EXECUTION FLOW ---
if file_issues and file_losses:
    # 1. Load Raw Data
    try:
        if source == "Fictional sample reports":
            df_issues_raw = pd.read_csv(file_issues)
            df_losses_raw = pd.read_csv(file_losses)
        else:
            df_issues_raw = pd.read_excel(file_issues)
            df_losses_raw = pd.read_excel(file_losses)
    except Exception as exc:
        st.error(f"Unable to load files: {exc}")
        st.stop()
    df_issues_raw.columns = df_issues_raw.columns.str.strip()
    df_losses_raw.columns = df_losses_raw.columns.str.strip()

    missing_issues = {"Date", "Advisor", "Category", "Issue", "Account"} - set(df_issues_raw.columns)
    missing_losses = {"Date", "Rep Name", "Type", "Reason", "Account"} - set(df_losses_raw.columns)
    if missing_issues or missing_losses:
        st.error(f"Missing required columns — Issues: {sorted(missing_issues)}; Losses: {sorted(missing_losses)}")
        st.stop()
    if df_issues_raw.empty or df_losses_raw.empty:
        st.error("Both reports must contain at least one row.")
        st.stop()
    date_col_losses = 'Date'
    df_issues_raw['Date'] = pd.to_datetime(df_issues_raw['Date'], errors='coerce')
    df_losses_raw[date_col_losses] = pd.to_datetime(df_losses_raw[date_col_losses], errors='coerce')

    valid_dates = pd.concat([df_issues_raw["Date"], df_losses_raw[date_col_losses]]).dropna()
    if valid_dates.empty:
        st.error("No valid dates found in the reports.")
        st.stop()

    # 2. Extract Unique Regions for Filter
    # (Matches 'Region' for Issues, 'Location' for Losses)
    region_col_issues = 'Region' if 'Region' in df_issues_raw.columns else None
    region_col_losses = 'Location' if 'Location' in df_losses_raw.columns else None

    all_regions = set()
    if region_col_issues:
        all_regions.update(df_issues_raw[region_col_issues].dropna().astype(str).unique())
    if region_col_losses:
        all_regions.update(df_losses_raw[region_col_losses].dropna().astype(str).unique())
    all_regions = sorted(list(all_regions))

    # 3. Sidebar Filters
    min_date = valid_dates.min().date()
    max_date = valid_dates.max().date()

    with st.sidebar:
        st.header("📅 2. Global Filters")
        st.write("*Note: Trend projection requires at least 3 months of data.*")
        date_range = st.date_input("Analysis Timeframe", value=(min_date, max_date), min_value=min_date,
                                   max_value=max_date)

        selected_regions = st.multiselect("Filter by Region / Location", options=all_regions, default=[])
        st.write("*Leave empty to include all regions.*")

    if len(date_range) == 2:
        start_date, end_date = date_range

        # Apply Date Filter
        df_issues = df_issues_raw[
            (df_issues_raw['Date'].dt.date >= start_date) & (df_issues_raw['Date'].dt.date <= end_date)]
        df_losses = df_losses_raw[(df_losses_raw[date_col_losses].dt.date >= start_date) & (
                    df_losses_raw[date_col_losses].dt.date <= end_date)]

        # Apply Region Filter
        if selected_regions:
            if region_col_issues:
                df_issues = df_issues[df_issues[region_col_issues].astype(str).isin(selected_regions)]
            if region_col_losses:
                df_losses = df_losses[df_losses[region_col_losses].astype(str).isin(selected_regions)]

        with st.spinner("Crunching the numbers and running ML models..."):
            # Only process if data exists after filters
            if not df_issues.empty:
                res_issues = process_data(df_issues, "Issues", 'Date', 'Advisor', 'Category', 'Issue', 'Account')
            else:
                res_issues = None

            if not df_losses.empty:
                res_losses = process_data(df_losses, "Losses", date_col_losses, 'Rep Name', 'Type', 'Reason', 'Account')
            else:
                res_losses = None

        # Create Tabs in the UI
        tab1, tab2, tab3, tab4 = st.tabs([
            "📜 Historical Summary",
            "🔮 Forecast Summary",
            "📈 Issues Dashboard",
            "📉 Losses Dashboard"
        ])

        # -------------------------------------------------------------
        # TAB 1: HISTORICAL SUMMARY
        # -------------------------------------------------------------
        with tab1:
            st.header(f"Historical Executive Summary ({start_date.strftime('%b %Y')} - {end_date.strftime('%b %Y')})")
            st.write("An analysis of what happened during the selected timeframe and regions.")

            col_h1, col_h2 = st.columns(2)

            with col_h1:
                st.markdown("### 🔹 Issues Report Context")
                if res_issues:
                    m1, m2 = st.columns(2)
                    m1.metric("Total Issues Recorded", res_issues['hist_total'])
                    m2.metric("Monthly Average Run-Rate", res_issues['hist_avg_monthly'])

                    st.info(
                        f"**📊 Volume Distribution:** The highest 3 reporting sources account for **{res_issues['hist_top_3_reps_pct']:.1f}%** of all issues during this period.")
                    st.info(
                        f"**🏷️ Primary Issue Type:** '**{res_issues['hist_top_cat']}**' was the leading driver, making up **{res_issues['hist_top_cat_pct']:.1f}%** of total volume.")
                    st.info(
                        f"**📅 Peak Volume:** The highest volume occurred in **{res_issues['hist_peak']}** with **{res_issues['hist_peak_vol']}** reports.")

                    if res_issues['top_keywords_str']:
                        st.success(f"**💬 Free-Text NLP Themes:** '{res_issues['top_keywords_str']}'")
                    if res_issues['top_outcome_str']:
                        st.success(f"**📧 Email Follow-Up Outcomes:** '{res_issues['top_outcome_str']}'")
                else:
                    st.warning("No Issues data available for the selected filters.")

            with col_h2:
                st.markdown("### 🔹 Losses Report Context")
                if res_losses:
                    m3, m4 = st.columns(2)
                    m3.metric("Total Losses Recorded", res_losses['hist_total'])
                    m4.metric("Monthly Average Run-Rate", res_losses['hist_avg_monthly'])

                    st.warning(
                        f"**📊 Volume Distribution:** The highest 3 reporting sources account for **{res_losses['hist_top_3_reps_pct']:.1f}%** of all losses during this period.")
                    st.warning(
                        f"**🏷️ Primary Loss Type:** '**{res_losses['hist_top_cat']}**' was the leading driver, making up **{res_losses['hist_top_cat_pct']:.1f}%** of total volume.")
                    st.warning(
                        f"**📅 Peak Volume:** The highest volume occurred in **{res_losses['hist_peak']}** with **{res_losses['hist_peak_vol']}** reports.")

                    if res_losses['top_keywords_str']:
                        st.success(f"**💬 Free-Text NLP Themes:** '{res_losses['top_keywords_str']}'")
                else:
                    st.warning("No Losses data available for the selected filters.")

        # -------------------------------------------------------------
        # TAB 2: FORECAST SUMMARY
        # -------------------------------------------------------------
        with tab2:
            st.header("Illustrative Trend Extrapolations")
            if (res_issues and res_issues['total_months'] < 3) or (res_losses and res_losses['total_months'] < 3):
                st.warning("⚠️ Please select a date range of at least 3 months to generate trend extrapolations.")
            else:
                st.write(
                    "A linear regression extends observed monthly volume trends for three months. These are illustrations, not validated forecasts.")

                col_f1, col_f2 = st.columns(2)

                with col_f1:
                    st.markdown("### 🔹 Issues Trend Observations")
                    if res_issues:
                        if not res_issues['insights']: st.write("No significant trends detected.")
                        for item in res_issues['insights'][:3]:
                            if "CRITICAL" in item['status']:
                                st.error(
                                    f"**{item['category']}**\n\n**Impact:** Expect ~{item['projected_vol']} issues next 90 days ({'+' if item['net_change'] > 0 else ''}{item['net_change']} vs last period).\n\n**Observation:** {item['observation']}")
                            elif "WATCHLIST" in item['status']:
                                st.warning(
                                    f"**{item['category']}**\n\n**Impact:** Expect ~{item['projected_vol']} issues next 90 days ({'+' if item['net_change'] > 0 else ''}{item['net_change']} vs last period).\n\n**Observation:** {item['observation']}")
                            elif "IMPROVING" in item['status']:
                                st.success(
                                    f"**{item['category']}**\n\n**Impact:** Volume dropping to ~{item['projected_vol']} issues next 90 days ({item['net_change']} vs last period).\n\n**Observation:** {item['observation']}")

                with col_f2:
                    st.markdown("### 🔹 Losses Trend Observations")
                    if res_losses:
                        if not res_losses['insights']: st.write("No significant trends detected.")
                        for item in res_losses['insights'][:3]:
                            if "CRITICAL" in item['status']:
                                st.error(
                                    f"**{item['category']}**\n\n**Impact:** Expect ~{item['projected_vol']} issues next 90 days ({'+' if item['net_change'] > 0 else ''}{item['net_change']} vs last period).\n\n**Observation:** {item['observation']}")
                            elif "WATCHLIST" in item['status']:
                                st.warning(
                                    f"**{item['category']}**\n\n**Impact:** Expect ~{item['projected_vol']} issues next 90 days ({'+' if item['net_change'] > 0 else ''}{item['net_change']} vs last period).\n\n**Observation:** {item['observation']}")
                            elif "IMPROVING" in item['status']:
                                st.success(
                                    f"**{item['category']}**\n\n**Impact:** Volume dropping to ~{item['projected_vol']} issues next 90 days ({item['net_change']} vs last period).\n\n**Observation:** {item['observation']}")

        # -------------------------------------------------------------
        # TAB 3 & 4: VISUAL DASHBOARDS
        # -------------------------------------------------------------
        with tab3:
            if res_issues:
                render_dashboard(res_issues)
            else:
                st.info("No Issues data to display.")

        with tab4:
            if res_losses:
                render_dashboard(res_losses)
            else:
                st.info("No Losses data to display.")

    else:
        st.sidebar.warning("Please select a complete Date Range.")
else:
    st.info("👈 Select the demo or upload both Excel reports to begin.")