import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
import plotly.graph_objects as go
import scipy.optimize as optimize

# --- 1. QUANTITATIVE RISK ENGINES ---

class MarketDataPipeline:
    def __init__(self, ticker: str, lookback_years: int = 5):
        self.ticker = ticker
        self.lookback_years = lookback_years
        
    def fetch_market_context(self):
        """Fetches historical stock prices and flattens modern MultiIndex columns safely."""
        end_date = pd.Timestamp.now()
        start_date = end_date - pd.DateOffset(years=self.lookback_years)
        
        # Download data with auto_adjust=False to force 'Adj Close' presence
        data = yf.download(self.ticker, start=start_date, end=end_date, progress=False, auto_adjust=False)
        if data.empty:
            raise ValueError(f"No market data returned for ticker: {self.ticker}")
            
        # --- FIXED BLOCK: Flatten MultiIndex columns if present ---
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
            
        adj_close = data['Adj Close'].squeeze()
        log_returns = np.log(adj_close / adj_close.shift(1)).dropna()
        
        current_spot = float(adj_close.iloc[-1])
        realized_vol = float(log_returns.std() * np.sqrt(252))
        
        return current_spot, log_returns.values, realized_vol

class PortfolioStressTester:
    def __init__(self, S0, K, T, r, sigma, stock_shares, option_contracts):
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.stock_shares = stock_shares
        self.option_contracts = option_contracts
        self.contract_size = 100

    def calculate_greeks(self):
        """Calculates Black-Scholes Delta (Δ) and Gamma (Γ) for a European Call."""
        t = max(1e-5, self.T)
        d1 = (np.log(self.S0 / self.K) + (self.r + 0.5 * self.sigma**2) * t) / (self.sigma * np.sqrt(t))
        delta = norm.cdf(d1)
        gamma = norm.pdf(d1) / (self.S0 * self.sigma * np.sqrt(t))
        return delta, gamma

    def calculate_historical_var_pnl(self, historical_returns):
        """Calculates historical PnL distribution via Delta-Gamma Taylor approximation."""
        delta, gamma = self.calculate_greeks()
        stock_shifts = self.S0 * historical_returns
        stock_pnl = self.stock_shares * stock_shifts
        total_options_scaled = self.option_contracts * self.contract_size
        option_pnl = total_options_scaled * (delta * stock_shifts + 0.5 * gamma * (stock_shifts**2))
        return stock_pnl + option_pnl

    def execute_deterministic_shock(self, spot_shock_pct: float, vol_shock_abs: float):
        """Performs exact Black-Scholes full revaluation under severe macro shocks."""
        t = max(1e-5, self.T)
        d1_base = (np.log(self.S0 / self.K) + (self.r + 0.5 * self.sigma**2) * t) / (self.sigma * np.sqrt(t))
        base_option_price = (self.S0 * norm.cdf(d1_base) - self.K * np.exp(-self.r * t) * norm.cdf(d1_base - self.sigma * np.sqrt(t)))
        base_portfolio_value = (self.stock_shares * self.S0) + (self.option_contracts * self.contract_size * base_option_price)
        
        shocked_spot = self.S0 * (1 + spot_shock_pct)
        shocked_vol = max(0.01, self.sigma + vol_shock_abs)
        
        if shocked_spot <= 0:
            shocked_option_price = 0.0
        else:
            d1_shock = (np.log(shocked_spot / self.K) + (self.r + 0.5 * shocked_vol**2) * t) / (self.sigma * np.sqrt(t))
            shocked_option_price = (shocked_spot * norm.cdf(d1_shock) - self.K * np.exp(-self.r * t) * norm.cdf(d1_shock - shocked_vol * np.sqrt(t)))
            
        shocked_portfolio_value = (self.stock_shares * shocked_spot) + (self.option_contracts * self.contract_size * shocked_option_price)
        return shocked_portfolio_value - base_portfolio_value

class FixedIncomeAsset:
    def __init__(self, face_value, coupon_rate, maturity_years, frequency=2):
        self.face_value = face_value
        self.coupon_rate = coupon_rate
        self.maturity = maturity_years
        self.frequency = frequency
        
    def _get_cash_flows(self):
        periods = int(self.maturity * self.frequency)
        times = np.array([i / self.frequency for i in range(1, periods + 1)])
        coupon_payment = (self.coupon_rate * self.face_value) / self.frequency
        cash_flows = np.full(periods, coupon_payment)
        cash_flows[-1] += self.face_value
        return times, cash_flows

    def calculate_ytm(self, current_price):
        times, cash_flows = self._get_cash_flows()
        p_v_func = lambda y: np.sum(cash_flows / (1 + y / self.frequency) ** (self.frequency * times)) - current_price
        try:
            return optimize.newton(p_v_func, self.coupon_rate)
        except RuntimeError:
            return self.coupon_rate

    def calculate_sensitivities(self, ytm):
        times, cash_flows = self._get_cash_flows()
        discount_factors = 1 / (1 + ytm / self.frequency) ** (self.frequency * times)
        pv_cash_flows = cash_flows * discount_factors
        bond_price = np.sum(pv_cash_flows)
        macaulay_duration = np.sum(times * pv_cash_flows) / bond_price
        modified_duration = macaulay_duration / (1 + ytm / self.frequency)
        convexity = np.sum(times * (times + 1 / self.frequency) * pv_cash_flows) / (bond_price * (1 + ytm / self.frequency)**2)
        return modified_duration, convexity

    def estimate_interest_rate_pnl(self, current_price, ytm_shift_bps):
        ytm = self.calculate_ytm(current_price)
        d_mod, convexity = self.calculate_sensitivities(ytm)
        dy = ytm_shift_bps / 10000
        pct_price_change = (-d_mod * dy) + (0.5 * convexity * (dy**2))
        return current_price * pct_price_change

class MultiAssetVaRAggregator:
    def __init__(self, stock_returns, yield_returns):
        df = pd.DataFrame({'Stock': stock_returns, 'Yield': yield_returns}).dropna()
        self.cov_matrix = df.cov().values
        
    def calculate_portfolio_var(self, stock_shares, spot_price, option_contracts, option_delta, bond_price, bond_d_mod, confidence_level=0.99):
        """Cross-Asset Parametric VaR combining equity delta-equivalents and bond dollar durations."""
        stock_dollar_exposure = stock_shares * spot_price
        option_dollar_exposure = option_contracts * 100 * spot_price * option_delta
        total_equity_sensitivity = stock_dollar_exposure + option_dollar_exposure
        total_bond_sensitivity = bond_price * bond_d_mod
        
        w = np.array([total_equity_sensitivity, total_bond_sensitivity])
        portfolio_variance = np.dot(w.T, np.dot(self.cov_matrix, w))
        portfolio_volatility = np.sqrt(portfolio_variance)
        z_score = norm.ppf(confidence_level)
        
        # Stand-alone risks
        stock_standalone_var = abs(total_equity_sensitivity) * np.sqrt(self.cov_matrix[0][0]) * z_score
        bond_standalone_var = abs(total_bond_sensitivity) * np.sqrt(self.cov_matrix[1][1]) * z_score
        
        undiversified_var = stock_standalone_var + bond_standalone_var
        diversified_var = portfolio_volatility * z_score
        diversification_benefit = undiversified_var - diversified_var
        
        return diversified_var, undiversified_var, diversification_benefit

# --- 2. STREAMLIT INTERFACE AND CONTROL LAYOUT ---

st.set_page_config(page_title="Multi-Asset Risk Matrix", layout="wide")
st.title("📊 Multi-Asset Cross-Asset Parametric & Historical Risk Engine")
st.markdown("A unified quantitative environment assessing non-linear derivatives risk and structural fixed income duration constraints.")

# Sidebar Control Deck
st.sidebar.header("🕹️ Global Portfolio Parameters")
ticker = st.sidebar.text_input("Equity Underlying Ticker", value="AAPL")
lookback = st.sidebar.slider("Historical Matrix Window (Years)", 1, 5, 5)
conf_level = st.sidebar.selectbox("VaR Model Confidence Level", [0.95, 0.99], index=1)

st.sidebar.subheader("Equity & Options Positions")
shares = st.sidebar.number_input("Equity Shares Held", value=10000, step=500)
contracts = st.sidebar.number_input("Option Contracts (Short = Negative)", value=-100, step=10)
strike_offset = st.sidebar.slider("Option Strike Offset (% of Spot)", -20, 20, 0)
days_to_expiry = st.sidebar.slider("Days to Option Expiration", 10, 365, 180)

st.sidebar.subheader("Fixed-Income Allocation")
bond_face = st.sidebar.number_input("Bond Face Value ($)", value=1000000, step=100000)
bond_coupon = st.sidebar.slider("Coupon Rate (%)", 0.0, 15.0, 5.0) / 100
bond_maturity = st.sidebar.slider("Bond Maturity (Years)", 1, 30, 10)
bond_market_price = st.sidebar.number_input("Current Bond Clean Price ($)", value=950000, step=10000)

# Run Live Ingestion Pipeline
try:
    pipeline = MarketDataPipeline(ticker=ticker, lookback_years=lookback)
    spot, historical_returns, empirical_vol = pipeline.fetch_market_context()
    
    # Ingest 10-Year Treasury proxy for Cross-Asset Matrix mapping
    yield_data = yf.download("^TNX", start=pd.Timestamp.now() - pd.DateOffset(years=lookback), progress=False, auto_adjust=False)
    
    # --- FIXED BLOCK: Flatten MultiIndex columns for fixed-income asset data ---
    if isinstance(yield_data.columns, pd.MultiIndex):
        yield_data.columns = yield_data.columns.get_level_values(0)
        
    yield_raw = yield_data['Adj Close'].squeeze()
    yield_returns = (yield_raw / 100).diff().dropna().values
except Exception as e:
    st.error(f"Data Pipeline Interruption: {e}")
    st.stop()

# Initialize Analytical Instantiations
strike_price = spot * (1 + strike_offset / 100)
time_to_expiry = days_to_expiry / 365
risk_free_rate = float(yield_raw.iloc[-1]) / 100

equity_engine = PortfolioStressTester(spot, strike_price, time_to_expiry, risk_free_rate, empirical_vol, shares, contracts)
delta, gamma = equity_engine.calculate_greeks()

portfolio_pnl_distribution = equity_engine.calculate_historical_var_pnl(historical_returns)
bond_asset = FixedIncomeAsset(bond_face, bond_coupon, bond_maturity)
bond_ytm = bond_asset.calculate_ytm(bond_market_price)
bond_duration, bond_convexity = bond_asset.calculate_sensitivities(bond_ytm)

min_len = min(len(historical_returns), len(yield_returns))
aggregator = MultiAssetVaRAggregator(historical_returns[-min_len:], yield_returns[-min_len:])
div_var, undiv_var, div_benefit = aggregator.calculate_portfolio_var(shares, spot, contracts, delta, bond_market_price, bond_duration, conf_level)

st.markdown("### Executive Risk Allocation Diagnostics")
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Live Underlying Spot", f"${spot:,.2f}")
kpi2.metric("Option Delta (Δ Exposure)", f"{delta:.4f}")
kpi3.metric("Bond Modified Duration", f"{bond_duration:.4f}")
kpi4.metric(f"Total Diversified {int(conf_level*100)}% VaR", f"${div_var:,.2f}")
st.markdown("---")

left_col, right_col = st.columns(2)

with left_col:
    st.subheader("🌐 Cross-Asset Covariance Breakdown")
    st.markdown("Accounts for non-linear option delta equivalents alongside fixed income duration limits.")
    
    # Render Parametric VaR Card Layouts
    st.write(f"**Undiversified Standalone Risk Sum:** `${undiv_var:,.2f}`")
    st.write(f"**Diversification Capital Relief:** `${div_benefit:,.2f}`")
    st.success(f"**Net Risk Reduction Profile:** {((div_benefit/max(1, undiv_var))*100):.2f}% portfolio correlation diversification benefit.")

    # # Plotly Equity Tail-Risk Return Distribution Chart
    st.subheader("📈 Simulated Equity Cluster Tails")
    
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=portfolio_pnl_distribution, 
        nbinsx=100, 
        name="Simulated PnL", 
        marker_color='#1f77b4', 
        opacity=0.75
    ))
    
    hist_var_threshold = np.percentile(portfolio_pnl_distribution, (1 - conf_level) * 100)
    
    fig.add_vline(
        x=hist_var_threshold, 
        line_width=3, 
        line_dash="dash", 
        line_color="red", 
        annotation_text=" Historical VaR Cutoff"
    )
    
    fig.update_layout(
        xaxis_title="Simulated Dollar PnL ($)", 
        yaxis_title="Frequency", 
        margin=dict(l=20, r=20, t=20, b=20), 
        height=300
    )
    
    st.plotly_chart(fig, use_container_width=True)

with right_col:
    st.subheader("⚡ Macro Structural Stress Testing Framework")
    st.markdown("Subject the mixed book to absolute full valuation repricing dislocations.")
    
    scenario = st.selectbox(
        "Select Core Systemic Scenario Profile", 
        [
            "Manual Risk Adjustments",
            "2008 Systemic Meltdown (-30% Spot, +25% Vol, -150bps Yield)",
            "Inflationary Rate Squeeze (-15% Spot, +10% Vol, +200bps Yield)"
        ]
    )

    if "2008 Systemic Meltdown" in scenario:
        spot_shock, vol_shock, yield_shock = -0.30, 0.25, -150
    elif "Inflationary Rate Squeeze" in scenario:
        spot_shock, vol_shock, yield_shock = -0.15, 0.10, 200
    else:
        spot_shock = st.slider("Underlying Spot Shock (%)", -50, 50, -10) / 100
        vol_shock = st.slider("Absolute Vol Shock (+/- Vol)", -20, 50, 10) / 100
        yield_shock = st.slider("Yield Curve Parallel Shift (bps)", -300, 300, 0, step=10)

    # Compute Stressed PnL across components
    equity_stress_pnl = equity_engine.execute_deterministic_shock(spot_shock, vol_shock)
    bond_stress_pnl = bond_asset.estimate_interest_rate_pnl(bond_market_price, yield_shock)
    total_stress_pnl = equity_stress_pnl + bond_stress_pnl
    
    st.markdown("#### Scenario Vulnerability Reconciliation")
    st.write(f"Equity Shock Impact: `${equity_stress_pnl:,.2f}`")
    st.write(f"Fixed Income Rate Impact: `${bond_stress_pnl:,.2f}`")
    
    if total_stress_pnl < 0:
        st.error(f"**Total Consolidated Stressed Portfolio PnL:** -${abs(total_stress_pnl):,.2f}")
    else:
        st.success(f"**Total Consolidated Stressed Portfolio PnL:** +${total_stress_pnl:,.2f}")
