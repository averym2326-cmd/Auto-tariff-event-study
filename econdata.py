from pathlib import Path
DATA_DIR = Path(__file__).parent

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
import matplotlib.pyplot as plt
from market_data import download_raw

START, END = "2024-10-01", "2025-08-31"
TICKERS = {"hyundai": "005380.KS", "kia": "000270.KS", "gm": "GM",
           "ford": "F", "kospi": "^KS11", "sp": "^GSPC"}
download_raw(DATA_DIR, TICKERS, START, END)

hyundai = pd.read_csv(DATA_DIR / "hyundai.csv", parse_dates=["Date"])
hyundai = hyundai[["Date", "Close"]].rename(columns={"Close": "hyundai"})

kia = pd.read_csv(DATA_DIR / "kia.csv", parse_dates=["Date"])
kia = kia[["Date", "Close"]].rename(columns={"Close": "kia"})

gm = pd.read_csv(DATA_DIR / "gm.csv", parse_dates=["Date"])
gm = gm[["Date", "Close"]].rename(columns={"Close": "gm"})

ford = pd.read_csv(DATA_DIR / "ford.csv", parse_dates=["Date"])
ford = ford[["Date", "Close"]].rename(columns={"Close": "ford"})

sp = pd.read_csv(DATA_DIR / "sp.csv", parse_dates=["Date"])
sp = sp[["Date", "Close"]].rename(columns={"Close": "sp"})

kospi = pd.read_csv(DATA_DIR / "kospi.csv", parse_dates=["Date"])
kospi = kospi[["Date", "Close"]].rename(columns={"Close": "kospi"})

fx = pd.read_csv(DATA_DIR / "fx.csv", parse_dates=["observation_date"])
fx = fx[["observation_date", "DEXKOUS"]].rename(columns={"observation_date": "Date", "DEXKOUS": "krw_usd"})
#{"oldname": "newname"} renaming the column

price_cols = ["hyundai", "kia", "gm", "ford", "sp", "kospi", "krw_usd"]
# Use each series' previous observed session, before other markets add holiday rows.
for frame in [hyundai, kia, gm, ford, sp, kospi, fx]:
    frame.sort_values("Date", inplace=True)
    col = frame.columns[1]
    observed = frame[col].dropna()
    frame[col + "_ret"] = np.log(observed / observed.shift(1))

m1 = hyundai.merge(kia, on="Date", how="outer")
m2 = m1.merge(gm, on="Date", how="outer")
m3 = m2.merge(ford, on="Date", how="outer")
m4 = m3.merge(sp, on="Date", how="outer")
m5 = m4.merge(kospi, on="Date", how="outer")
m6 = m5.merge(fx, on="Date", how="outer")
m6 = m6.sort_values("Date").reset_index(drop=True)

print(m6.head())
m6.to_csv(DATA_DIR / "merged_data.csv", index=False, columns=["Date", *price_cols])

print(m6.head())

est = m6[(m6["Date"] >= "2024-10-01") & (m6["Date"] <= "2025-02-28")]

event_idx = m6.index[m6["Date"] == "2025-04-03"][0]
event_window = m6.iloc[event_idx - 2 : event_idx + 3]   # 2 trading days before through 2 after

print(est.shape)

event_idx = m6.index[m6["Date"] == "2025-04-03"][0]
event_window = m6.iloc[event_idx - 3 : event_idx + 4]          

print(event_window.shape)
print(event_window["Date"])

import statsmodels.api as sm

X = sm.add_constant(est["kospi_ret"])
y = est["hyundai_ret"]
model = sm.OLS(y, X, missing="drop").fit()
print(model.summary())

X = sm.add_constant(est["kospi_ret"])
y = est["kia_ret"]
model = sm.OLS(y, X, missing="drop").fit()
print(model.summary())

X = sm.add_constant(est["sp_ret"])
y = est["ford_ret"]
model = sm.OLS(y, X, missing="drop").fit()
print(model.summary())

X = sm.add_constant(est["sp_ret"])
y = est["gm_ret"]
model = sm.OLS(y, X, missing="drop").fit()
print(model.summary())

hyundai_model = sm.OLS(est["hyundai_ret"], sm.add_constant(est["kospi_ret"]), missing="drop").fit()
alpha = hyundai_model.params["const"]
beta = hyundai_model.params["kospi_ret"]

event_window["hyundai_expected"] = alpha + beta * event_window["kospi_ret"]
event_window["hyundai_abnormal"] = event_window["hyundai_ret"] - event_window["hyundai_expected"]

car_hyundai = event_window["hyundai_abnormal"].sum()
print("Hyundai CAR:", car_hyundai)

kia_model = sm.OLS(est["kia_ret"], sm.add_constant(est["kospi_ret"]), missing="drop").fit()
alpha = kia_model.params["const"]
beta = kia_model.params["kospi_ret"]

event_window["kia_expected"] = alpha + beta * event_window["kospi_ret"]
event_window["kia_abnormal"] = event_window["kia_ret"] - event_window["kia_expected"]

car_kia = event_window["kia_abnormal"].sum()
print("Kia CAR:", car_kia)

ford_model = sm.OLS(est["ford_ret"], sm.add_constant(est["sp_ret"]), missing="drop").fit()
alpha = ford_model.params["const"]
beta = ford_model.params["sp_ret"]

event_window["ford_expected"] = alpha + beta * event_window["sp_ret"]
event_window["ford_abnormal"] = event_window["ford_ret"] - event_window["ford_expected"]

car_ford = event_window["ford_abnormal"].sum()
print("Ford CAR:", car_ford)

gm_model = sm.OLS(est["gm_ret"], sm.add_constant(est["sp_ret"]), missing="drop").fit()
alpha = gm_model.params["const"]
beta = gm_model.params["sp_ret"]

event_window["gm_expected"] = alpha + beta * event_window["sp_ret"]
event_window["gm_abnormal"] = event_window["gm_ret"] - event_window["gm_expected"]

car_gm = event_window["gm_abnormal"].sum()
print("GM CAR:", car_gm)

resid_std_hyundai = hyundai_model.resid.std()
n_days = len(event_window)
car_se = resid_std_hyundai * (n_days ** 0.5)
t_stat_hyundai = car_hyundai / car_se
p_value_hyundai = 2 * (1 - stats.t.cdf(abs(t_stat_hyundai), hyundai_model.df_resid))

resid_std_kia = kia_model.resid.std()
n_days = len(event_window)
car_se = resid_std_kia * (n_days ** 0.5)
t_stat_kia = car_kia / car_se
p_value_kia = 2 * (1 - stats.t.cdf(abs(t_stat_kia), kia_model.df_resid))

resid_std_ford = ford_model.resid.std()
n_days = len(event_window)
car_se = resid_std_ford * (n_days ** 0.5)
t_stat_ford = car_ford / car_se
p_value_ford = 2 * (1 - stats.t.cdf(abs(t_stat_ford), ford_model.df_resid))

resid_std_gm = gm_model.resid.std()
n_days = len(event_window)
car_se = resid_std_gm * (n_days ** 0.5)
t_stat_gm = car_gm / car_se
p_value_gm = 2 * (1 - stats.t.cdf(abs(t_stat_gm), gm_model.df_resid))

print("Hyundai t-stat:", t_stat_hyundai, " p-value:", p_value_hyundai)
print("Kia t-stat:", t_stat_kia, " p-value:", p_value_kia)
print("Ford t-stat:", t_stat_ford, " p-value:", p_value_ford)
print("GM t-stat:", t_stat_gm, " p-value:", p_value_gm)

korea_car_avg = (car_hyundai + car_kia) / 2
us_car_avg = (car_ford + car_gm) / 2
did = korea_car_avg - us_car_avg

print("Korea avg CAR:", korea_car_avg)
print("US avg CAR:", us_car_avg)
print("DiD (Korea - US):", did) 

placebo_date = "2025-01-15"
placebo_idx = m6.index[m6["Date"] == placebo_date][0]
placebo_window = m6.iloc[placebo_idx - 3 : placebo_idx + 4]

placebo_window["hyundai_expected"] = hyundai_model.params["const"] + hyundai_model.params["kospi_ret"] * placebo_window["kospi_ret"]
placebo_window["hyundai_abnormal"] = placebo_window["hyundai_ret"] - placebo_window["hyundai_expected"]
placebo_car_hyundai = placebo_window["hyundai_abnormal"].sum()

print("Placebo Hyundai CAR:", placebo_car_hyundai)

placebo_date = "2025-01-15"
placebo_idx = m6.index[m6["Date"] == placebo_date][0]
placebo_window = m6.iloc[placebo_idx - 3 : placebo_idx + 4]

placebo_window["kia_expected"] = kia_model.params["const"] + kia_model.params["kospi_ret"] * placebo_window["kospi_ret"]
placebo_window["kia_abnormal"] = placebo_window["kia_ret"] - placebo_window["kia_expected"]
placebo_car_kia = placebo_window["kia_abnormal"].sum()

print("Placebo Kia CAR:", placebo_car_kia)

placebo_date = "2025-01-15"
placebo_idx = m6.index[m6["Date"] == placebo_date][0]
placebo_window = m6.iloc[placebo_idx - 3 : placebo_idx + 4]

placebo_window["gm_expected"] = gm_model.params["const"] + gm_model.params["sp_ret"] * placebo_window["sp_ret"]
placebo_window["gm_abnormal"] = placebo_window["gm_ret"] - placebo_window["gm_expected"]
placebo_car_gm = placebo_window["gm_abnormal"].sum()

print("Placebo GM CAR:", placebo_car_gm)

placebo_date = "2025-01-15"
placebo_idx = m6.index[m6["Date"] == placebo_date][0]
placebo_window = m6.iloc[placebo_idx - 3 : placebo_idx + 4]

placebo_window["ford_expected"] = ford_model.params["const"] + ford_model.params["sp_ret"] * placebo_window["sp_ret"]
placebo_window["ford_abnormal"] = placebo_window["ford_ret"] - placebo_window["ford_expected"]
placebo_car_ford = placebo_window["ford_abnormal"].sum()

print("Placebo Ford CAR:", placebo_car_ford)

def compute_car(model, market_col, asset, center_date, window=3):
    idx = m6.index[m6["Date"] == center_date][0]
    window_df = m6.iloc[idx - window : idx + window + 1]
    alpha = model.params["const"]
    beta = model.params[market_col]
    expected = alpha + beta * window_df[market_col]
    abnormal = window_df[asset + "_ret"] - expected
    return abnormal.sum()

for date in ["2024-10-21", "2024-11-19", "2025-01-08"]:
    car_h = compute_car(hyundai_model, "kospi_ret", "hyundai", date)
    car_k = compute_car(kia_model, "kospi_ret", "kia", date)
    car_g = compute_car(gm_model, "sp_ret", "gm", date)
    car_f = compute_car(ford_model, "sp_ret", "ford", date)
    korea_avg = (car_h + car_k) / 2
    us_avg = (car_g + car_f) / 2
    print(date, "Korea avg:", korea_avg, "US avg:", us_avg, "DiD:", korea_avg - us_avg)

    idx = m6.index[m6["Date"] == "2024-10-21"][0]
window_df = m6.iloc[idx - 3 : idx + 4]
print(window_df[["Date", "hyundai_ret", "kia_ret", "gm_ret", "ford_ret", "kospi_ret", "sp_ret"]])
