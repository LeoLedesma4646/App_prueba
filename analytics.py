"""
Núcleo analítico compartido de la app consolidada (Grupo 5).

Reúne en un solo lugar los cálculos de las cuatro dimensiones para que todas
las páginas (y la réplica del Power BI) usen exactamente los mismos números:
  - Forecast  (Leonardo): forecast_core.py + artefactos del notebook.
  - Elasticidad (Cinzia): regresión log-log OLS con controles, VIF y Lasso.
  - Inventarios (Sofía): SS, ROP e inventario máximo S.
  - Integración BI (Valeria): tablas equivalentes a las del archivo PBIX.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import streamlit as st
from scipy.stats import norm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

import forecast_core as fc

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "weekly_df_final_for_modeling.csv"
ART = ROOT / "artifacts"
MODELS = ROOT / "models"

FORECAST_HORIZON = 12  # mismo horizonte que el script R de Power BI

MODEL_COLORS = {
    "Realidad": "#5B6573",
    fc.MODEL_NAIVE: "#E66101",
    fc.MODEL_RIDGE: "#7B3FA0",
    fc.MODEL_SARIMAX: "#1F78B4",
    fc.MODEL_DHR: "#33A02C",
}
STATUS_COLORS = {
    "Agotamiento crítico": "#D9534F",
    "Reordenar urgente": "#F0AD4E",
    "Óptimo": "#5CB85C",
    "Sobre-stock": "#0275D8",
}
STATUS_ORDER = list(STATUS_COLORS)

NUMERIC_VARS = [
    "units_sold", "price_unit", "promotion_flag", "stock_available",
    "delivery_days", "is_holiday_peak", "is_holiday_week", "is_summer",
    "is_winter", "lag_1", "lag_2", "rolling_mean_4", "rolling_std_4",
    "momentum", "target_next_week",
]
VAR_LABELS = {
    "units_sold": "Unidades vendidas",
    "price_unit": "Precio unitario",
    "promotion_flag": "Promoción (0/1)",
    "stock_available": "Stock disponible",
    "delivery_days": "Días de entrega",
    "is_holiday_peak": "Pico festivo (0/1)",
    "is_holiday_week": "Semana festiva (0/1)",
    "is_summer": "Verano (0/1)",
    "is_winter": "Invierno (0/1)",
    "lag_1": "Rezago 1 semana",
    "lag_2": "Rezago 2 semanas",
    "rolling_mean_4": "Media móvil 4 sem.",
    "rolling_std_4": "Desv. móvil 4 sem.",
    "momentum": "Momentum",
    "target_next_week": "Demanda semana siguiente",
}


# ============================================================
# DATOS
# ============================================================
@st.cache_data(show_spinner="Cargando la base de datos…")
def get_raw() -> pd.DataFrame:
    df = fc.load_data(DATA_PATH)
    df["sku"] = df["sku"].astype(str)
    for c in NUMERIC_VARS + ["week_number", "month", "year", "sku_age"]:
        if c in df.columns:
            if df[c].dtype == bool:
                df[c] = df[c].astype(int)
            elif df[c].dtype == object:
                df[c] = df[c].astype(str).str.strip().str.lower().map(
                    lambda v: 1 if v == "true" else 0 if v == "false" else v)
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def get_model_df() -> pd.DataFrame:
    """Base para elasticidad: solo filas con precio y unidades positivas."""
    df = get_raw()
    df = df.dropna(subset=["week", "sku", "price_unit", "units_sold"])
    return df[(df["price_unit"] > 0) & (df["units_sold"] > 0)].copy()


@st.cache_data(show_spinner=False)
def get_summary() -> pd.DataFrame:
    return fc.sku_summary(get_raw())


@st.cache_data(show_spinner=False)
def priority_skus() -> list[str]:
    """10 SKU admisibles (≥104 semanas comunes) de mayor ingreso histórico."""
    return fc.select_top_skus(get_summary(), fc.TOP_N)["sku"].tolist()


@st.cache_data(show_spinner=False)
def get_series(sku: str) -> pd.DataFrame:
    return fc.aggregate_sku(get_raw(), sku)


@st.cache_resource(show_spinner="Reconstruyendo modelos guardados…")
def get_models(sku: str) -> dict:
    return fc.load_light_models(sku, MODELS, get_series(sku))


@st.cache_data(show_spinner=False)
def get_artifacts() -> dict:
    out = {}
    for name in ["model_metrics", "backtest_predictions", "model_registry",
                 "selected_parameters", "ridge_coefficients", "sku_selection"]:
        p = ART / f"{name}.csv"
        out[name] = pd.read_csv(p, parse_dates=["week"] if name == "backtest_predictions" else None)
    return out


def winner_info(sku: str) -> dict:
    reg = get_artifacts()["model_registry"]
    row = reg[reg["sku"].eq(sku)]
    if row.empty:
        return {"model": fc.MODEL_RIDGE, "WAPE": np.nan, "MAE": np.nan}
    r = row.iloc[0]
    return {"model": r["winner"], "WAPE": float(r["winner_WAPE"]), "MAE": float(r["winner_MAE"])}


def oos_rmse(sku: str, model: str) -> float:
    """Desviación del error fuera de muestra (RMSE del backtest de 12 semanas)."""
    p = get_artifacts()["backtest_predictions"]
    p = p[p["sku"].eq(sku) & p["model"].eq(model)]
    return float(np.sqrt(np.mean(p["residual"] ** 2))) if not p.empty else np.nan


# ============================================================
# FORECAST
# ============================================================
@st.cache_data(show_spinner="Calculando pronósticos…")
def forecast_long(horizon: int = FORECAST_HORIZON, price_change_pct: float = 0.0,
                  promo_share: float | None = None, holiday: bool = False) -> pd.DataFrame:
    """Forecast de los 4 modelos para los 10 SKU (equivalente a BI_Forecast)."""
    raw = get_raw()
    rows = []
    for sku in priority_skus():
        series = get_series(sku)
        models = get_models(sku)
        fut = fc.future_scenario(series, horizon, price_change_pct=price_change_pct,
                                 promotion_share=promo_share, holiday=holiday, source_df=raw)
        win = winner_info(sku)["model"]
        for m in fc.MODEL_NAMES:
            pred = fc.forecast_from_saved(series, fut, m, models)
            rows.append(pd.DataFrame({
                "sku": sku, "week": fut["week"], "model": m, "value": pred,
                "is_winner": m == win,
            }))
    return pd.concat(rows, ignore_index=True)


def winner_forecast(horizon: int = FORECAST_HORIZON, **kw) -> pd.DataFrame:
    f = forecast_long(horizon, **kw)
    return f[f["is_winner"]].copy()


# ============================================================
# ELASTICIDAD (misma especificación que app_elasticidad_vFINAL)
# ============================================================
CONTROLS = ["promotion_flag", "stock_available", "delivery_days", "is_holiday_peak",
            "is_holiday_week", "is_summer", "is_winter"]


def weighted_price(group: pd.DataFrame) -> float:
    q = group["units_sold"].sum()
    if q <= 0:
        return group["price_unit"].mean()
    return float(np.average(group["price_unit"], weights=group["units_sold"]))


@st.cache_data(show_spinner=False)
def weekly_history(sku: str) -> pd.DataFrame:
    x = get_model_df()
    x = x[x["sku"] == sku]
    hist = (
        x.groupby("week")
        .apply(lambda g: pd.Series({
            "units_sold": g["units_sold"].sum(),
            "price_unit": weighted_price(g),
            "promotion_share": g["promotion_flag"].mean(),
            "stock_available": g["stock_available"].mean(),
        }), include_groups=False)
        .reset_index()
        .sort_values("week")
    )
    hist["year"] = hist["week"].dt.year
    hist["month"] = hist["week"].dt.month
    return hist


def fit_loglog(x: pd.DataFrame):
    x = x[(x["price_unit"] > 0) & (x["units_sold"] > 0)].dropna(subset=["price_unit", "units_sold"])
    if len(x) < 8:
        return None
    price_range = x["price_unit"].max() - x["price_unit"].min()
    if price_range < 1e-6 * max(abs(x["price_unit"].mean()), 1):
        return None
    X = sm.add_constant(np.log(x["price_unit"]), has_constant="add")
    try:
        return sm.OLS(np.log(x["units_sold"]), X).fit()
    except Exception:
        return None


def annual_elasticities(hist: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year, g in hist.groupby("year"):
        m = fit_loglog(g)
        if m is None or "price_unit" not in m.params.index:
            rows.append({"year": int(year), "elasticity": np.nan, "pvalue": np.nan, "observations": len(g)})
        else:
            rows.append({"year": int(year), "elasticity": m.params["price_unit"],
                         "pvalue": m.pvalues["price_unit"], "observations": int(m.nobs)})
    return pd.DataFrame(rows)


def _design(sku: str):
    x = get_model_df()
    x = x[x["sku"] == sku].copy()
    x["ln_units"] = np.log(x["units_sold"])
    x["ln_price"] = np.log(x["price_unit"])
    controls = [c for c in CONTROLS if c in x.columns and x[c].nunique(dropna=True) > 1]
    categorical = [c for c in ["channel", "region"] if c in x.columns and x[c].nunique(dropna=True) > 1]
    X = x[["ln_price"] + controls].copy()
    if categorical:
        X = pd.concat([X, pd.get_dummies(x[categorical].astype(str), drop_first=True, dtype=float)], axis=1)
    X = X.apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(x["ln_units"], errors="coerce")
    valid = X.notna().all(axis=1) & y.notna()
    return X.loc[valid], y.loc[valid], controls, categorical


def build_full_model(sku: str):
    """ln(Q) = β0 + E·ln(P) + controles + dummies canal/región + ε (OLS sin penalizar)."""
    X, y, controls, categorical = _design(sku)
    if len(X) < 30 or X["ln_price"].nunique() < 2:
        return None, "No hay suficientes datos para estimar el modelo."
    X = sm.add_constant(X, has_constant="add")
    try:
        model = sm.OLS(y, X).fit()
    except Exception as exc:  # pragma: no cover
        return None, f"No se pudo estimar el modelo: {exc}"
    ci = model.conf_int()
    return {
        "elasticity": float(model.params["ln_price"]),
        "pvalue": float(model.pvalues["ln_price"]),
        "r2": float(model.rsquared), "adj_r2": float(model.rsquared_adj),
        "n": int(model.nobs),
        "ci_low": float(ci.loc["ln_price", 0]), "ci_high": float(ci.loc["ln_price", 1]),
        "controls": controls, "categorical": categorical,
        "model": model, "X": X, "y": y,
    }, None


def compute_vif_table(X: pd.DataFrame, cols: list[str]) -> pd.DataFrame | None:
    cols = [c for c in cols if c in X.columns]
    if len(cols) < 2:
        return None
    sub = X[cols].apply(pd.to_numeric, errors="coerce").dropna()
    # El VIF se calcula con constante para que no se infle artificialmente
    sub = sm.add_constant(sub, has_constant="add")
    rows = []
    for i, col in enumerate(sub.columns):
        if col == "const":
            continue
        try:
            v = variance_inflation_factor(sub.values, i)
        except Exception:
            v = np.nan
        rows.append({"Variable": col, "VIF": v})
    return pd.DataFrame(rows)


def lasso_variable_selection(X, y, alpha_value, protected=("const", "ln_price")):
    Xs = X.copy()
    to_scale = [c for c in Xs.columns if c not in protected]
    if to_scale:
        Xs[to_scale] = (Xs[to_scale] - Xs[to_scale].mean()) / Xs[to_scale].std(ddof=0).replace(0, 1.0)
    alpha_vec = np.array([0.0 if c in protected else alpha_value for c in Xs.columns])
    try:
        reg = sm.OLS(y, Xs).fit_regularized(alpha=alpha_vec, L1_wt=1.0)
    except Exception:
        return None
    out = pd.DataFrame({"Variable": Xs.columns, "Coeficiente estandarizado": np.asarray(reg.params)})
    out["¿Incluida en el modelo?"] = np.where(
        out["Variable"].isin(protected), "Base del modelo (no penalizada)",
        np.where(out["Coeficiente estandarizado"].abs() > 1e-8, "Sí", "No (eliminada)"))
    return out


def elasticity_class(e: float) -> tuple[str, str]:
    if pd.isna(e):
        return "No disponible", "No fue posible estimar la elasticidad."
    if abs(e) > 1:
        return "ELÁSTICA", "La demanda responde proporcionalmente más que el precio."
    if abs(e) < 1:
        return "INELÁSTICA", "La demanda responde proporcionalmente menos que el precio."
    return "UNITARIA", "La respuesta porcentual de la demanda es proporcional al precio."


@st.cache_data(show_spinner="Estimando elasticidades…")
def elasticity_table(skus: tuple[str, ...] | None = None) -> pd.DataFrame:
    skus = list(skus) if skus else priority_skus()
    md = get_model_df()
    rows = []
    for s in skus:
        res, err = build_full_model(s)
        g = md[md["sku"] == s]
        row = {"sku": s, "unidades_totales": float(g["units_sold"].sum()),
               "precio_promedio": float(g["price_unit"].mean())}
        if res is None:
            row.update(elasticidad=np.nan, p_valor=np.nan, ic_inf=np.nan, ic_sup=np.nan, r2=np.nan, n=0)
        else:
            row.update(elasticidad=res["elasticity"], p_valor=res["pvalue"], ic_inf=res["ci_low"],
                       ic_sup=res["ci_high"], r2=res["r2"], n=res["n"])
        rows.append(row)
    out = pd.DataFrame(rows)
    out["significativo_5pct"] = np.where(out["p_valor"] < 0.05, "Sí", "No")
    out["clasificacion"] = out["elasticidad"].apply(
        lambda e: "No disponible" if pd.isna(e) else ("Demanda elástica" if abs(e) > 1 else
                                                       "Demanda inelástica" if abs(e) < 1 else "Unitaria"))
    return out


def seasonal_profile(hist: pd.DataFrame) -> pd.DataFrame:
    monthly = hist.groupby("month").agg(avg_demand=("units_sold", "mean"),
                                        avg_price=("price_unit", "mean")).reset_index()
    od, op = monthly["avg_demand"].mean(), monthly["avg_price"].mean()
    monthly["demand_season_index"] = monthly["avg_demand"] / od if od > 0 else 1.0
    monthly["price_season_index"] = monthly["avg_price"] / op if op > 0 else 1.0
    return monthly


def scenario_calculation(hist, elasticity, mode, value, horizon, demand_mode="Histórica",
                         custom_demand=None, apply_seasonality=True):
    recent = hist.tail(4)
    base_price = recent["price_unit"].mean()
    base_demand = (float(custom_demand) if demand_mode == "Personalizada" and custom_demand is not None
                   else recent["units_sold"].mean())
    profile = seasonal_profile(hist)
    pmap = profile.set_index("month").to_dict("index")
    scen_price = base_price * (1 + float(value) / 100) if mode == "Por porcentaje" else float(value)
    if scen_price <= 0 or base_demand < 0:
        return None
    dates = pd.date_range(hist["week"].max() + pd.Timedelta(days=7), periods=horizon, freq="7D")
    rf = [pmap[int(m)]["demand_season_index"] for m in recent["month"] if int(m) in pmap]
    base_factor = np.mean(rf) if rf else 1.0
    base_factor = base_factor if base_factor > 0 else 1.0
    rows = []
    for d in dates:
        idx = pmap.get(int(d.month), {"demand_season_index": 1.0})["demand_season_index"] \
            if apply_seasonality else base_factor
        bd = base_demand * idx / base_factor
        sd = bd * (scen_price / base_price) ** elasticity
        rows.append({"week": d, "month": d.month, "season_index": idx, "base_demand": bd,
                     "scenario_price": scen_price, "scenario_demand": sd, "base_price": base_price,
                     "base_revenue": bd * base_price, "scenario_revenue": sd * scen_price})
    fut = pd.DataFrame(rows)
    return {
        "base_price": base_price, "scenario_price": scen_price,
        "scenario_demand": fut["scenario_demand"].mean(), "base_demand": fut["base_demand"].mean(),
        "base_revenue": fut["base_revenue"].mean(), "scenario_revenue": fut["scenario_revenue"].mean(),
        "base_revenue_horizon": fut["base_revenue"].sum(),
        "scenario_revenue_horizon": fut["scenario_revenue"].sum(),
        "future": fut, "profile": profile, "change_pct": (scen_price / base_price - 1) * 100,
    }


# ============================================================
# INVENTARIOS
# ============================================================
def policy(D, sigma_D, L, sigma_L, R, service):
    """SS = z·√(L·σD² + D²·σL²);  ROP = D·L + SS;  S = D·(L+R) + SS."""
    z = norm.ppf(service)
    D = np.asarray(D, float)
    ss = np.ceil(z * np.sqrt(L * sigma_D ** 2 + D ** 2 * sigma_L ** 2))
    rop = np.ceil(D * L + ss)
    smax = np.ceil(D * (L + R) + ss)
    return ss, rop, smax


def classify_stock(stock, ss, rop, smax) -> str:
    if stock < ss:
        return "Agotamiento crítico"
    if stock < rop:
        return "Reordenar urgente"
    if stock > smax:
        return "Sobre-stock"
    return "Óptimo"


@st.cache_data(show_spinner=False)
def lead_time_data(sku: str) -> tuple[float, float]:
    """Lead time (semanas) estimado con delivery_days de la base: media y desviación / 7."""
    d = get_raw()
    d = d.loc[d["sku"] == sku, "delivery_days"].dropna()
    return float(d.mean() / 7), float(d.std() / 7)


@st.cache_data(show_spinner=False)
def current_stock(sku: str) -> tuple[pd.Timestamp, float]:
    """Stock disponible de la última semana del intervalo común (suma canal × región)."""
    s = get_series(sku)
    last = s["week"].iloc[-1]
    raw = get_raw()
    v = raw.loc[(raw["sku"] == sku) & (raw["week"] == last), "stock_available"].sum()
    return last, float(v)


@st.cache_data(show_spinner="Calculando parámetros de inventario…")
def inventory_table(service: float = 0.95, lt_mode: str = "data", L: float = 2.0,
                    sigma_L: float = 0.5, R: float = 1.0, demand_source: str = "forecast",
                    sigma_source: str = "oos", horizon: int = FORECAST_HORIZON,
                    price_change_pct: float = 0.0) -> pd.DataFrame:
    """
    Parámetros de inventario de los 10 SKU.
      demand_source: 'forecast' = forecast del modelo campeón; 'hist4' = media últimas 4 semanas.
      sigma_source : 'oos' = RMSE fuera de muestra del campeón; 'hist' = desviación semanal histórica.
      lt_mode      : 'data' = L y σL desde delivery_days; 'fixed' = valores ingresados.
    """
    wf = winner_forecast(horizon, price_change_pct=price_change_pct)
    rows = []
    for sku in priority_skus():
        s = get_series(sku)
        win = winner_info(sku)
        if demand_source == "forecast":
            D = float(wf.loc[wf["sku"] == sku, "value"].mean())
        else:
            D = float(s["units_sold"].tail(4).mean())
        sD = oos_rmse(sku, win["model"]) if sigma_source == "oos" else float(s["units_sold"].std())
        if lt_mode == "data":
            l, sl = lead_time_data(sku)
        else:
            l, sl = float(L), float(sigma_L)
        ss, rop, smax = policy(D, sD, l, sl, R, service)
        week, stock = current_stock(sku)
        rows.append({
            "sku": sku, "modelo_campeon": win["model"], "D": D, "sigma_D": sD,
            "L_sem": l, "sigma_L_sem": sl, "R_sem": R, "nivel_servicio": service,
            "SS": float(ss), "ROP": float(rop), "S": float(smax),
            "semana_stock": week, "stock_actual": stock,
            "brecha_ROP": stock - float(rop),
            "estado": classify_stock(stock, float(ss), float(rop), float(smax)),
        })
    return pd.DataFrame(rows)


@st.cache_data(show_spinner="Preparando datos del tablero de inventarios…")
def bi_inventory_rows() -> pd.DataFrame:
    """Réplica fiel del script R de Power BI (PARTE DE SOFIA), a nivel de fila canal-región."""
    d = get_raw()[["sku", "week", "year", "channel", "region", "units_sold", "stock_available",
                   "delivery_days", "rolling_mean_4", "rolling_std_4"]].copy()
    z = norm.ppf(0.95)
    R, sigma_L = 1, 0.1
    L = d["delivery_days"] / 7
    D = np.where(d["rolling_mean_4"].isna() | (d["rolling_mean_4"] == 0), d["units_sold"], d["rolling_mean_4"])
    sD = d["rolling_std_4"].fillna(0)
    d["Stock_Seguridad"] = np.ceil(z * np.sqrt(L * sD ** 2 + D ** 2 * sigma_L ** 2))
    d["Punto_Reorden_ROP"] = np.ceil(D * L + d["Stock_Seguridad"])
    d["Inventario_Maximo_S"] = np.ceil(D * (L + R) + d["Stock_Seguridad"])
    return d


# ============================================================
# INTEGRACIÓN Y PCA
# ============================================================
@st.cache_data(show_spinner=False)
def executive_table(service: float = 0.95, price_change_pct: float = 0.0) -> pd.DataFrame:
    wf = winner_forecast(FORECAST_HORIZON, price_change_pct=price_change_pct)
    inv = inventory_table(service=service, price_change_pct=price_change_pct)
    el = elasticity_table()
    f12 = wf.groupby("sku")["value"].sum().rename("forecast_12w").reset_index()
    out = (f12.merge(el[["sku", "elasticidad", "p_valor", "significativo_5pct"]], on="sku")
              .merge(inv[["sku", "modelo_campeon", "ROP", "SS", "S", "stock_actual", "estado"]], on="sku"))
    out["WAPE"] = out["sku"].map(lambda s: winner_info(s)["WAPE"])
    return out


@st.cache_data(show_spinner=False)
def sku_indicator_matrix(inv_preset: str = "informe") -> pd.DataFrame:
    """Matriz SKU × 10 indicadores usada en la reducción de dimensiones (PCA)."""
    raw = get_raw()
    if inv_preset == "informe":   # escenario Tabla 8 del informe
        inv = inventory_table(0.95, "fixed", 2.0, 0.5, 1.0, "hist4", "hist")
    else:
        inv = inventory_table()
    el = elasticity_table()
    summ = get_summary().set_index("sku")
    rows = []
    for s in priority_skus():
        g = raw[raw["sku"] == s]
        rows.append({
            "sku": s,
            "Ingresos": summ.loc[s, "total_revenue"],
            "Unidades": summ.loc[s, "total_units"],
            "Precio promedio": g["price_unit"].mean(),
            "Participación promocional": g["promotion_flag"].mean(),
            "Stock promedio": g["stock_available"].mean(),
            "Días de entrega": g["delivery_days"].mean(),
            "WAPE ganador": winner_info(s)["WAPE"],
            "Elasticidad": float(el.loc[el["sku"] == s, "elasticidad"].iloc[0]),
            "Stock de seguridad": float(inv.loc[inv["sku"] == s, "SS"].iloc[0]),
            "ROP": float(inv.loc[inv["sku"] == s, "ROP"].iloc[0]),
        })
    return pd.DataFrame(rows).set_index("sku")


def run_pca(matrix: pd.DataFrame, threshold: float = 0.80) -> dict:
    scaler = StandardScaler()
    Z = scaler.fit_transform(matrix.values)
    pca = PCA().fit(Z)
    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)
    k = int(np.argmax(cum >= threshold) + 1)
    scores = pd.DataFrame(pca.transform(Z)[:, :k], index=matrix.index,
                          columns=[f"PC{i+1}" for i in range(k)])
    loadings = pd.DataFrame(pca.components_[:k].T, index=matrix.columns,
                            columns=[f"PC{i+1}" for i in range(k)])
    return {"Z": pd.DataFrame(Z, index=matrix.index, columns=matrix.columns),
            "evr": evr, "cum": cum, "k": k, "scores": scores, "loadings": loadings}


def describe_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        if s.empty:
            continue
        mode = s.mode()
        rows.append({
            "Variable": c, "Descripción": VAR_LABELS.get(c, c), "n": int(s.size),
            "Media": s.mean(), "Mediana": s.median(), "Moda": mode.iloc[0] if len(mode) else np.nan,
            "Desv. estándar": s.std(), "Mínimo": s.min(), "Máximo": s.max(),
            "Coef. variación (%)": 100 * s.std() / s.mean() if s.mean() else np.nan,
        })
    return pd.DataFrame(rows)
