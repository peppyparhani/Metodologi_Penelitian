import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Prediksi Stunting Balita",
    layout="wide"
)

st.title("📊 Prediksi Persentase Stunting Balita")
st.markdown(
    """
    Aplikasi ini membandingkan **Decision Tree** dan **Random Forest**
    untuk memprediksi persentase stunting berdasarkan data historis.
    """
)

# ===============================
# UPLOAD DATA
# ===============================
uploaded_file = st.file_uploader(
    "📂 Upload dataset stunting (CSV)",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Silakan upload file CSV untuk memulai.")
    st.stop()

# ===============================
# LOAD & CLEAN DATA
# ===============================
df = pd.read_csv(uploaded_file)
df.columns = [c.strip() for c in df.columns]

df['persentase_balita_stunting'] = pd.to_numeric(
    df['persentase_balita_stunting'], errors='coerce'
)

df = df.dropna(subset=[
    'persentase_balita_stunting',
    'nama_kabupaten_kota',
    'tahun'
])

df['tahun'] = df['tahun'].astype(int)

st.subheader("📌 Contoh Data")
st.dataframe(df.head(), use_container_width=True)

# ===============================
# FEATURE ENGINEERING
# ===============================
rows = []

for (prov, kode, kab), g in df.groupby(
    ['nama_provinsi','kode_kabupaten_kota','nama_kabupaten_kota']
):
    g = g.sort_values('tahun')
    years = g['tahun'].values
    vals = g['persentase_balita_stunting'].values

    for i in range(1, len(vals)):
        v1 = vals[i-1]
        v2 = vals[i-2] if i-2 >= 0 else np.nan
        v3 = vals[i-3] if i-3 >= 0 else np.nan

        mean_prev = np.nanmean([v1, v2, v3])

        slope = 0
        if i >= 2:
            xi = years[max(0, i-3):i]
            yi = vals[max(0, i-3):i]
            if len(xi) >= 2:
                slope = np.polyfit(xi, yi, 1)[0]

        rows.append([
            prov, kode, kab,
            v1, v2, v3, mean_prev, slope,
            vals[i]
        ])

data = pd.DataFrame(rows, columns=[
    'provinsi','kode_kabupaten_kota','kabupaten_kota',
    'lag1','lag2','lag3','mean_prev','slope_prev',
    'target'
])

features = ['lag1','lag2','lag3','mean_prev','slope_prev']
X = data[features].fillna(data[features].mean())
y = data['target']

# ===============================
# SPLIT DATA
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
# MODEL SELECTION
# ===============================
model_choice = st.selectbox(
    "⚙️ Pilih Model",
    ["Decision Tree", "Random Forest"]
)

if model_choice == "Decision Tree":
    model = DecisionTreeRegressor(
        max_depth=5,
        min_samples_leaf=5,
        random_state=42
    )
else:
    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

# ===============================
# TRAIN & PREDICT
# ===============================
model.fit(X_train, y_train)
pred_test = model.predict(X_test)

# ===============================
# EVALUATION
# ===============================
MAE = mean_absolute_error(y_test, pred_test)
RMSE = sqrt(mean_squared_error(y_test, pred_test))
R2 = r2_score(y_test, pred_test)
MAPE = np.mean(np.abs((y_test - pred_test) / y_test)) * 100
Accuracy = 100 - MAPE

st.subheader("📈 Evaluasi Model")
col1, col2, col3 = st.columns(3)

col1.metric("MAE", f"{MAE:.3f}")
col2.metric("RMSE", f"{RMSE:.3f}")
col3.metric("R²", f"{R2:.3f}")

st.metric("Accuracy (%)", f"{Accuracy:.2f}")

# ===============================
# FEATURE IMPORTANCE (FIX SIZE)
# ===============================
st.subheader("🔍 Feature Importance")

imp = pd.DataFrame({
    'Fitur': features,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

fig, ax = plt.subplots(figsize=(5, 3))
ax.barh(imp['Fitur'], imp['Importance'])
ax.invert_yaxis()
ax.set_xlabel("Importance")
st.pyplot(fig, use_container_width=True)

# ===============================
# PREDICTION VS ACTUAL (FIX SIZE)
# ===============================
st.subheader("📉 Prediksi vs Aktual")

fig, ax = plt.subplots(figsize=(5, 4))
ax.scatter(y_test, pred_test, s=15)
ax.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()]
)
ax.set_xlabel("Aktual")
ax.set_ylabel("Prediksi")
st.pyplot(fig, use_container_width=True)

# ===============================
# NEXT YEAR PREDICTION
# ===============================
st.subheader("📆 Prediksi Tahun Berikutnya")

results = []

for (prov, kode, kab), g in df.groupby(
    ['nama_provinsi','kode_kabupaten_kota','nama_kabupaten_kota']
):
    g = g.sort_values('tahun')
    vals = g['persentase_balita_stunting'].values

    v1 = vals[-1]
    v2 = vals[-2] if len(vals) >= 2 else np.nan
    v3 = vals[-3] if len(vals) >= 3 else np.nan
    mean_prev = np.nanmean([v1, v2, v3])

    slope = 0
    if len(vals) >= 2:
        slope = np.polyfit(np.arange(len(vals[-3:])), vals[-3:], 1)[0]

    feat = np.array([v1, v2, v3, mean_prev, slope])
    feat = np.where(np.isnan(feat), np.nanmean(feat), feat)

    pred_val = model.predict(feat.reshape(1, -1))[0]
    results.append([prov, kab, pred_val])

pred_df = pd.DataFrame(results, columns=[
    'Provinsi','Kabupaten/Kota','Prediksi Stunting'
])

def classify(x):
    if x < 10:
        return "Rendah"
    elif x <= 20:
        return "Sedang"
    else:
        return "Tinggi"

pred_df['Prioritas'] = pred_df['Prediksi Stunting'].apply(classify)

st.dataframe(pred_df, use_container_width=True)

st.download_button(
    "📥 Download Hasil Prediksi",
    pred_df.to_csv(index=False),
    file_name="prediksi_stunting.csv"
)
