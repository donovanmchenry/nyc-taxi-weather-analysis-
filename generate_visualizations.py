import io
import os
import requests
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

OUTPUT = os.environ.get('OUTPUT_DIR', '/output')
os.makedirs(OUTPUT, exist_ok=True)

plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor':   'white',
    'axes.spines.top':  False,
    'axes.spines.right':False,
    'axes.grid':        True,
    'grid.alpha':       0.3,
    'font.size':        11,
})
BLUE  = '#185FA5'
RED   = '#A32D2D'
GREEN = '#3B6D11'
AMBER = '#BA7517'
TEAL  = '#0F6E56'
GRAY  = '#5F5E5A'

# ── 1. Load data ──────────────────────────────────────────────────────────────
print('Downloading TLC taxi data (12 months)...')
BASE = 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-{:02d}.parquet'
COLS = ['tpep_pickup_datetime', 'trip_distance', 'fare_amount', 'passenger_count']
frames = []
for month in range(1, 13):
    print(f'  Month {month:02d}...', flush=True)
    frames.append(pd.read_parquet(BASE.format(month), columns=COLS))
raw_taxi = pd.concat(frames, ignore_index=True)
raw_taxi['date'] = pd.to_datetime(raw_taxi['tpep_pickup_datetime']).dt.normalize()
raw_taxi = raw_taxi[raw_taxi['date'].dt.year == 2023]
taxi_daily = (
    raw_taxi.groupby('date')
    .agg(trip_count=('tpep_pickup_datetime','count'),
         fare_avg=('fare_amount','mean'),
         dist_avg=('trip_distance','mean'))
    .reset_index()
)
print(f'Taxi daily shape: {taxi_daily.shape}')

print('Fetching NOAA weather data...')
resp = requests.get(
    'https://www.ncei.noaa.gov/access/services/data/v1',
    params={
        'dataset':   'daily-summaries',
        'stations':  'USW00094728',
        'startDate': '2023-01-01',
        'endDate':   '2023-12-31',
        'dataTypes': 'PRCP,SNOW,SNWD,TMAX,TMIN',
        'format':    'csv',
        'units':     'standard',
    },
    timeout=60,
)
resp.raise_for_status()
weather_df = pd.read_csv(io.StringIO(resp.text))
weather_df = weather_df.rename(columns={
    'DATE':'date','TMAX':'tmax','TMIN':'tmin',
    'PRCP':'prcp','SNOW':'snow','SNWD':'snwd'
})
weather_df['date'] = pd.to_datetime(weather_df['date'])
weather_df = weather_df[['date','tmax','tmin','prcp','snow','snwd']]
print(f'Weather shape: {weather_df.shape}')

# ── 2. Merge & feature engineering ───────────────────────────────────────────
daily = taxi_daily.merge(weather_df, on='date', how='inner')
daily = daily[daily['date'].dt.year == 2023].sort_values('date').reset_index(drop=True)

daily['dayofweek'] = daily['date'].dt.dayofweek
daily['month']     = daily['date'].dt.month
daily['dow_sin']   = np.sin(2 * np.pi * daily['dayofweek'] / 7)
daily['dow_cos']   = np.cos(2 * np.pi * daily['dayofweek'] / 7)
daily['adverse']   = ((daily['prcp'] > 0) | (daily['tmax'] < 32)).astype(int)
daily['snow_day']  = (daily['snow'] > 0).astype(int)
daily['prcp_cat']  = pd.cut(daily['prcp'], bins=[-0.01, 0, 0.25, 999],
                             labels=['None','Light (0-0.25")','Heavy (>0.25")'])
daily['weather_cat'] = 'Favorable'
daily.loc[(daily['tmax'] < 32) & (daily['snow'] == 0), 'weather_cat'] = 'Cold (<32F)'
daily.loc[daily['prcp_cat'] == 'Light (0-0.25")', 'weather_cat'] = 'Light rain'
daily.loc[daily['prcp_cat'] == 'Heavy (>0.25")', 'weather_cat'] = 'Heavy rain'
daily.loc[daily['snow_day'] == 1, 'weather_cat'] = 'Snow day'
print(f'Merged shape: {daily.shape}')

# ── 3. Figure 1: Feature distributions ───────────────────────────────────────
print('Generating fig_distributions.png...')
plot_vars = [
    ('trip_count','Daily Trip Count', BLUE),
    ('tmax',      'Max Temp (F)',     AMBER),
    ('prcp',      'Precipitation (in)',TEAL),
    ('snow',      'Snowfall (in)',     GRAY),
    ('snwd',      'Snow Depth (in)',   GRAY),
    ('fare_avg',  'Avg Fare ($)',      GREEN),
]
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
fig.suptitle('Feature Distributions', fontsize=14, fontweight='bold')
for ax, (col, label, color) in zip(axes.flat, plot_vars):
    data = daily[col].dropna()
    ax.hist(data, bins=30, color=color, alpha=0.75, edgecolor='white')
    ax.axvline(data.mean(),   color='black', linestyle='--', linewidth=1.2, label=f'Mean={data.mean():.1f}')
    ax.axvline(data.median(), color='red',   linestyle=':',  linewidth=1.2, label=f'Median={data.median():.1f}')
    ax.set_title(label, fontweight='bold')
    ax.set_xlabel(label); ax.set_ylabel('Days')
    ax.legend(fontsize=9)
    ax.text(0.97, 0.95, f'skew={data.skew():.2f}', transform=ax.transAxes,
            ha='right', va='top', fontsize=9, color='gray')
plt.tight_layout()
plt.savefig(f'{OUTPUT}/fig_distributions.png', dpi=150, bbox_inches='tight')
plt.close()

# ── 4. Figure 2: Correlation heatmap ─────────────────────────────────────────
print('Generating fig_correlation.png...')
corr_cols = ['trip_count','fare_avg','dist_avg','tmax','tmin','prcp','snow','snwd']
corr = daily[corr_cols].corr()
fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            vmin=-1, vmax=1, linewidths=0.5, ax=ax, square=True,
            cbar_kws={'shrink': 0.8})
ax.set_title('Pearson Correlation Matrix', fontsize=13, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig(f'{OUTPUT}/fig_correlation.png', dpi=150, bbox_inches='tight')
plt.close()

# ── 5. Figure 3: Bivariate analysis ──────────────────────────────────────────
print('Generating fig_bivariate.png...')
fav = daily[daily['adverse'] == 0]
adv = daily[daily['adverse'] == 1]
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax = axes[0]
ax.scatter(fav['tmax'], fav['trip_count']/1000, color=BLUE, alpha=0.5, s=25, label=f'Favorable (n={len(fav)})')
ax.scatter(adv['tmax'], adv['trip_count']/1000, color=RED,  alpha=0.5, s=25, label=f'Adverse (n={len(adv)})')
ax.axvline(32, color=GRAY, linestyle='--', linewidth=1, alpha=0.7, label='Freezing (32F)')
z = np.polyfit(daily['tmax'], daily['trip_count']/1000, 1)
xr = np.linspace(daily['tmax'].min(), daily['tmax'].max(), 100)
ax.plot(xr, np.polyval(z, xr), color='black', linewidth=1.5, alpha=0.6, label='OLS trend')
ax.set_xlabel('Max Temperature (F)', fontweight='bold')
ax.set_ylabel('Daily Trips (thousands)', fontweight='bold')
ax.set_title('Temperature vs. Trip Volume', fontweight='bold')
ax.legend(fontsize=9)
ax = axes[1]
order = ['Favorable','Cold (<32F)','Light rain','Heavy rain','Snow day']
colors_box = [BLUE, AMBER, TEAL, RED, GRAY]
bp = ax.boxplot(
    [daily[daily['weather_cat']==c]['trip_count'].values/1000 for c in order],
    labels=order, patch_artist=True,
    medianprops={'color':'white','linewidth':2}
)
for patch, color in zip(bp['boxes'], colors_box):
    patch.set_facecolor(color); patch.set_alpha(0.75)
ax.set_ylabel('Daily Trips (thousands)', fontweight='bold')
ax.set_title('Trip Volume by Weather Condition', fontweight='bold')
ax.tick_params(axis='x', rotation=25)
plt.tight_layout()
plt.savefig(f'{OUTPUT}/fig_bivariate.png', dpi=150, bbox_inches='tight')
plt.close()

# ── 6. Figure 4: Time series ──────────────────────────────────────────────────
print('Generating fig_timeseries.png...')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
ax1.fill_between(daily['date'], daily['trip_count']/1000, alpha=0.3, color=BLUE)
ax1.plot(daily['date'], daily['trip_count']/1000, color=BLUE, linewidth=0.8)
snow_days = daily[daily['snow_day'] == 1]
ax1.scatter(snow_days['date'], snow_days['trip_count']/1000, color=GRAY, s=30, zorder=5, label='Snow day', marker='v')
ax1.set_ylabel('Daily Trips (k)', fontweight='bold')
ax1.set_title('Daily Taxi Trip Volume 2023 (v = snow days)', fontweight='bold')
ax1.legend(fontsize=9)
ax2.bar(daily['date'], daily['tmax'], color=AMBER, alpha=0.6, label='Tmax (F)')
ax2.bar(daily[daily['snow']>0]['date'], daily[daily['snow']>0]['snow']*3,
        color=TEAL, alpha=0.8, label='Snowfall x3 (in)')
ax2.axhline(32, color=RED, linestyle='--', linewidth=1, label='Freezing (32F)')
ax2.set_ylabel('F / Snowfall', fontweight='bold')
ax2.set_xlabel('Date', fontweight='bold')
ax2.legend(fontsize=9)
plt.tight_layout()
plt.savefig(f'{OUTPUT}/fig_timeseries.png', dpi=150, bbox_inches='tight')
plt.close()

# ── 7. Hypothesis test ────────────────────────────────────────────────────────
favorable = daily[daily['adverse']==0]['trip_count']
adverse   = daily[daily['adverse']==1]['trip_count']
t_stat, p_value = stats.ttest_ind(favorable, adverse, equal_var=False)
pooled_std = np.sqrt((favorable.std()**2 + adverse.std()**2) / 2)
cohens_d   = (favorable.mean() - adverse.mean()) / pooled_std
base_mean  = favorable.mean()
cold_mean  = daily[daily['weather_cat']=='Cold (<32F)']['trip_count'].mean()
snow_mean  = daily[daily['weather_cat']=='Snow day']['trip_count'].mean()
_, p_light = stats.ttest_ind(favorable, daily[daily['weather_cat']=='Light rain']['trip_count'])
_, p_heavy = stats.ttest_ind(favorable, daily[daily['weather_cat']=='Heavy rain']['trip_count'])

print('Generating fig_hypothesis.png...')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
favorable.plot.kde(ax=ax1, color=BLUE, linewidth=2, label=f'Favorable (mean={favorable.mean()/1000:.0f}k)')
adverse.plot.kde(ax=ax1,   color=RED,  linewidth=2, label=f'Adverse   (mean={adverse.mean()/1000:.0f}k)')
ax1.axvline(favorable.mean(), color=BLUE, linestyle='--', alpha=0.7)
ax1.axvline(adverse.mean(),   color=RED,  linestyle='--', alpha=0.7)
ax1.set_xlabel('Daily Trip Count'); ax1.set_ylabel('Density')
ax1.set_title(f'Trip Distribution by Weather\n(t={t_stat:.2f}, p={p_value:.2e})', fontweight='bold')
ax1.legend()
ax1.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x/1000:.0f}k'))
order2 = ['Favorable','Cold (<32F)','Light rain','Heavy rain','Snow day']
colors2 = [BLUE, AMBER, TEAL, RED, GRAY]
means2, cis2, ns2 = [], [], []
for g in order2:
    d = daily[daily['weather_cat']==g]['trip_count']
    means2.append(d.mean()); cis2.append(1.96*d.std()/np.sqrt(len(d))); ns2.append(len(d))
bars = ax2.bar(order2, [m/1000 for m in means2], color=colors2, alpha=0.8)
ax2.errorbar(order2, [m/1000 for m in means2], yerr=[c/1000 for c in cis2],
             fmt='none', color='black', capsize=5, linewidth=1.5)
for bar, n in zip(bars, ns2):
    ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
             f'n={n}', ha='center', va='bottom', fontsize=9, color=GRAY)
ax2.set_ylabel('Mean Daily Trips (thousands)', fontweight='bold')
ax2.set_title('Mean Trip Volume by Weather Sub-Group\n(error bars = 95% CI)', fontweight='bold')
ax2.tick_params(axis='x', rotation=20)
plt.tight_layout()
plt.savefig(f'{OUTPUT}/fig_hypothesis.png', dpi=150, bbox_inches='tight')
plt.close()

# ── 8. Model building ─────────────────────────────────────────────────────────
FEATURES = ['tmax','prcp','snow','snwd','dow_sin','dow_cos']
X = daily[FEATURES]; y = daily['trip_count']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler().fit(X_train)
X_train_s = pd.DataFrame(scaler.transform(X_train), columns=FEATURES)
X_test_s  = pd.DataFrame(scaler.transform(X_test),  columns=FEATURES)

lr = LinearRegression().fit(X_train, y_train)
lr_s = LinearRegression().fit(X_train_s, y_train)
y_pred_lr = lr.predict(X_test)
r2_lr  = r2_score(y_test, y_pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
mae_lr  = mean_absolute_error(y_test, y_pred_lr)

rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
r2_rf   = r2_score(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
mae_rf  = mean_absolute_error(y_test, y_pred_rf)

perm_imp = permutation_importance(rf, X_test, y_test, n_repeats=20, random_state=42)
fi_df = pd.DataFrame({'feature': FEATURES,
                      'importance': perm_imp.importances_mean,
                      'std': perm_imp.importances_std}).sort_values('importance', ascending=False)
total = fi_df['importance'].sum()

print('Generating fig_model_eval.png...')
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
lim = [min(y_test.min(), y_pred_lr.min())/1000-2, max(y_test.max(), y_pred_lr.max())/1000+2]
ax = axes[0]
ax.scatter(y_test/1000, y_pred_lr/1000, color=BLUE, alpha=0.5, s=20)
ax.plot(lim, lim, color='red', linestyle='--', linewidth=1.5, label='Perfect fit')
ax.set_xlabel('Actual (k)'); ax.set_ylabel('Predicted (k)')
ax.set_title(f'OLS: Predicted vs Actual\n(R2={r2_lr:.3f}, RMSE={rmse_lr/1000:.1f}k)', fontweight='bold')
ax.legend(fontsize=9)
ax = axes[1]
ax.scatter(y_test/1000, y_pred_rf/1000, color=GREEN, alpha=0.5, s=20)
ax.plot(lim, lim, color='red', linestyle='--', linewidth=1.5, label='Perfect fit')
ax.set_xlabel('Actual (k)'); ax.set_ylabel('Predicted (k)')
ax.set_title(f'Random Forest: Predicted vs Actual\n(R2={r2_rf:.3f}, RMSE={rmse_rf/1000:.1f}k)', fontweight='bold')
ax.legend(fontsize=9)
ax = axes[2]
fi_sorted = fi_df.sort_values('importance')
colors_fi = [BLUE if 'dow' in f else RED if f in ['snow','prcp','snwd'] else AMBER
             for f in fi_sorted['feature']]
ax.barh(fi_sorted['feature'], fi_sorted['importance']/total*100, color=colors_fi, alpha=0.8)
ax.errorbar(fi_sorted['importance']/total*100, fi_sorted['feature'],
            xerr=fi_sorted['std']/total*100, fmt='none', color='black', capsize=3)
ax.set_xlabel('Permutation Importance (%)')
ax.set_title('RF Feature Importance', fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT}/fig_model_eval.png', dpi=150, bbox_inches='tight')
plt.close()

# ── 9. Figure 7: Knowledge discovery ─────────────────────────────────────────
weather_feats  = ['tmax','prcp','snow','snwd']
temporal_feats = ['dow_sin','dow_cos']
weather_total  = fi_df[fi_df['feature'].isin(weather_feats)]['importance'].sum() / total * 100
temporal_total = fi_df[fi_df['feature'].isin(temporal_feats)]['importance'].sum() / total * 100

print('Generating fig_discovery.png...')
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Knowledge Discovery: Key Findings', fontsize=14, fontweight='bold')
ax = axes[0,0]
ax.barh(['Day-of-week\n(temporal)','Weather\n(combined)'], [temporal_total, weather_total],
        color=[BLUE, RED], alpha=0.8)
ax.set_xlabel('% of RF Feature Importance')
ax.set_title('Finding 1: Day-of-Week vs. Weather', fontweight='bold')
for i, v in enumerate([temporal_total, weather_total]):
    ax.text(v+0.3, i, f'{v:.1f}%', va='center', fontweight='bold')
ax = axes[0,1]
grp_labels = ['Light rain\n(<=0.25")', 'Cold\n(<32F)', 'Heavy rain\n(>0.25")', 'Snow day']
grp_data   = ['Light rain', 'Cold (<32F)', 'Heavy rain', 'Snow day']
pcts2, pvals2 = [], []
for g in grp_data:
    d = daily[daily['weather_cat']==g]['trip_count']
    if len(d) < 3: pcts2.append(0); pvals2.append(1); continue
    _, p = stats.ttest_ind(favorable, d)
    pcts2.append((d.mean()-base_mean)/base_mean*100); pvals2.append(p)
colors_sig = [GRAY if p > 0.05 else RED for p in pvals2]
bars2 = ax.bar(grp_labels, pcts2, color=colors_sig, alpha=0.8)
for bar, p in zip(bars2, pvals2):
    label = 'n.s.' if p > 0.05 else f'p={p:.4f}'
    ypos  = bar.get_height() + (0.3 if bar.get_height() > 0 else -1.5)
    ax.text(bar.get_x()+bar.get_width()/2, ypos, label, ha='center', fontsize=8)
ax.axhline(0, color='black', linewidth=0.8)
ax.set_ylabel('% change vs favorable'); ax.set_title('Finding 2: Light Rain Not Significant', fontweight='bold')
ax = axes[1,0]
cats3  = ['Favorable','Cold only\n(<32F)','Snow day']
means3 = [base_mean, cold_mean, snow_mean]
ax.bar(cats3, [m/1000 for m in means3], color=[BLUE, AMBER, GRAY], alpha=0.8)
ax.set_ylabel('Mean daily trips (thousands)'); ax.set_title('Finding 3: Snow >> Cold Alone', fontweight='bold')
for i, m in enumerate(means3):
    ax.text(i, m/1000+0.3, f'{m/1000:.0f}k', ha='center', fontweight='bold')
ax = axes[1,1]
resid_lr = np.abs(y_test - y_pred_lr)
resid_rf = np.abs(y_test - y_pred_rf)
ax.scatter(y_test/1000, resid_lr/1000, color=BLUE,  alpha=0.4, s=15, label=f'OLS  MAE={mae_lr/1000:.1f}k')
ax.scatter(y_test/1000, resid_rf/1000, color=GREEN, alpha=0.4, s=15, label=f'RF   MAE={mae_rf/1000:.1f}k')
ax.set_xlabel('Actual trips (k)'); ax.set_ylabel('Absolute error (k)')
ax.set_title('Finding 4: RF Reduces Errors', fontweight='bold'); ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(f'{OUTPUT}/fig_discovery.png', dpi=150, bbox_inches='tight')
plt.close()

print(f'\nDone. All figures saved to {OUTPUT}/')
for f in sorted(os.listdir(OUTPUT)):
    print(f'  {f}')
