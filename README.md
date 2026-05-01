# NYC Taxi + Weather Analysis

Does weather actually change how many people take a cab in New York City? That's the question we set out to answer. Using a full year of NYC yellow taxi trip records alongside daily weather data from Central Park, we ran a data science investigation to find out how precipitation, snowfall, and temperature affect daily taxi demand.

**Team:** Donovan McHenry, Rich Cavanagh, Christopher Kucharek
**Course:** CS301, Milestone 2

## What We Did

We pulled 12 months of 2023 TLC yellow taxi trip data and merged it with NOAA daily weather summaries from the Central Park station. From there we ran the full pipeline: cleaning, EDA, hypothesis testing, and two regression models.

A few things we found that weren't obvious going in:

- Snow days drop trip volume by about 27%, which is way more than cold weather alone (~13%)
- Light rain (under 0.25 inches) doesn't significantly change demand. The "it's raining, grab a cab" effect is mostly a myth for light drizzle
- Day of week actually explains more variance than any single weather variable, which says a lot about how predictable NYC commuters are

## Data Sources

- **NYC TLC Yellow Taxi Trip Records** from [nyc.gov](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page), downloaded directly in the notebook
- **NOAA Central Park Daily Summaries** (Station GHCND:USW00094728), pulled via the [NCEI API](https://www.ncei.noaa.gov/cdo-web/)

No manual downloads needed. The notebook handles everything automatically.

## How to Run

Open the notebook in Google Colab and run the cells top to bottom. It will download the taxi data from the TLC CDN and fetch weather data from the NOAA API on its own.

The taxi download pulls about 600MB across 12 monthly files, so give it 2 to 3 minutes.

```
Runtime > Run all
```

## What's in the Notebook

1. Data loading and merging
2. Exploratory data analysis: distributions, correlation heatmap, bivariate plots
3. Hypothesis testing: Welch's t-test with normality and variance checks, plus a Mann-Whitney robustness check
4. Model building: OLS linear regression and Random Forest, both with cross-validation
5. Knowledge discovery: feature importance, sub-group breakdowns, actionable findings

## Requirements

The notebook runs on standard Colab. No extra setup needed beyond what's already installed. If running locally:

```
pip install pandas numpy matplotlib seaborn scipy scikit-learn pyarrow
```
