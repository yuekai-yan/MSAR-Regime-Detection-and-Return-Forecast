## Speech-Driven Market Regime Detection with Markov-Switching Models

## Abstract  
Financial markets experience abrupt shifts between growth and downturn cycles due to changing macroeconomic conditions, policy shocks, and investor sentiment. Traditional single-regime time-series models struggle to capture nonlinear structural breaks, resulting in delayed market risk recognition and sub-optimal decisions.  

This project proposes a **Markov-Switching Autoregressive (MSAR) early-warning system** that adaptively detects market regime transitions and triggers anomaly-based alerts for risk management and trading support. Using stocks/ETFs time-series and initial probabilities for each regime based on **exogenous variables derived solely from CEO speeches (linguistic signals)**, we aim to identify bull/bear phases, quantify switch probabilities, and explore how corporate communication may affect market regime shifts.  

Our goal is to build a **transparent, data-driven risk dashboard** for real-time monitoring and actionable financial insights, bridging econometric rigor and practical trading workflow. The entire data processing pipeline, including preprocessing, model fitting, and analysis, is implemented in `results.ipynb`.

The project website is available at: https://epfl-ada.github.io/ada-2025-project-alwaysdominatingacademics/


## Research Questions  
- Does the **content or linguistic style of CEO speeches** influence market regimes or volatility patterns?
- How can CEO speech information be incorporated into the MSAR model?
- Will incorporating **time-varying exogenous variables** (such as information extracted from CEO speeches) into the transition probability matrix improve the MSAR model’s ability to capture regime switches and forecast next-day income?
- Under what data characteristics does MSAR outperform LSTM in predicting income, and under what characteristics does LSTM outperform MSAR?


## Proposed Additional Datasets

| Dataset | Source | Use | Format | Size |
|---|---|---|---|---|
| Equity Index Data (S&P500, NASDAQ, etc.) | Yahoo Finance / Kaggle | Regime detection baseline | CSV / API | ~5–20 years, <5MB |
| [NASDAQ-Index(2016-2020)](#10-visualization) | Internal dataset | Refined NASDAQ historical index for labeling | XLSX | 78KB |
| [2016_2020_speech](#part-ii--speechmarket-link-ceo-language-features-as-exogenous-signals) | Internal transcripts (earnings calls, interviews) | CEO linguistic feature extraction | TXT (ZIP) | 3.5MB |
| VIX Volatility Index | CBOE | Stress indicator | CSV / API | ~2MB |
| US Macro Data (GDP, Unemployment, CPI) | FRED API | Macro-regime triggers | JSON / CSV | <10MB |



**Plan**
- Fetch via `yfinance` & `fredapi`
- Daily/Monthly frequency syncing
- Standardization & missing value interpolation
- Feature engineering: return vol, macro growth shocks, transition probability curves

## Methods

## Part I — Method (Speech → Market Link)

We test whether **changes in CEO language** are associated with subsequent market movements.

**1) Text → embedding vector**  
For each CEO transcript, we convert the text into a numerical representation using an OpenAI embedding model. The transcript is split into chunks, each chunk is embedded, and we average chunk embeddings to obtain a single document vector $\mathbf{d}$.

**2) Linguistic feature scoring (anchor similarities)**  
To quantify linguistic traits (e.g., *joy*, *uncertainty*, *fear*, *trust*), we compare the document vector $\mathbf{d}$ with pre-defined anchor vectors representing each trait. Feature scores are computed using cosine similarity and mapped to $[0,1]$.

Cosine similarity:

$$
\cos(\mathbf{x},\mathbf{y})=\frac{\mathbf{x}\cdot\mathbf{y}}{\|\mathbf{x}\|\|\mathbf{y}\|}
$$

Normalization:

$$
\mathrm{to01}(c)=\frac{c+1}{2}
$$


This yields transcript-level feature values $x_{t,c}^{(k)}\in[0,1]$ for each feature $k\in\{\text{joy},\text{uncertainty},\text{fear},\text{trust}\}$, where $t$ denotes date and $c$ denotes CEO/company.

**3) Normalization and “sentiment change” (z-score)**  
To capture *relative* linguistic shifts rather than fixed speaking styles, we standardize each feature within each CEO/company using only past history: $$z_{t,c}=\frac{x_{t,c}-\mu_c}{\sigma_c}$$.  
Here $\mu_c$ and $\sigma_c$ are the historical mean and standard deviation for that CEO/company. The resulting $z_{t,c}$ is interpreted as a **linguistic (sentiment) change**: $z_{t,c}>0$ indicates the feature is higher than the speaker’s baseline, and $z_{t,c}<0$ indicates it is lower.

**4) Trading-day alignment and aggregation**  
Each transcript date is aligned to the **next trading day** to handle weekends/holidays. If multiple CEOs speak on the same date, we aggregate their z-scores by daily mean (or a weighted mean if external weights are available) to obtain a single market-level signal $z_t$ per trading day.

**5) Market outcome definition (3-day window)**  
We measure the market response using the future 3-trading-day cumulative return starting from the next trading day.

$$
\mathrm{cum\_logret}_3(t)=\mathrm{logret}_{t+1}+\mathrm{logret}_{t+2}+\mathrm{logret}_{t+3}
$$


$$
\mathrm{cum\_ret}_3(t)=\exp\!\left(\mathrm{cum\_logret}_3(t)\right)-1
$$


**6) Statistical association tests**  
To quantify the relationship between linguistic change $z_t$ and the market outcome, we compute Pearson and Spearman correlations: $$r_P=\mathrm{corr}(z_t,\mathrm{cum\_ret}_3(t))$$ and $$r_S=\mathrm{corr}(\mathrm{rank}(z_t),\mathrm{rank}(\mathrm{cum\_logret}_3(t)))$$, together with their corresponding p-values. Consistent sign and significance across both tests is treated as stronger evidence of a robust relationship.



## Part II — Econometric Modeling: Market Regimes and Information-Driven Transitions

Financial markets are commonly characterized by persistent regimes—such as **Bull** and **Bear** markets—that differ in their return dynamics, volatility levels, and persistence. These regimes are not directly observable and evolve in response to changes in risk perceptions and the overall information environment faced by investors.

To capture this structure, we employ a **Markov-Switching Autoregressive (MSAR)** model, which allows both return dynamics and regime persistence to depend on an unobserved (latent) market state. The central economic idea is that **market regimes reflect persistent states of confidence and risk, while changes in information quality can affect the stability of these states**.

---

### 1. Regime-Switching Return Dynamics

Let $r_t$ denote the asset return at time $t$. Returns follow a regime-dependent autoregressive process:

$$
r_t = \mu_{s_t} + \varphi_{s_t} r_{t-1} + \sigma_{s_t} \varepsilon_t,
\quad s_t \in \{0,1\}
$$

where:

- $s_t = 1$ denotes a **Bull** market and $s_t = 0$ denotes a **Bear** market
- Each regime has its own mean $\mu_{s_t}$, volatility $\sigma_{s_t}^2$, and persistence $\varphi_{s_t}$
- $\varepsilon_t \sim \mathcal{N}(0,1)$

This specification captures the idea that market behavior differs systematically across regimes rather than fluctuating around a single stationary process.

---

### 2. Regime Transitions and Persistence

The latent regime evolves according to a first-order Markov process. In the baseline MSAR model, transition probabilities are constant over time:

$$
P_{ij} = \Pr(s_{t+1} = i \mid s_t = j), 
\quad i,j \in \{0,1\}
$$

These probabilities summarize **regime persistence** and the likelihood of regime shifts. Based on this structure, the model delivers filtered and predictive regime probabilities as well as one-step-ahead return forecasts.

---

### 3. Time-Varying Transition Probabilities (TVTP)

Market regimes are influenced not only by past price dynamics but also by changes in the information environment. To capture this effect, we allow transition probabilities to depend on an exogenous variable through a **Time-Varying Transition Probability (TVTP)** specification.

Let $x_t \in \mathbb{R}$ denote an observed exogenous signal. In this project, $x_t$ represents a **CEO uncertainty score** extracted from managerial speech by the . Transition probabilities are specified as:

$$
\Pr(s_{t+1} = i \mid s_t = j, x_t) = \frac{\exp\!\left(\alpha_{ij} + \beta_{ij} x_t \right)}{\sum_{k \in \{0,1\}} \exp\!\left(\alpha_{kj} + \beta_{kj} x_t \right)}, \quad i,j \in \{0,1\}
$$

This multinomial logit formulation ensures that transition probabilities are well-defined and sum to one for each current regime $j$.

**Economic interpretation.**  
CEO uncertainty is interpreted as a measure of forward-looking ambiguity and information quality, rather than a direct predictor of returns. Consequently, it enters **only the transition dynamics** and not the return equation. Higher uncertainty is expected to reduce regime persistence and increase the probability of regime switches, while lower uncertainty is associated with greater regime stability.

---

### 4. Regime Identification: Bull vs Bear

After estimation, regimes are labeled as **Bull** or **Bear** based on a composite economic criterion that captures both expected performance and stability:

$$
\text{score}_j = w_1 \mu_j - w_2 \sigma_j + w_3 \Pr(s_t = j \mid \text{data}) + w_4 \text{Sharpe}_j, \quad j \in \{0,1\}
$$

where $\text{Sharpe}_j = \mu_j / \sigma_j$.  
The regime with the higher score is classified as the **Bull** market, while the other is classified as **Bear**.

This labeling step is used for interpretation and visualization; regime dynamics themselves are learned endogenously from the data.

---

### 5. Alternative Baseline: Long Short-Term Memory (LSTM)

To compare the performance of the MSAR framework, we implement a **Long Short-Term Memory (LSTM)** neural network as an alternative benchmark model. Unlike MSAR, which explicitly models discrete regime transitions, LSTM captures temporal dependencies through a continuous recurrent architecture.

**Model Architecture:**

The LSTM model consists of:
- **Input layer**: Historical returns over a lookback window (shape: $L \times F$, where $L$ is lookback days and $F$ is number of features/stocks)
- **LSTM layer**: 50 hidden units to capture temporal patterns and long-term dependencies
- **Output layer**: Dense layer with 1 unit predicting next-day return for each stock

**Training Strategy:**

$$
\mathbf{X}^{(i)} = [r_{t-L}, r_{t-L+1}, \ldots, r_{t-1}], \quad y^{(i)} = r_t
$$

- **Sliding windows**: Create training samples by sliding a fixed-length window across the time series
- **Multi-input, single-output**: Each stock is modeled separately with input from all stocks to capture cross-stock relationships
- **Loss function**: Huber loss ($\delta = 0.01$) for robustness to outliers (e.g., flash crashes, extreme returns)
- **Validation**: 20% split for monitoring overfitting during training

**Incorporating CEO Speech Features:**

Similar to MSAR's TVTP extension, LSTM can incorporate CEO linguistic variables as additional input features. When CEO uncertainty $x_t$ is included, the input dimension expands from $F$ to $F+1$:

$$
\mathbf{X}^{(i)}_{\text{CEO}} = [r_{t-L}, \ldots, r_{t-1}, x_{t-L}, \ldots, x_{t-1}]
$$

The LSTM learns to weight both historical returns and CEO uncertainty jointly when predicting future returns.

MSAR is better suited to low-volatility regimes characterized by stable and persistent market states, whereas LSTM is designed to capture nonlinear dynamics prevalent in high-volatility environments. This motivates a volatility-based model selection mechanism, in which MSAR is applied during low-volatility periods and LSTM is activated when return volatility exceeds a specified threshold.

## Contribution

| Contributor | Task Description |
|-------------|------------------|
| Haowei Zhang | Design the CEO speech modeling framework and quantify uncertainty for subsequent models |
| Di Xiao | Preprocess the data and conduct numerical experiments |
| Yuekai Yan | Train and refine the MSAR model |
| Thibaut Peiffer | Train and refine the LSTM model |
| Luyao Tang | Develop the project website and design the data-driven narrative |




