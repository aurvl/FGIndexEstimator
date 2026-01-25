# CNN Fear & Greed Estimator
---
`Finance, Machine Learning, Sentiment Analysis, Statistical Modeling`

Ce projet vise à **reconstruire un indice de type _CNN Fear & Greed_** à partir de données publiques (Yahoo Finance, FRED) en combinant plusieurs indicateurs de marché (actions, volatilité, obligations, crédit).  
L’objectif est double :

1. **Recoller au plus près de l’indice CNN Fear & Greed officiel** (approche « benchmark-driven »).  
2. **Proposer un indice théorique transparent et interprétable**, pouvant être utilisé dans des analyses quantitatives ou des stratégies de trading.

L’ensemble du pipeline est implémenté en Python (pandas, scikit-learn, yfinance) et documenté dans un notebook principal.

<img src="img/panic.png" alt="Illustration" width="100%">

---

## Project Structure

```
FGIndexEstimator/
├── data/                   # Données brutes et prétraitées
├── deploy/                 # Interfaces de déploiement (API, dashboard)
├── img/                    # Images pour le README et le notebook
├── notebooks.ipynb         # Notebook principal d’analyse et d’estimation
├── requirements.txt        # Dépendances Python
└── README.md               # Documentation du projet
```
---
## Data

- **Yahoo Finance (`yfinance`)**
  - `^GSPC` – S&P 500 (niveau de marché actions US).
  - `^VIX` – indice de volatilité implicite sur options S&P 500.
  - `RSP` – ETF S&P 500 equal-weight (proxy de breadth / participation).
  - `TLT` – ETF Treasuries long terme (proxy d’actif refuge).
  - `HYG` – ETF high yield US (proxy d’appétit pour le crédit risqué).

- **FRED (St. Louis Fed)**
  - `BAMLH0A0HYM2` – ICE BofA US High Yield Option-Adjusted Spread (écart de rendement high-yield vs investment grade).
  - `PUTCALL` – ratio put/call CBOE (si disponible pour l’API utilisée).

- **Référence CNN**
  - Série historique de l’indice **CNN Fear & Greed** (score 0–100) téléchargée depuis le site CNN Business, utilisée comme **benchmark** pour la calibration.

Toutes les séries sont alignées sur un **calendrier de jours ouvrés US**. Les jours fériés et valeurs manquantes sont gérés par **forward-fill** sur les prix/liquidités, ce qui permet de calculer sans rupture les moyennes mobiles et rendements.

---

## Methodology

### 1. Construction des indicateurs bruts

À partir des prix et spreads, le script `build_raw_indicators` calcule plusieurs composantes brutes :

1. **Market momentum – `momentum_spx`**  
   Distance relative du S&P 500 à sa moyenne mobile 125 jours :
   $$
   momentum\_spx_t
   = \frac{SPX_t - MA_{125}(SPX)_t}{MA_{125}(SPX)_t}.
   $$
   Valeur élevée ⇒ marché au-dessus de sa tendance ⇒ **Greed**.

2. **Trend / strength proxy – `strength_proxy`**  
   Distance relative à la moyenne mobile 200 jours :  
   $$
   strength\_proxy_t
   = \frac{SPX_t - MA_{200}(SPX)_t}{MA_{200}(SPX)_t}.
   $$
   Capte la **force du trend de long terme** (plus proche du composant « 52-week highs vs lows » de CNN).

3. **Stock price breadth – `breadth_rsp_spx`**  
   Sur 60 jours, on compare la performance de `RSP` (equal-weight) et du S&P 500 cap-weight :  
   $$
   breadth\_rsp\_spx_t
   = r_{60}^{RSP}(t) - r_{60}^{SPX}(t).
   $$  
   Quand l’equal-weight surperforme, la hausse est **plus largement partagée** ⇒ signal de **Greed**.

4. **Safe haven demand – `safe_haven_20d`**  
   Différence de rendement 20 jours actions vs Treasuries :  
   $$
   safe\_haven_{20d,t}
   = r_{20}^{SPX}(t) - r_{20}^{TLT}(t).
   $$  
   - Valeur positive : actions > obligations ⇒ fuite des actifs refuges ⇒ **Greed**.  
   - Valeur négative : Treasuries surperforment ⇒ **flight to safety / Fear**.

5. **Junk bond demand – `junk_bond_mom_20d` & `hy_spread`**  
   - **`hy_spread`** : écart de rendement high-yield vs IG (FRED). Spread élevé ⇒ aversion au risque ⇒ composante **inversée** lors du scoring.  
   - **`junk_bond_mom_20d`** : rendement 20 jours de `HYG`. Momentum positif ⇒ forte demande pour le crédit risqué ⇒ **Greed**.

6. **Market volatility – `vix_rel`**  
   Volatilité relative :  
   $$
   vix\_rel_t
   = \frac{VIX_t - MA_{50}(VIX)_t}{MA_{50}(VIX)_t}.
   $$  
   Volatilité au-dessus de sa tendance ⇒ **Fear**, au-dessous ⇒ **Greed** (composante inversée au scoring).

7. **Put/Call ratio – `put_call` (optionnel)**  
   Série FRED `PUTCALL` si disponible. Ratio élevé ⇒ couverture accrue / nervosité ⇒ **Fear** (score inversé).


### 2. Transformation en scores 0–100

Chaque indicateur brut $X_t$ est converti en **score de sentiment** $S_t \in [0,100]$ via une fonction `percentile_score` :

1. Pour chaque date $t$, on considère l’historique disponible $\{X_\tau\}_{\tau \le t}$ (avec un minimum de `min_periods`, typiquement 252 jours).
2. On calcule le **rang percentile** du point courant dans cet historique.
3. On mappe ce rang sur [0,100] :
   $$
   S_t = 100 \times \text{rank\_pct}(X_t).
   $$
4. Pour les variables de « peur » (VIX, `hy_spread`, put/call), on inverse le score :
   $$
   S^{inv}_t = 100 - S_t,
   $$
   de sorte qu’un score élevé signifie toujours **Greed**, et un score faible **Fear**.

Cette approche par percentiles est :

- **Sans dimension** : les composantes deviennent directement comparables, quelle que soit l’unité d’origine.  
- **Adaptative** : les extrêmes sont définis relativement à l’histoire récente, ce qui évite de figer des seuils arbitraires.

<img src="./img/all_scores.png" alt="Illustration" width="50%">
<img src="./img/heatmap_corr.png" alt="Illustration" width="50%">

### 3. Construction de l’indice composite

La fonction `compute_fear_greed` agrège les scores composantes en un indice global :

- Version **théorique** : moyenne simple de toutes les composantes disponibles  
  $$
  FG\_t^{(mean)} = \frac{1}{K} \sum_{k=1}^K S_{k,t}.
  $$

- Version **calibrée** : combinaison linéaire des scores estimée par régression OLS sur l’indice CNN :  
  $$
  FG\_t^{(cal)} = \alpha + \sum_{k} w_k S_{k,t},
  $$
  où $\alpha$ et $w_k$ sont ajustés pour minimiser l’erreur quadratique sur la période où l’indice CNN est observé.  
  Les poids obtenus mettent fortement l’accent sur la composante **safe haven**, le **momentum**, la **volatilité** et le **spread high-yield**, ce qui est économiquement cohérent.

---

## Results

<img src="./img/results1.png" alt="Illustration" width="100%">
<img src="./img/results2.png" alt="Illustration" width="100%">

### 1. Réplication naïve (moyenne simple)

La première version de l’indice utilise une moyenne simple des scores de composantes, sans calibration supervisée.

Sur la période commune avec l’indice CNN :

- $R^2 \approx 0.50$  
- RMSE ≈ 14 points (sur une échelle 0–100)

Visuellement, cette estimation apparaît **trop lissée et trop élevée** : elle reste souvent autour de 60–70 quand l’indice CNN chute vers des zones de peur marquée. L’estimateur capture la tendance générale, mais **manque les drawdowns de sentiment** et montre un **biais haussier**.


### 2. Indice calibré par régression linéaire

Une seconde version utilise les mêmes composantes mais ajuste les poids et l’intercept via une **régression linéaire (OLS)** sur les observations CNN récentes (≈ 250 jours).

Sur cet échantillon de calibration/validation :

- RMSE ≈ **6.3**  
- $R^2 \approx 0.90$  
- Nombre d’observations ≈ 250

Les trois modèles testés (Linear Regression, Ridge, Bayesian Ridge) donnent des performances extrêmement proches, ce qui confirme la **stabilité de la solution linéaire**.

| n° | Models on val            | RMSE | R2 | N obs |
|---|-------------------|---------:|--------:|-------:|
| 1 | Linear Regression      | 6.3104 | 0.8970 | 250 |
| 2 | Ridge Regression     | 6.3104 | 0.8970 | 250 |
| 3 | Bayesian Ridge Regression | 6.3117 | 0.8969 | 250 |

L’indice calibré suit désormais **au plus près les phases d’euphorie et de panique** observées par CNN, tout en restant analytique (combinaison linéaire explicite de signaux économiques).


### 3. Interprétation par régimes (classification)

<img src="./img/results3.png" alt="Illustration" width="100%">

Pour mieux interpréter les scores, on projette l’indice sur cinq catégories :  

> **Extreme Fear, Fear, Neutral, Greed, Extreme Greed** (via seuils sur $[0,100]$).

Sur la période de test de 251 jours, l’indice calibré atteint une **accuracy de 76 %** et un **F1-score pondéré de 0.76**.

> Les régimes d’**Extreme Fear** et de **Greed** sont particulièrement bien identifiés (F1 ≈ 0.87 et 0.80).  
La catégorie **Fear** reste la plus difficile à détecter (F1 ≈ 0.50, avec peu d’observations), ce qui reflète la difficulté à distinguer les zones de prudence modérée du simple « bruit » de marché.

|                | precision | recall | f1-score | support |
|----------------|-----------|--------|----------|---------|
| Extreme Fear   | 0.92      | 0.82   | 0.87     | 57      |
| Fear           | 0.75      | 0.38   | 0.50     | 8       |
| Neutral        | 0.70      | 0.81   | 0.75     | 57      |
| Greed          | 0.86      | 0.75   | 0.80     | 87      |
| Extreme Greed  | 0.54      | 0.69   | 0.60     | 42      |
|                |           |        |          |         |
| accuracy       |           |        | 0.76     | 251     |
| macro avg      | 0.75      | 0.69   | 0.70     | 251     |
| weighted avg   | 0.78      | 0.76   | 0.76     | 251     |

Globalement, l’indice reconstruit fournit donc une **approximation crédible de la dynamique de sentiment CNN**, tout en restant contrôlable et interprétable au niveau de chaque composante.

---

## How to run the project

1. **Cloner le dépôt**

   ```bash
   git clone https://github.com/aurvl/FGIndexEstimator.git
   cd FGIndexEstimator
   ```

2. **Créer l’environnement Python**

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # sous Windows : .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Configurer les clés et chemins**

   * Créer un fichier `.env` ou éditer `src/config.py` pour renseigner :

     * votre **API key FRED** (`API_KEY`),
     * les chemins vers les dossiers `data/` si besoin.

4. **Lancer le notebook**

   ```bash
   jupyter notebook notebooks/CNN_Fear_and_Greed_Estimator.ipynb
   ```

   Le notebook télécharge les données (ou recharge les caches), calcule les composantes, construit l’indice naïf, puis effectue la calibration OLS et les évaluations.

---

## Limitations & possible extensions

* Certaines composantes CNN (nombre de nouveaux plus hauts / plus bas NYSE, McClellan volume index, put/call exact) ne sont pas disponibles en open data. Elles sont donc **approximées par des proxies** (RSP vs SPX, HYG, spreads FRED, etc.).
* L’approche actuelle est volontairement **linéaire et transparente**. Des modèles non linéaires (Random Forest, Gradient Boosting, réseaux neuronaux) pourraient capturer des interactions plus complexes, au prix d’une interprétabilité moindre.
* De nouvelles sources (sentiment textuel, flux d’ETF sectoriels, order-flow) pourraient enrichir l’indice et améliorer la détection des phases de peur « intermédiaire ».