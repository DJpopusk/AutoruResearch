# Presentation Notebooks — структура исследования для защиты

Каждый ноутбук — один смысловой блок. Запускать строго в порядке нумерации:
блоки 2-8 используют артефакты, сохранённые предыдущими (`selected_features.json`,
`*_models.pkl`).

| # | Ноутбук | Что внутри | Ключевые артефакты |
|---|---|---|---|
| 01 | `01_feature_analysis.ipynb` | Описание 23 числовых признаков (ед.изм., fill %, mean/std), корреляции с `ln(price)`, частные корреляции, матрицы по 5 группам, межгрупповая `|corr|`, PCA-биплот, scree, communality | — |
| 02 | `02_feature_selection.ipynb` | Иерархическая кластеризация по `1−|corr|`, champion-логика (`corr_y + communality`), KEEP/DROP таблица с причинами | `selected_features.json` |
| 03 | `03_baseline_models.ipynb` | OLS на Full (23) / Best (18 KEEP) / Min (backward); сравнение R², AIC, BIC, VIF, значимости | `baseline_models.pkl` |
| 04 | `04_diagnostics_baseline.ipynb` | Диагностика остатков M_min: стандартизованные остатки, QQ-plot, гистограмма, e vs ŷ, e vs предикторы. Тесты с **H₀, статистикой, p-value, вердиктом**: Jarque–Bera, Shapiro, Breusch–Pagan, White, Goldfeld–Quandt, Durbin–Watson, RESET | — |
| 05 | `05_feature_engineering.ipynb` | Ratio (`hp_per_kg`, `hp_per_liter`, `ln_weight_curb`) + центрирование (`age_c, mileage_c`) + нелинейности (`age_c²`, `mileage_c²`, `engine_x_age`); backward; сравнение диагностик до/после FE; новый VIF | `fe_model.pkl` |
| 06 | `06_wls.ipynb` | Двухшаговая FGLS: оценка `σ²(x)` через `ln(e²) ~ X`, веса `1/σ̂²`, WLS. Распределение весов; сравнение коэффициентов и SE; графики OLS vs WLS | `wls_model.pkl` |
| 07 | `07_segmentation.ipynb` | Сегментация по `vehicle_state` (на ходу / битые) и по возрасту (≤7 / >7 лет); per-сегмент BP, R²adj; сравнение коэффициентов | `seg_models.pkl` |
| 08 | `08_final_summary.ipynb` | Итоговая таблица всех тестов с H₀ и p-value по всем 4 моделям; финальные коэффициенты и интерпретация в %; проверка критериев успеха; диагностическая сводка финальной M_wls | — |

## Поддержка

- `helpers.py` — общий код (загрузка, импьют, стиль графиков, дизайн-матрица, FGLS, форматирование тестов). Все ноутбуки начинают с `from helpers import ...`.
- Стиль графиков по требованиям защиты (PDF): белый фон, чёрный текст, заголовки, подписи осей с единицами, сетка, легенда.

## Как запустить

```bash
cd presentation_notebooks
jupyter notebook
```
или
```bash
jupyter nbconvert --to notebook --execute 01_feature_analysis.ipynb
```
