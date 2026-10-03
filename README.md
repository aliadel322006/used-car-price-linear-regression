# Used car price prediction with linear regression from scratch

In this project I explore a used car sales dataset and then predict the selling price with linear regression. I wrote the model myself in NumPy, trained it with Batch Gradient Descent and with Stochastic Gradient Descent (SGD), and compared it with scikit-learn.

## Dataset

The cleaned data has 558,723 rows and 28 columns. Each row is one used car sale, and the target is `SellingPrice`. It is stored in `vehicle_sales_clean.zip`, and the notebook reads it directly with `pd.read_csv`.

For the model I use `CarAge`, `Odometer`, `ConditionValue`, `Make`, `Model`, `Body`, `Transmission`, `State`, `Color` and `Interior`. I also added two centered squared columns, `CarAge2` and `Odometer2`.

## What I did

1. `EDA_CARSPRIVE.ipynb`: data quality checks, cleaning, feature engineering, charts and KPIs. It loads the raw data from a local SQL Server database, so you can't re-run it without that database. The saved outputs are still readable on GitHub.
2. `ml.ipynb`: linear regression from scratch with Batch GD and SGD, compared with scikit-learn's `LinearRegression` and `SGDRegressor`. Before training I grouped rare categories into "other", one-hot encoded the text columns, and scaled all the columns.
3. `used car sales dashboard.pbix.pbix`: a Power BI dashboard.

## Results

R² on the test set:

| Model | Training method | R² |
|---|---|---|
| My `linearregression` (lr=0.3, 2000 iterations) | Batch GD | 0.7906 |
| sklearn `LinearRegression` | Closed form | 0.7907 |
| My `sgd` (lr=0.00001, 5 epochs) | SGD | 0.7900 |
| sklearn `SGDRegressor` (eta0=0.00001, constant learning rate, 5 epochs) | SGD | 0.7886 |

## What I learned

- My Batch GD ends very close to sklearn's `LinearRegression` (0.7906 vs 0.7907). My first try scored only 0.6567, and it got much better after I scaled all the columns and centered the squared features.
- SGD needed only 5 epochs, while Batch GD needed about 2000 iterations.
- SGD blew up at first. Scaled rare one-hot columns have large values for the few rows that have them, and with a big learning rate one of those rows makes a huge step. A smaller learning rate fixed my SGD. For sklearn I also had to use a constant learning rate.
- Linear regression can't capture curves and interactions. A quick tree-based model test in an earlier experiment (not shown in the notebook) reached 0.93, so I want to study tree models next.

## Sample charts

![Distributions](photos/output.png)
![Price vs odometer and car age](photos/output1.png)
![Top 10 models by sales volume](photos/output5.png)

## How to run

```bash
pip install -r requirements.txt
jupyter notebook ml.ipynb
```

Run `ml.ipynb` from this folder so it can find `vehicle_sales_clean.zip`. To run `EDA_CARSPRIVE.ipynb` you also need `pyodbc` and a local SQL Server with the `MotoDB.VehicleSales` table.

## Requirements

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- jupyter
- pyodbc (only for `EDA_CARSPRIVE.ipynb`)

## Files

| File | Description |
|---|---|
| `ml.ipynb` | Linear regression from scratch vs scikit-learn |
| `EDA_CARSPRIVE.ipynb` | Data cleaning and exploratory analysis |
| `vehicle_sales_clean.zip` | Cleaned dataset (compressed CSV) |
| `used car sales dashboard.pbix.pbix` | Power BI dashboard |
| `photos/` | Chart images |
