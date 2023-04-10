from datetime import datetime
from esurprise.data import (
    DataLoader,
    TimeChopper,
    earning_history,
    grading_history,
    income_statement_history,
)

N_JOBS = -1
DATE = datetime.now().strftime('%Y%b%d').lower()

dl = DataLoader("russell3000")
symbols = dl.fetch_symbols()
earn_hist = earning_history(symbols.symbol, n_jobs=N_JOBS)
inc_hist = income_statement_history(symbols.symbol, n_jobs=N_JOBS)
grad_hist = grading_history(symbols.symbol, n_jobs=N_JOBS)

earn_hist.to_csv(f"matrices/_raw/earning_history_{DATE}.csv", index=False)
inc_hist.to_csv(f"matrices/_raw/incone_statement_history_{DATE}.csv", index=False)
grad_hist.to_csv(f"matrices/_raw/grading_history_{DATE}.csv", index=False)

tc = TimeChopper(
    Symbols=symbols,
    EarningHistory=earn_hist,
    IncomeStatementHistory=inc_hist,
    GradingHistory=grad_hist,
)

df_prev = tc.createDataset(NumQuarters=2, Delay=1)
df_curr = tc.createDataset(NumQuarters=2, Delay=0)
df_fut = tc.createDataset(NumQuarters=2, Delay=-1)

y_train = df_prev['target'].copy()
X_train = df_prev.fillna(0).drop(columns="target").copy()

y_test = df_curr['target'].copy()
X_test = df_curr.fillna(0).drop(columns="target").copy()
X_future = df_fut.fillna(0)

X_train.to_csv(f'matrices/X_train_{DATE}.csv', index=False)
y_train.to_csv(f'matrices/y_train_{DATE}.csv', index=False)
X_test.to_csv(f'matrices/X_test_{DATE}.csv', index=False)
y_test.to_csv(f'matrices/y_test_{DATE}.csv', index=False)
X_future.to_csv(f'matrices/X_future_{DATE}.csv', index=False)
