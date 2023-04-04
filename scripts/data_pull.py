from esurprise.data import (
    DataLoader,
    earning_history,
    grading_history,
    income_statement_history,
)

N_JOBS = -1

dl = DataLoader("russell3000")
symbols = dl.fetch_symbols()
earn_hist = earning_history(symbols.symbol, n_jobs=N_JOBS)
inc_hist = income_statement_history(symbols.symbol, n_jobs=N_JOBS)
grad_hist = grading_history(symbols.symbol, n_jobs=N_JOBS)
