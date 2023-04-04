from copy import deepcopy
import pandas as pd
from yahooquery import Ticker


def income_statement_history(symbols):
    """
    Download and preprocess income statements.
    """
    data = Ticker(symbols).all_modules

    out_df = pd.DataFrame()

    for s in symbols:
        try:
            temp_data = data[s]["incomeStatementHistoryQuarterly"][
                "incomeStatementHistory"
            ]
            if len(temp_data) == 0:
                continue
        except:
            continue

        temp_df = pd.DataFrame(temp_data)

        # Clean up dataframe
        temp_df["symbol"] = s
        temp_df["date"] = pd.to_datetime(temp_df["endDate"], format="%Y-%m-%d")
        temp_df = temp_df.filter(
            items=[
                "symbol",
                "date",
                "totalRevenue",
                "costOfRevenue",
                "grossProfit",
                "totalOperatingExpenses",
                "operatingIncome",
                "netIncome",
            ]
        )

        # Create period level features
        temp_df = temp_df.sort_values(by=["date"], ascending=False)
        temp_df["period"] = temp_df.apply(lambda row: (row.name + 1) * -1.0, axis=1)

        # Normalize features based on 'netIncome'
        features = [
            "totalRevenue",
            "costOfRevenue",
            "grossProfit",
            "totalOperatingExpenses",
            "operatingIncome",
        ]
        for feature in features:
            try:
                temp_df[feature] = temp_df.apply(
                    lambda row: (row[feature] / row["netIncome"]), axis=1
                )
            except:
                temp_df[feature] = np.nan

        # Create trend features
        for feature in features:
            key = f"{feature}_trend"
            temp_df[key] = temp_df[feature] - temp_df[feature].shift(-1)
        temp_df = temp_df.drop(columns=["netIncome"])

        out_df = out_df.append(temp_df, ignore_index=True)

    return out_df


def grading_history(symbols):
    """
    Download and preprocess grading history.
    """
    data = Ticker(symbols).grading_history
    data = data.reset_index(drop=True)

    # Clean up date features
    data["epochGradeDate"] = data["epochGradeDate"].apply(lambda x: x[0:10])
    data["date"] = pd.to_datetime(data["epochGradeDate"], format="%Y-%m-%d")

    # Create action feature
    actions = ["main", "reit", "down", "up", "init"]
    for action in actions:
        data[action] = data["action"].apply(lambda x: 1 if x == action else 0)
    data = data.filter(items=["symbol", "date", "main", "reit", "down", "up", "init"])

    data["period"] = np.nan

    return data


def earning_history(symbols):
    """
    Download and preprocess earning history.
    """
    out_df = Ticker(symbols).earning_history.reset_index().copy()

    bad_symbols = set(out_df[out_df["surprisePercent"] == {}]["symbol"])
    out_df = out_df[~out_df["symbol"].isin(bad_symbols)]

    out_df = out_df.rename(columns={"quarter": "date"})
    out_df["date"] = pd.to_datetime(out_df["date"], format="%Y-%m-%d")
    out_df["period"] = out_df.period.str.replace("q", "").astype(int)

    # Create new features
    out_df["surpriseBinary"] = (out_df["epsDifference"] > 0).astype(int)

    out_df = out_df.filter(
        items=[
            "symbol",
            "epsActual",
            "epsEstimate",
            "epsDifference",
            "surprisePercent",
            "date",
            "period",
            "surpriseBinary",
        ]
    )

    return out_df


class DataLoader:
    """
    Fetches companies' symbols and industries according to a given set of indices.

    This module supports 4 indices:
    - 'russell1000'
    - 'russell3000'
    - 'snp500'
    - 'allUSstocks'

    Parameters
    ----------
    indices : str, list, default=None
        List containing the indices from which to extract the companies' symbols and
        industry. If ``None``, loads all indices.

    Returns
    -------
    self
    """

    _supported_lists = ["russell1000", "russell3000", "snp500", "allUSstocks"]

    def __init__(self, indices=None):
        self.indices = indices

    def _check(self):
        if self.indices is None:
            self.indices_ = deepcopy(self._supported_lists)
        elif type(self.indices) == list:
            self.indices_ = deepcopy(self.indices)
        elif type(self.indices) == str:
            self.indices_ = [self.indices]

        if not all([index in self._supported_lists for index in self.indices_]):
            raise KeyError(f"Currently supported indices: {self._supported_lists}")

    def fetch_symbols(self):
        self._check()

        self.symbols_ = []
        for index in self.indices_:
            functionName = "self." + "load_" + index + "()"
            symbols = eval(functionName)
            self.symbols_.append(symbols)

    def load_test_set(self, symbols):
        df = self.load_russell3000()

        df = df[df["symbol"].isin(symbols)]

        return df

    def load_russell1000(self):
        source = pd.read_html("https://en.wikipedia.org/wiki/Russell_1000_Index")
        df = source[2].copy()
        df = df.rename(columns={"Ticker": "symbol", "GICS Sector": "sector"})
        df = df.filter(items=["symbol", "sector"])

        return df

    def load_russell3000(self):
        source = pd.read_html(
            "http://www.kibot.com/Historical_Data/"
            "Russell_3000_Historical_Intraday_Data.aspx"
        )
        df = source[1].copy()
        header = df.iloc[0]  # grab the first row for the header
        df = df[1:]  # take the data less the header row
        df.columns = header  # set the header row as the df header
        df = df.rename(
            columns={"Symbol": "symbol", "Sector": "sector", "Industry": "industry"}
        )
        df = df.filter(items=["symbol", "industry", "sector"])

        return df

    def load_snp500(self):
        source = pd.read_html(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        )
        df = source[0].copy()
        df = df.rename(columns={"Symbol": "symbol", "GICS Sector": "sector"})
        df = df.filter(items=["symbol", "sector"])

        return df

    def load_allUSstocks(self):
        source = pd.read_csv("us_symbols.csv")
        df = source.copy()

        return df


class TimeChopper:
    FinalDf = None

    def __init__(self, Symbols, EarningHistory, IncomeStatementHistory, GradingHistory):
        self.Symbols = Symbols
        self.EarningHistory = EarningHistory
        self.IncomeStatementHistory = IncomeStatementHistory
        self.GradingHistory = GradingHistory

    def createDataset(self, NumQuarters, Delay):
        out_df = pd.DataFrame()

        for symbol in set(self.Symbols["symbol"]):
            r = {}

            # Add symbol
            r["symbol"] = symbol

            # Add sector
            sector_name = self.Symbols[self.Symbols["symbol"] == symbol]["sector"].iloc[
                0
            ]
            r["sector"] = sector_name

            # EarningHistory
            EarningHistory = self.EarningHistory[
                self.EarningHistory["symbol"] == symbol
            ].copy()
            EarningHistory = EarningHistory.sort_values(by=["period"], ascending=False)

            # Create target
            if Delay != -1:
                try:
                    r["target"] = EarningHistory["surpriseBinary"].iloc[Delay]
                except:
                    continue

                # Identify the 'as_of_date' and 'max_history_date'
                as_of_date = EarningHistory["date"].iloc[Delay]
                as_of_period = EarningHistory["period"].iloc[Delay]
                key = NumQuarters + Delay
                max_history_date = EarningHistory["date"].iloc[key]
                max_history_period = EarningHistory["period"].iloc[key]
            else:
                as_of_period = 0
                try:
                    max_history_period = EarningHistory["period"].iloc[
                        (NumQuarters - 1)
                    ]
                    as_of_date = pd.to_datetime("today").date()
                    max_history_date = as_of_date - DateOffset(months=(3 * NumQuarters))
                except:
                    continue

            # EarningHistory is filtered by period
            EarningHistory = EarningHistory[
                (EarningHistory["period"] < as_of_period)
                & (EarningHistory["period"] >= max_history_period)
            ]
            EarningHistoryFeatures = [
                col
                for col in EarningHistory.columns
                if col not in ["symbol", "period", "date"]
            ]

            # Period features
            for period in range(0, EarningHistory.shape[0]):
                for feature in EarningHistoryFeatures:
                    r[f"company_period_{period}_{feature}"] = EarningHistory.iloc[
                        period
                    ][feature]

            # Company level features
            for feature in EarningHistoryFeatures:
                r[f"company_mean_{feature}"] = EarningHistory[feature].mean()
                r[f"company_median_{feature}"] = EarningHistory[feature].median()
                r[f"company_max_{feature}"] = EarningHistory[feature].max()
                r[f"company_min_{feature}"] = EarningHistory[feature].min()

            # IncomeStatementHistory is fileted by period
            IncomeStatementHistory = self.IncomeStatementHistory[
                self.IncomeStatementHistory["symbol"] == symbol
            ].copy()
            IncomeStatementHistory = IncomeStatementHistory.sort_values(
                by=["period"], ascending=False
            )
            IncomeStatementHistory = IncomeStatementHistory[
                (IncomeStatementHistory["period"] < as_of_period)
                & (IncomeStatementHistory["period"] >= max_history_period)
            ]
            IncomeStatementHistoryFeatures = [
                col
                for col in IncomeStatementHistory.columns
                if col not in ["symbol", "period", "date"]
            ]

            for period in range(0, IncomeStatementHistory.shape[0]):
                for feature in IncomeStatementHistoryFeatures:
                    r[
                        f"company_period_{period}_{feature}"
                    ] = IncomeStatementHistory.iloc[period][feature]

            # Company level features
            for feature in IncomeStatementHistoryFeatures:
                r[f"company_mean_{feature}"] = IncomeStatementHistory[feature].mean()
                r[f"company_median_{feature}"] = IncomeStatementHistory[
                    feature
                ].median()
                r[f"company_max_{feature}"] = IncomeStatementHistory[feature].max()
                r[f"company_min_{feature}"] = IncomeStatementHistory[feature].min()

            # Grading history
            GradingHistory = self.GradingHistory[
                self.GradingHistory["symbol"] == symbol
            ].copy()
            GradingHistory = GradingHistory.sort_values(by=["period"], ascending=False)
            GradingHistory = GradingHistory[
                (GradingHistory["date"] < np.datetime64(as_of_date))
                & (GradingHistory["date"] >= np.datetime64(max_history_date))
            ]

            GradingHistory = (
                GradingHistory.groupby("symbol").sum().reset_index(drop=True)
            )

            GradingActions = ["main", "reit", "down", "up", "init"]
            for action in GradingActions:
                try:
                    r[f"action_{action}_sum"] = GradingHistory[action].iloc[0]
                except:
                    r[f"action_{action}_sum"] = 0

            out_df = out_df.append(r, ignore_index=True)

        # Fill sector one hot values
        # sectorCols = [col for col in out_df if col.startswith('in_sector_')]
        # for col in sectorCols:
        #    out_df[col] = out_df[col].fillna(0)

        sector_out = pd.DataFrame()

        for sector in set(self.Symbols["sector"]):
            r = {}
            r["sector"] = sector

            SectorDf = out_df[out_df["sector"] == sector].copy()
            SectorDfFeatures = [
                col for col in SectorDf if col not in ["symbol", "sector", "target"]
            ]
            for feature in SectorDfFeatures:
                r[f"sector_mean_{feature}"] = SectorDf[feature].mean()
                r[f"sector_median_{feature}"] = SectorDf[feature].median()
                r[f"sector_max_{feature}"] = SectorDf[feature].max()
                r[f"sector_min_{feature}"] = SectorDf[feature].min()

            sector_out = sector_out.append(r, ignore_index=True)

        SectorOneHot = pd.get_dummies(sector_out["sector"], drop_first=False)
        sector_out = pd.merge(
            sector_out,
            SectorOneHot.add_prefix("in_sector_"),
            how="left",
            left_index=True,
            right_index=True,
        )

        final_out = out_df.merge(sector_out, left_on="sector", right_on="sector")
        final_out = final_out.drop(columns=["sector"])

        self.FinalDf = final_out

        return self.FinalDf
