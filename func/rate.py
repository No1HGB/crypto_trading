import pyfredapi as pf
import pandas as pd
import os
from dotenv import load_dotenv

# 현재 스크립트의 디렉토리 경로를 찾습니다
script_dir = os.path.dirname(os.path.abspath(__file__))

# 스크립트 디렉토리의 상위 디렉토리에서 .env 파일을 찾습니다
env_path = os.path.join(script_dir, "../.env")

# 환경 변수를 로드합니다
load_dotenv(dotenv_path=env_path)


def append_rate_data(symbol):
    data = pf.get_series(series_id="DFF", api_key=os.getenv("FRED_API_KEY"))
    data.to_csv("../data/rate.csv", index=False)
    df = pd.read_csv(f"../data/{symbol.lower()}.csv", parse_dates=["Open_time"])
    new_data_df = pd.read_csv("../data/rate.csv", parse_dates=["date"], dayfirst=True)
    df["Date"] = df["Open_time"].dt.date
    new_data_df["date"] = new_data_df["date"].dt.date
    value_series = new_data_df.set_index("date")["value"]
    df["rate"] = df["Date"].map(value_series)
    df.drop(columns=["Date"], inplace=True)
    return df
