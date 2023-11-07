import pyfredapi as pf
import pandas as pd
import os
from dotenv import load_dotenv
from datetime import datetime

# 현재 스크립트의 디렉토리 경로를 찾습니다
script_dir = os.path.dirname(os.path.abspath(__file__))

# 스크립트 디렉토리의 상위 디렉토리에서 .env 파일을 찾습니다
env_path = os.path.join(script_dir, "../.env")

# 환경 변수를 로드합니다
load_dotenv(dotenv_path=env_path)


def get_rate_data(start_date: str, end_date: str) -> pd.DataFrame:
    # API에서 데이터를 가져옵니다.
    fred = pf.Fred(api_key=os.getenv("FRED_API_KEY"))
    data = fred.get_series_all_releases(series_id="DFF")
    # 필요한 컬럼만 선택합니다.
    data = data[["date", "value"]]
    # 'date' 컬럼을 datetime 타입으로 변환합니다.
    data["date"] = pd.to_datetime(data["date"])
    # start_date와 end_date를 datetime 객체로 변환합니다.
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")
    # start_date와 end_date 사이의 데이터만 필터링합니다.
    filtered_data = data[(data["date"] >= start_date) & (data["date"] <= end_date)]

    return filtered_data
