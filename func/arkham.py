import os, aiohttp, asyncio
from datetime import datetime
import pandas as pd
import pickle
from dotenv import load_dotenv

# 현재 스크립트의 디렉토리 경로를 찾습니다
script_dir = os.path.dirname(os.path.abspath(__file__))

# 스크립트 디렉토리의 상위 디렉토리에서 .env 파일을 찾습니다
env_path = os.path.join(script_dir, "../.env")

# 환경 변수를 로드합니다
load_dotenv(dotenv_path=env_path)


# 비동기 처리
async def fetch(session, url, params, headers):
    retries = 3
    for retry in range(retries):
        try:
            async with session.get(url, params=params, headers=headers) as response:
                response.raise_for_status()
                return await response.json()
        except aiohttp.ClientPayloadError:
            if retry < retries - 1:
                print("Payload error, retrying...")
                await asyncio.sleep(1)
            else:
                print("Payload error, response payload was not completed")
                return {"error": "payload error"}
        except aiohttp.ClientResponseError as e:
            print(f"HTTP error occurred: {e}")
            return {"error": str(e)}
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return {"error": str(e)}


async def get_arkham_data(
    start_timestamp, end_timestamp, from_p="", to_p=""
) -> pd.DataFrame:
    API_URL_ARKHAM = "https://api.arkhamintelligence.com/transfers"
    interval_timestamp = 15 * 60 * 1000

    # 새로운 DataFrame 생성
    columns = []
    if from_p:
        columns.append(f"from_{from_p}")
    elif to_p:
        columns.append(f"to_{to_p}")
    df = pd.DataFrame(columns=columns)

    async with aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(ssl=False)
    ) as session:
        tasks = []
        for i in range(start_timestamp, end_timestamp + 1, interval_timestamp):
            date_index = datetime.fromtimestamp(i / 1000).strftime("%Y.%-m.%-d %H:%M")

            base_param = {
                "base": "binance,coinbase",
                "chains": "bitcoin",
                "tokens": "bitcoin",
                "sortDir": "asc",
                "valueGte": 5,
                "timeGte": i,
                "timeLte": i + interval_timestamp,
            }
            from_param = {"from": from_p}
            to_param = {"to": to_p}

            parameters = {**base_param}
            if from_p:
                parameters.update(from_param)
            elif to_p:
                parameters.update(to_param)

            headers = {"API-Key": os.getenv("ARKHAM_API_KEY")}

            tasks.append(fetch(session, API_URL_ARKHAM, parameters, headers))

        batch_size = 7
        for i in range(0, len(tasks), batch_size):
            results = await asyncio.gather(*tasks[i : i + batch_size])
            await asyncio.sleep(1)

            for j, response in enumerate(results):
                timestamp = start_timestamp + (i + j) * interval_timestamp
                date_index = datetime.fromtimestamp(timestamp / 1000).strftime(
                    "%Y.%-m.%-d %H:%M"
                )

                if "transfers" in response:
                    transfers = response["transfers"]
                    total_unit_value = sum(
                        transfer["unitValue"] for transfer in transfers
                    )
                    df.at[date_index, columns[0]] = total_unit_value
                    print(f"{columns[0]} {total_unit_value} 추가 완료")
                else:
                    print(response)
                    df.at[date_index, columns[0]] = 0
                    print(f"No data for timestamp: {date_index}")

    return df
