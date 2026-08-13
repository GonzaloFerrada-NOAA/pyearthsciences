from datetime import datetime


def msg(text: str) -> None:
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S}    {text}")
