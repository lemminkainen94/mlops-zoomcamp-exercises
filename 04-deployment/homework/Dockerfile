FROM agrigorev/zoomcamp-model:mlops-3.9.7-slim

RUN pip install -U pip
RUN pip install pipenv 

WORKDIR /app

COPY [ "Pipfile", "Pipfile.lock", "./" ]
COPY [ "starter.py", "./" ]

RUN pipenv install --system --deploy
RUN pip install pandas pyarrow fastparquet

ENTRYPOINT [ "python", "starter.py", "2021", "4" ]