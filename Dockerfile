FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY . .

EXPOSE 7860 7861 7862 8000

CMD ["sh", "-c", "\
    uv run python data/manager.py & \
    uv run python central/main.py & \
    sleep 3 && \
    uv run python client/main.py --port 7861 --client-id client_0 & \
    uv run python client/main.py --port 7863 --client-id client_1 & \
    wait"]
