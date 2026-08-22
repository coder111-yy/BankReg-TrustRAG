FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV BANKREG_ARTIFACT_DIR=/app/artifacts
ENV BANKREG_DB=/app/artifacts/bankreg.sqlite3
EXPOSE 8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]

