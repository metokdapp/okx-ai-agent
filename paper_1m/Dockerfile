FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 DATA_DIR=/data
WORKDIR /app
COPY agent.py ./agent.py
RUN mkdir -p /data
CMD ["python", "agent.py"]
