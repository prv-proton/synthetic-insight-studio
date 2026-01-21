FROM python:3.11-slim

WORKDIR /app
COPY ui/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY ui/streamlit_app.py /app/streamlit_app.py

EXPOSE 8501

CMD ["streamlit", "run", "/app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
