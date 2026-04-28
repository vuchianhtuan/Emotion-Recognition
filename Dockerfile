FROM python:3.13-slim
WORKDIR /app

# Cài thư viện hệ thống cần thiết cho vẽ biểu đồ/xử lý ảnh
RUN apt-get update && apt-get install -y libgl1 && rm -rf /var/lib/apt/lists/*

# Cài đặt thư viện Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501
CMD ["streamlit", "run", "app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]