FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Source files
COPY server.py rebuilder.py 2_ingest.py 3_build_index.py ./

# V5 geocoder data (sparse_grid.bin + master_dict.json are tracked via Git LFS)
COPY v5_data/ ./v5_data/

# Pre-built V6 entity index (small files, no LFS needed)
COPY data/ ./data/

EXPOSE 7860

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
