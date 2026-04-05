FROM python:3.11-slim

WORKDIR /app

# Install system dependencies required by trimesh / scipy / shapely
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy source and data (excluding cache)
COPY src/ ./src/
COPY data/ ./data/

# Remove cache dir if it was copied in (belt-and-suspenders)
RUN rm -rf ./data/cache

# Install Python dependencies for CPU-only mode
# Warp is NOT installed — backend="cpu" is the default and only mode here
RUN pip install --no-cache-dir \
        fastapi==0.115.0 \
        uvicorn[standard]==0.30.0 \
        pydantic==2.7.0 \
        trimesh==4.11.3 \
        scipy==1.17.1 \
        numpy==2.4.3 \
        ezdxf==1.4.3 \
        shapely==2.1.2

EXPOSE 8000

# PYTHONPATH ensures `from pipeline import ...` resolves correctly inside src/
ENV PYTHONPATH=/app/src

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
