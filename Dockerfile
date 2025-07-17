### ---- FRONTEND STAGE ----
FROM node:20 AS frontend
WORKDIR /app/inference-ui
COPY inference-ui/ ./
RUN npm install && npm run build

### ---- BACKEND STAGE ----
FROM ubuntu:22.04 AS backend

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl bzip2 ca-certificates \
 && curl -fsSL -o /tmp/miniforge.sh \
    "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh" \
 && bash /tmp/miniforge.sh -b -p /opt/conda \
 && rm /tmp/miniforge.sh && /opt/conda/bin/conda clean -afy

ENV PATH="/opt/conda/bin:$PATH"

# Copy Conda environment file
WORKDIR /app
COPY environment.yml .

# Create conda env
RUN mamba env create -f environment.yml && \
    mamba clean --all -y

# Use conda environment by default
SHELL ["conda", "run", "-n", "thyroid-classifier", "/bin/bash", "-lc"]

# Copy backend source and frontend build
COPY . .
COPY --from=frontend /app/inference-ui/dist ./frontend

# Make startup script executable
RUN chmod +x start.sh

# Expose port & use tini
EXPOSE 8000
# ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["bash", "start.sh"]
