# Dockerfile para ambiente de desenvolvimento (Conda)
FROM continuumio/miniconda3:latest

LABEL maintainer="Equipe RoboCIn <contato@robocin.example>"

WORKDIR /workspace

# Copia environment para criar o ambiente conda
COPY environment.yml /workspace/environment.yml

# Cria o ambiente conda (nome: heartenv)
RUN conda env create -f /workspace/environment.yml -n heartenv && \
    conda clean -afy

# Use o shell do conda para futuros RUNs
SHELL ["conda", "run", "-n", "heartenv", "/bin/bash", "-lc"]

# Copia o código fonte (assuma que haverá uma pasta src/ no mesmo nível do Dockerfile)
COPY src/ /workspace/src/

# Cria pastas de trabalho
RUN mkdir -p /workspace/data /workspace/annotations /workspace/models /workspace/logs

# Porta (se for expor algum serviço web depois)
EXPOSE 5000

# Entrada padrão: roda o script de ingestão (pode ser sobrescrito ao iniciar o container)
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "heartenv", "python", "src/dicom_ingest.py"]