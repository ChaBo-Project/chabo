#CHATFED_ORCHESTRATOR

FROM python:3.11.11

# ---------- Create Non-Root User ----------
# Ensures proper file permissions for dev and runtime
RUN useradd -m -u 1000 user

WORKDIR /app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---------- Copy Project Files ----------
# Set appropriate ownership and permissions
COPY --link --chown=1000 . .

# fastapi and gradio
EXPOSE 7860 

# launch with unbuffered output
CMD ["python", "-u", "app/main.py"]