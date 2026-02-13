FROM python:3.11.11

RUN useradd -m -u 1000 user

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --link --chown=1000 . .

EXPOSE 7860

CMD ["python", "-u", "src/main.py"]
