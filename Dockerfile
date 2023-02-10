FROM python:3.10

WORKDIR /workspace

COPY requirements.txt app /workspace/

RUN pip install --no-cache-dir --upgrade -r /workspace/requirements.txt

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]