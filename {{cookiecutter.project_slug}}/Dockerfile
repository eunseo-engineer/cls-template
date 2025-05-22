FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

{% if cookiecutter.include_api == 'yes' %}
CMD ["uvicorn", "src.inference_api:app", "--host", "0.0.0.0", "--port", "8000"]
{% else %}
CMD ["python", "src/train.py"]
{% endif %}