FROM python:3.13

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir -r /code/requirements.txt

COPY ./app /code/app

EXPOSE 1001

CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "1001"]
