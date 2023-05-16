FROM tensorflow/tensorflow

COPY ./api /app
WORKDIR /app

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

CMD ["uvicorn", "run:app", "--host", "0.0.0.0", "--port", "8000"]
