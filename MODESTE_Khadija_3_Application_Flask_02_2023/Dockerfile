FROM tensorflow/tensorflow

COPY . /app
WORKDIR /app

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

CMD python run.py -h 0.0.0.0 -p 5000
