FROM python:3.7.13-slim-buster

RUN pip install numpy
RUN pip install pandas
RUN pip install flask
RUN pip install scikit-learn
RUN pip install gensim==4.1.2
RUN pip install elasticsearch==7.17.1
RUN pip install tensorflow==2.7.0
RUN pip install pyvi


WORKDIR /app
COPY . /app

ENV IP_ELASTICSEARCH="localhost"
ENV PORT_ELASTICSEARCH=9200

ENTRYPOINT ["python"]

CMD ["serve.py"]