FROM continuumio/anaconda3

RUN apt-get update && apt-get install -y libgtk2.0-dev && \
    rm -rf /var/lib/apt/lists/*

RUN /opt/conda/bin/conda update -n base -c defaults conda && \
    /opt/conda/bin/conda install python=3.7 && \
    /opt/conda/bin/conda install anaconda-client && \	
    /opt/conda/bin/conda install numpy pandas flask scikit-learn keras -y && \
    /opt/conda/bin/conda upgrade dask && \
    pip install gensim==4.1.2 && \
    pip install elasticsearch==7.17 && \
    pip install tensorflow==2.6.0 && \
    pip install pyvi

WORKDIR /app

COPY . /app

ENTRYPOINT [ "python" ]

CMD ["serve.py"]
