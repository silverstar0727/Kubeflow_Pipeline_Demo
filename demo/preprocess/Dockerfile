FROM frolvlad/alpine-python-machinelearning
#RUN pip3 install — upgrade pandas
#RUN pip3 install — upgrade gcsfs
#RUN pip3 install — upgrade argparse
ENV PYTHONUNBUFFERED 1
RUN mkdir -p /titanic/src
COPY . /titanic/src
WORKDIR /titanic/src
ENTRYPOINT [“python”, “titanic_preprocess.py”]