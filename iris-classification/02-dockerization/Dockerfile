FROM continuumio/anaconda3
COPY ./iris-inferencing /usr/local/python/
EXPOSE 5000
WORKDIR /usr/local/python
RUN pip install -r requirements.txt
CMD python 02-model-inferencing.py