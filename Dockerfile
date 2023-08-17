FROM python:3.10
LABEL authors="ravid"
WORKDIR /code
COPY ./requirements.txt /code/requirements.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
COPY ./AppFastAPI /code
COPY ./AppTFJS /AppTFJS
EXPOSE 8000
CMD ["python", "main.py"]

#WORKDIR /code
#COPY ./requirements.txt /code/requirements.txt
#RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
#COPY ./app /code/app
#CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]