FROM python:3.10

WORKDIR /app
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt
RUN apt-get -y update && apt-get --yes install libsndfile1
RUN apt-get install -y ffmpeg
EXPOSE 8501
COPY . /app
ENTRYPOINT ["streamlit", "run"]

CMD ["streamlit_app.py"]