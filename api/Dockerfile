FROM python:3.10
ADD ./ /api

WORKDIR /api

RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get -y update && apt-get --yes install libsndfile1
RUN apt-get install -y ffmpeg

# ENV PORT=8080

# Command that will start the application
# CMD uvicorn main:app --host 0.0.0.0 --port=${PORT}
CMD python main.py