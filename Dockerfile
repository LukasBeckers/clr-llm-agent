FROM python:3.12.3

WORKDIR /app

COPY ./app ./

RUN pip install --no-cache-dir -r /app/requirements.txt

RUN apt-get update && apt-get install -y \
    wget \
    gnupg2 \
    unzip \
    && rm -rf /var/lib/apt/lists/*

RUN wget -q -O - https://dl.google.com/linux/linux_signing_key.pub | apt-key add - \
    && sh -c 'echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google-chrome.list' \
    && apt-get update && apt-get install -y \
    google-chrome-stable \
    && rm -rf /var/lib/apt/lists/*

RUN CHROME_DRIVER_VERSION=$(curl -sS chromedriver.storage.googleapis.com/LATEST_RELEASE) && \
    wget -O /tmp/chromedriver.zip https://chromedriver.storage.googleapis.com/$CHROME_DRIVER_VERSION/chromedriver_linux64.zip && \
    unzip /tmp/chromedriver.zip -d /usr/local/bin/ && \
    rm /tmp/chromedriver.zip

ENV CHROME_DRIVER_PATH /usr/local/bin/chromedriver
ENV PATH $PATH:/usr/local/bin/chromedriver

CMD ["python", "/app/main.py"]
