FROM public.ecr.aws/lambda/python:3.8

COPY requirements.txt ./
RUN pip install --disable-pip-version-check -r requirements.txt && pip cache purge

ENV TRANSFORMERS_CACHE /var/cache/transformers
RUN transformers-cli download rinna/japanese-gpt2-small

COPY app.py .

CMD ["app.lambda_handler"]
