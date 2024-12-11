# Python 3.10 Lambda base image
FROM public.ecr.aws/lambda/python:3.10

# Copy requirements.txt to container root directory
COPY requirements.txt ./
COPY threshold_config.json /tmp/threshold_config.json
COPY model ./model
COPY submodules ./submodules

RUN yum install -y mesa-libGL mesa-libGL-devel

# Install dependencies
RUN pip3 install -r ./requirements.txt

# Copy function code to container
COPY lambda_function.py ./

# Setting the CMD to your handler
CMD [ "lambda_function.lambda_handler" ]

