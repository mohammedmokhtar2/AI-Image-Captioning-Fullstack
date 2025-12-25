# Use Python 3.9 as base image
FROM python:3.9

# Set the working directory inside the container
WORKDIR /code

# Copy requirements and install dependencies
# We use --no-cache-dir to keep the image small
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Install python-multipart (required for UploadFile in FastAPI) if not in requirements.txt
RUN pip install python-multipart

# Copy the entire project into the container
COPY . /code

# Create a directory for Hugging Face cache to avoid permission errors
RUN mkdir -p /code/cache
ENV TRANSFORMERS_CACHE=/code/cache
ENV TORCH_HOME=/code/cache
RUN chmod -R 777 /code/cache

# Expose the default Hugging Face port
EXPOSE 7860

# Command to run the application
# We point to src.api.app:app because app.py is inside src/api/
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "7860"]