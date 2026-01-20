# FastAPI Framework

FastAPI is a modern, fast web framework for building APIs with Python 3.7+ based on standard Python type hints.

## Installation

Install FastAPI using pip:
```bash
pip install fastapi uvicorn
```

## Basic Example

Here's a simple FastAPI application:
```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/items/{item_id}")
def read_item(item_id: int):
    return {"item_id": item_id}
```

## Key Features

- Fast performance, comparable to NodeJS and Go
- Easy to use with automatic interactive API documentation
- Based on open standards: OpenAPI and JSON Schema
- Type hints for better editor support

## Running the Application

Use uvicorn to run your FastAPI application:
```bash
uvicorn main:app --reload
```
