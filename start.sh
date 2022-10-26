#!/bin/sh

app_path="app"

cd ${app_path}

nohup python -m uvicorn main:app --host 0.0.0.0 --port 8086 > sentence-align-server.log 2>&1 &
