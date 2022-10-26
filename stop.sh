#!/bin/bash

PID=$(ps aux | grep 'uvicorn main:app' | grep -v grep | awk {'print $2'} | xargs)

if [ "$PID" != "" ]; then
   kill -9 $PID
else
   echo "No such process."
fi
