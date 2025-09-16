@echo off
cd /d C:\kafka

REM =========================
REM Start Zookeeper
REM =========================
echo Starting Zookeeper...
start "Zookeeper" cmd /k ".\bin\windows\zookeeper-server-start.bat .\config\zookeeper.properties"
timeout /t 10 >nul

REM =========================
REM Start Kafka Broker
REM =========================
echo Starting Kafka Broker...
start "Kafka Broker" cmd /k ".\bin\windows\kafka-server-start.bat .\config\server.properties"
timeout /t 15 >nul

REM =========================
REM Wait for Kafka to be ready (check if port 9092 is listening)
REM =========================
echo Waiting for Kafka to be ready...
:WAIT_FOR_KAFKA
    echo Checking if Kafka is listening on port 9092...
    netstat -an | findstr "9092" | findstr "LISTENING" >nul
    if %ERRORLEVEL% neq 0 (
        echo Kafka is not ready yet. Retrying in 5 seconds...
        timeout /t 5 >nul
        goto WAIT_FOR_KAFKA
    )

REM =========================
REM Create Kafka topic
REM =========================
echo Kafka is ready. Creating topic 'vigilix-stream'...
.\bin\windows\kafka-topics.bat --create --if-not-exists --topic vigilix-stream --bootstrap-server localhost:9092 --partitions 1 --replication-factor 1

echo Kafka setup complete.
pause
