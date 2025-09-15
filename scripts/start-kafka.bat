@echo off
title Kafka Setup Script
echo ========================================
echo 🚀 Starting Zookeeper and Kafka Broker
echo ========================================

REM --- CONFIGURABLE KAFKA PATH ---
set KAFKA_DIR=C:\kafka

REM --- Clean up previous Kafka data ---
echo ▶ Cleaning up previous Kafka data...
rmdir /s /q "%KAFKA_DIR%\tmp"
mkdir "%KAFKA_DIR%\tmp"

REM --- Step 1: Start Zookeeper ---
echo ▶ Starting Zookeeper...
start "Zookeeper" cmd /k "cd /d %KAFKA_DIR% && bin\windows\zookeeper-server-start.bat config\zookeeper.properties"

REM --- Wait for Zookeeper to initialize ---
timeout /t 10 /nobreak >nul

REM --- Step 2: Start Kafka Broker ---
echo ▶ Starting Kafka Broker...
start "Kafka Broker" cmd /k "cd /d %KAFKA_DIR% && bin\windows\kafka-server-start.bat config\server.properties"

REM --- Wait for Kafka Broker to initialize ---
timeout /t 15 /nobreak >nul

REM --- Step 3: Create Topic vigilix-stream ---
echo ▶ Creating topic 'vigilix-stream' if it doesn't exist...
cd /d %KAFKA_DIR%
bin\windows\kafka-topics.bat --create --if-not-exists --topic vigilix-stream --bootstrap-server localhost:9092 --partitions 1 --replication-factor 1

REM --- Step 4: List all topics to confirm creation ---
echo ▶ Listing all Kafka topics...
bin\windows\kafka-topics.bat --list --bootstrap-server localhost:9092

echo ========================================
echo ✅ Kafka is running with topic: vigilix-stream
echo ========================================
pause