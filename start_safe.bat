@echo off
set FORCE_CPU=1
echo Iniciando PaleographIA em MODO DE SEGURANCA (CPU)...
echo Isso evitara que sua GPU sobresquente e o notebook reinicie.
echo Utilizando 16GB de RAM disponiveis.
python server/main.py
pause
