chcp 65001
set "input_dir=D:\Han\software\Math\CSUpan\ShareCache\尹涵(数学与统计学院)\赛\一流专业人才计划\Codes\split_sequences"
setlocal enabledelayedexpansion
for %%f in ("%input_dir%\*.fasta") do (
    set "base_name=%%~nf"
    echo !base_name!
)
endlocal