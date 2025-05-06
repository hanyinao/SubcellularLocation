@chcp 65001 >nul
@echo off
set "input_dir=D:\Han\software\Math\CSUpan\ShareCache\尹涵(数学与统计学院)\赛\一流专业人才计划\Codes\split_sequences"
set "output_dir=D:\Han\software\Math\CSUpan\ShareCache\尹涵(数学与统计学院)\赛\一流专业人才计划\Codes\pssm_sequences"
set "db_path=D:\SubcellularLocalization\BLAST\ncbi-blast-2.16.0+\db\human_subset"

if not exist "%output_dir%" mkdir "%output_dir%"

rem 启用延迟扩展
setlocal enabledelayedexpansion

rem 遍历输入目录下的所有 .fasta 文件
for %%f in ("%input_dir%\*.fasta") do (
    rem 检查文件是否存在
    if exist "%%f" (
        rem 提取文件名（不含扩展名）
        set "base_name=%%~nf"
        echo 正在处理文件: %%f
        rem 运行PSI - BLAST生成PSSM
        psiblast -query "%%f" -db "%db_path%" -num_iterations 3 -comp_based_stats 1 ^
    	-out_ascii_pssm "%output_dir%\!base_name!.pssm" -outfmt 6 -num_threads 2
        rem 检查psiblast命令是否执行成功
        if !errorlevel! equ 0 (
            echo 成功生成PSSM文件: "%output_dir%\!base_name!.pssm"
        ) else (
            echo 错误: 处理 %%f 时psiblast命令执行失败。
        )
    ) else (
        echo 警告: 未找到匹配的 .fasta 文件。
    )
)

endlocal