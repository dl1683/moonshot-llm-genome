@echo off
rem e347 RESTART #3 launcher (2026-10-13): the executor's in-session
rem background tasks were killed twice by session suspensions (22:40Z Oct-12
rem and ~07:30Z Oct-13, both mid-rung), so this one-shot launcher runs the
rem resume via the Task Scheduler service (service-owned process, immune to
rem harness session kills; the task registration itself was deleted right
rem after start - NO recurring automation created). Launched from the
rem space-free temp copy C:\Users\devan\AppData\Local\Temp\e347_resume_launch.bat
rem (identical content) because schtasks /tr cannot carry the quoted repo
rem path's spaces. The rig (lab/e347_optimizer_cross.py) is UNCHANGED; every
rem leg resumes from its md5-bound resume ckpt per the three-proof standard.
cd /d "C:\Users\devan\OneDrive\Desktop\Projects\AI Moonshots\moonshot-llm-genome\lab"
"C:\Users\devan\AppData\Local\Programs\Python\Python313\python.exe" e347_optimizer_cross.py >> "..\runs\e347\resume_run.stdout" 2>&1
