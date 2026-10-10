bash project/scripts/run.sh python3 -m project.scripts.audit_v4_training \
  --stage v4_base --train-run main_20261009T100336Z_3e49ab

它会区分全零奖励、全满分、相同部分得分、有奖励差异的组，并导出候选模型输出。无需 GPU，不重新推理。

运行后，把下面目录中的 summary.txt 和 candidate_traces.txt 发给我：

runtime/runs/v4_base/main_20261009T100336Z_3e49ab/offline_audit/