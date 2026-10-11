1. 总体配对分析：

bash project/scripts/run.sh python3 -m project.scripts.review_v4 \
  --evidence --eval-run dev_20261011T031323Z_e7670f

2. 生日题完整轨迹导出：

bash project/scripts/run.sh python3 -m project.scripts.cases_v4 \
  --stage evidence --eval-run dev_20261011T031323Z_e7670f \
  --case-id 199fbba20bdc11eba7f7acde48001122 --quiet

3. 导演题完整轨迹导出：

bash project/scripts/run.sh python3 -m project.scripts.cases_v4 \
  --stage evidence --eval-run dev_20261011T031323Z_e7670f \
  --question-contains "Some Came Running" --quiet

文件都保存在：

runtime/runs/v4_base_evidence_eval/dev_20261011T031323Z_e7670f/

先发 review_summary.txt，再发两个 case_*.txt 即可