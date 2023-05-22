from experiments.utils import load_summary_benchmarks


validation_benchmarks = load_summary_benchmarks("validation")
test_benchmarks = load_summary_benchmarks("test")

# Drop ES, FR, JA, KO, ZH
validation_benchmarks = validation_benchmarks[:-5]
test_benchmarks = test_benchmarks[:-5]

benchmarks = validation_benchmarks + test_benchmarks
assert len(benchmarks) == 10

template = """\
Dataset & Document pairs & Tokens & Labels $< 0.5$ & Labels $\\ge 0.5$ & Unlabeled Tokens \\\\ \\midrule
\\textit{Validation split} & & & & & \\\\
iSTS                       &  &  &  &  &  \\\\
\mbox{+ Negatives (50\% paraphrases)}                &  &  &  &  &  \\\\
\mbox{+ Documents (5 sentences)}                &  &  &  &  &  \\\\
\mbox{+ Permuted (5 inversions)}                 &  &  &  &  &  \\\\
+ Cross-lingual (\\textsc{de})         &  &  &  &  &  \\\\ \\midrule
\\textit{Test split} & & & & & \\\\
iSTS                       &  &  &  &  &  \\\\
\mbox{+ Negatives (50\% paraphrases)}                &  &  &  &  &  \\\\
\mbox{+ Documents (5 sentences)}                &  &  &  &  &  \\\\
\mbox{+ Permuted (5 inversions)}                 &  &  &  &  &  \\\\
+ Cross-lingual (\\textsc{de})         &  &  &  &  &  \\\\ \\bottomrule"""

for benchmark in benchmarks:
    template = template.replace("&  ", f"& {benchmark.num_document_pairs} ", 1)
    template = template.replace("&  ", f"& {benchmark.num_tokens} ", 1)
    template = template.replace("&  ", f"& {100 * benchmark.num_labels_lt_05 / benchmark.num_tokens:.1f}\\% ", 1)
    template = template.replace("&  ", f"& {100 * benchmark.num_labels_gte_05 / benchmark.num_tokens:.1f}\\% ", 1)
    template = template.replace("&  ", f"& {100 * benchmark.num_unlabeled_tokens / benchmark.num_tokens:.1f}\\% ", 1)

print(template)
