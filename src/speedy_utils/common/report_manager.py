import os
from collections import defaultdict
from datetime import datetime



class ReportManager:
    def __init__(self):
        self.cache_dir = os.path.expanduser("~/.cache/speedy_utils")
        os.makedirs(self.cache_dir, exist_ok=True)

    def save_report(self, errors, results, execution_time=None, metadata=None):
        report_path = os.path.join(
            self.cache_dir, f"report_{datetime.now().strftime('%m%d_%H%M')}.md"
        )
        os.makedirs(os.path.dirname(report_path), exist_ok=True)

        # Group errors by error type
        error_groups = defaultdict(list)
        for err in errors[:10]:
            error_type = err["error"].__class__.__name__
            error_groups[error_type].append(err)

        md_content = [
            "# Multi-thread Execution Report",
            f"\n## Summary (Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')})",
        ]

        if metadata:
            md_content.extend(
                [
                    "\n### Execution Configuration",
                    f"- Mode: {metadata['mode']}",
                    f"- Workers: {metadata['max_workers']}",
                    f"- Execution type: {metadata['execution_mode']}",
                    f"- Total inputs: {metadata['total_inputs']}",
                ]
            )

        md_content.extend(
            [
                "\n### Results Overview",
                f"- Total items processed: {len(results)}",
                f"- Success rate: {(len(results) - len(errors))/len(results)*100:.1f}%",
                f"- Total errors: {len(errors)}",
            ]
        )

        if execution_time:
            md_content.append(f"- Execution time: {execution_time:.2f}s")
            md_content.append(
                f"- Average speed: {len(results)/execution_time:.1f} items/second"
            )

        if error_groups:
            md_content.extend(
                ["\n## Errors by Type", "Click headers to expand error details."]
            )

            for error_type, errs in error_groups.items():
                md_content.extend(
                    [
                        "\n<details>",
                        f"<summary><b>{error_type}</b> ({len(errs)} occurrences)</summary>\n",
                        "| Index | Input | Error Message |",
                        "|-------|-------|---------------|",
                    ]
                )

                for err in errs:
                    md_content.append(
                        f"| {err['index']} | `{err['input']}` | {str(err['error'])} |"
                    )

                # Add first traceback as example
                md_content.extend(
                    [
                        "\nExample traceback:",
                        "```python",
                        errs[0]["traceback"],
                        "```",
                        "</details>",
                    ]
                )

            # Add a section listing all error indices
            md_content.extend(
                [
                    "\n## Error Indices",
                    "List of indices for items that encountered errors:",
                    ", ".join(str(err["index"]) for err in errors),
                ]
            )

        md_content.extend(
            [
                "\n## Results Summary",
                f"- Successful executions: {len(results) - len(errors)}",
                f"- Failed executions: {len(errors)}",
                "\n<details>",
                "<summary>First 5 successful results</summary>\n",
                "```python",
                str([r for r in results[:5] if r is not None]),
                "```",
                "</details>",
            ]
        )

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_content))
        print(f"Report saved at: {report_path}")
