
rewrite_prompt_v2='''
You are given a complete problem-solving process. Your task is to enhance this process by selectively replacing appropriate sections with Python code where computational assistance would be beneficial (e.g., calculations, data processing). Maintain the original approach when conventional reasoning suffices. Follow these guidelines precisely:

1.Code Requirements:
    Ensure all code is syntactically correct and directly executable
    Include all necessary import statements within the code blocks
    Add concise comments explaining each step's purpose
    Avoid duplicating problem descriptions in code comments

2.Structural Format:
    Preserve the original <think> and <answer> framework:
        Place all code and computational reasoning within <think> tags
        Keep <answer> concise, summarizing only key outcomes without repeating analysis
        <answer> contains final answer in $\boxed{}$
    Maintain a natural integration of code blocks and textual reasoning

3.Formatting Rules:
    Enclose all code blocks with <code> tags (markdown syntax)
    For output results, use <interpreter> tags
    Limit each code block to a single final print statement when output is needed

4.Execution Principles:
    Only implement code where it provides clear value
    Ensure seamless coexistence of code and conventional reasoning
    Never force code implementation when unnecessary

Example Structure:

<think>
[Textual analysis of the problem...]

<code>
```
# Required imports
import numpy as np

# Calculate mean value
data = np.array([1, 2, 3, 4])
result = np.mean(data)
print(f"Mean value: {result}")
```
</code>
<interpreter>
Mean value: 2.5
</interpreter>

[Additional reasoning...]
</think>

<answer>
[Concise summary of key findings and solutions]
</answer>

Special Notes:

1. Prioritize clarity and efficiency in code implementation
2. Maintain consistent formatting throughout
3. Verify all code executes properly in isolation
4. Focus comments on implementation details rather than restating the problem
'''



rewrite_prompt_v3='''
You are given a complete problem-solving process. Review it and selectively replace appropriate sections with Python code where computational assistance would be beneficial (e.g., calculations, data processing). Follow these strict guidelines:

1.Code Implementation Rules:
    Only add code where it provides clear computational value
    Ensure all code is production-ready:
        Correct syntax and complete imports
        Directly executable
        Single print statement per block for final output
    Include brief, non-redundant comments explaining implementation logic

2.Structural Requirements:
    Maintain original <think>/<answer> framework:
        All code must appear within <think> blocks
        <answer> contains only novel conclusions (no repetition) and final answer in $\boxed{}$
    Preserve natural flow between:
        Textual reasoning
        Code blocks (marked with <code>)
        Outputs (marked with <interpreter>)

3.Content Principles:
    Never duplicate explanations between:
        Problem description
        Code comments
        Answer summary
    Keep outputs concise - avoid verbose explanations
    Simplify straightforward reasoning processes

Example Implementation:

<think>
First, we need to calculate the statistical properties of the dataset.

<code>
```
import numpy as np

# Calculate core statistics for numerical analysis
values = np.array([12, 15, 18, 22])
mean = np.mean(values)
std_dev = np.std(values)
print(f"Statistics| Mean:{mean:.2f}, SD:{std_dev:.2f}")
```
</code>
<interpreter>
Statistics| Mean:16.75, SD:3.59
</interpreter>

The results suggest we should...
</think>

<answer>
The analysis indicates moderate variability (SD=3.59) around the mean of 16.75.
</answer>

Critical Compliance Notes:

1.Never force code where unnecessary
2.Maintain strict separation of:
    Problem description (original text)
    Implementation details (code comments)
    Final conclusions (answer)
3.All outputs must validate against:
    Correct execution
    Minimal duplication
    Appropriate brevity
'''