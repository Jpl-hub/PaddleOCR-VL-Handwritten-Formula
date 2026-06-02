# Annotation Guideline

## Goal

Annotate handwritten mathematical expressions as LaTeX that can be rendered by standard math engines.

## Core Rules

- Preserve mathematical meaning over handwriting quirks.
- Use display math wrapper `\[` and `\]` for final labels.
- Do not include explanatory prose in the label.
- Use canonical LaTeX commands: `\frac`, `\sqrt`, `\sum`, `\int`, `\lim`, `\sin`, `\cos`, `\log`, and similar commands.
- Keep variables and constants exactly as written when visually distinguishable.
- Use braces for all superscript/subscript groups longer than one token.

## Ambiguous Symbols

| Case | Rule |
|---|---|
| `1` vs `l` | Use surrounding expression context; mark ambiguous cases for review. |
| `0` vs `O` | Use mathematical context; prefer `0` in numeric contexts. |
| `x` vs `\times` | Use `\times` only when the symbol is clearly an operator. |
| decimal point vs dot operator | Use `.` for decimal values and `\cdot` for multiplication. |
| Greek letters | Use LaTeX commands such as `\alpha`, `\beta`, `\lambda`. |

## Structures

- Fractions: `\frac{numerator}{denominator}`.
- Roots: `\sqrt{x}` or `\sqrt[n]{x}`.
- Matrices: `\begin{matrix} ... \end{matrix}` when line structure is clear.
- Piecewise: `\begin{cases} ... \end{cases}`.
- Integrals: `\int_{a}^{b}` when limits are present.
- Summations: `\sum_{i=1}^{n}` when limits are present.

## Multi-Line Expressions

- If the image contains one expression split across lines, keep one LaTeX expression.
- If it contains separate equations, join them with `\\` inside an aligned environment when alignment is visually meaningful.
- Do not invent missing operators.

## Review Checklist

1. The label renders without LaTeX errors.
2. The rendered expression matches the source image.
3. Braces are balanced.
4. Superscripts and subscripts are attached to the intended base.
5. Ambiguous symbols are either resolved or marked for second review.
6. The sample has difficulty and source metadata.

## Final JSONL Format

```json
{
  "messages": [
    {"role": "user", "content": "<image>Formula Recognition:"},
    {"role": "assistant", "content": "\\[x+1=2\\]"}
  ],
  "images": ["images/eval/sample_000001.png"],
  "meta": {
    "source": "volunteer_phone_photo",
    "difficulty": "easy"
  }
}
```
