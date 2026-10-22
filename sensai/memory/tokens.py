"""Token estimation (M2 groundwork).

Ollama exposes no tokenize endpoint, so we estimate before sending and
record the real counts (prompt_eval_count / eval_count) afterwards.

The rule was fitted on real counts from qwen3-coder:30b (25 October 2026):
- French prose costs about 3.5 characters per token, but lesson material
  (Markdown tables, rōmaji, numbers, punctuation) only about 2.8: a plain
  chars/3.5 rule underestimated 75 lesson chunks out of 76, by up to 60 %.
  Letters, spaces, digits and punctuation are therefore counted apart:
  most digits and punctuation marks are a token of their own.
- Japanese and Korean cost 0.7 to 1 token per character: counting one per
  character stays on the safe side.
Over the 76 chunks of docs/lecons, this rule is 8 % above the real total
(worst chunk: 12 % under); it overestimates prose by 10 to 20 %, which only
wastes a little budget. ContextBuilder.observe() corrects what remains from
the real counts of each turn.
"""

import re

CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯＀-￯]")
PUNCT = re.compile(r"[^\w\s]")
DIGIT = re.compile(r"\d")
SPACE = re.compile(r"\s")
LETTERS_PER_TOKEN = 4.5
SPACES_PER_TOKEN = 2.0
MESSAGE_OVERHEAD = 5  # chat template tokens around each message (<|im_start|>role\n … <|im_end|>\n)


def estimate_tokens(text: str) -> int:
    cjk = len(CJK.findall(text))
    rest = CJK.sub("", text) if cjk else text
    punct = len(PUNCT.findall(rest))
    digits = len(DIGIT.findall(rest))
    spaces = len(SPACE.findall(rest))
    letters = len(rest) - punct - digits - spaces
    return int(letters / LETTERS_PER_TOKEN + spaces / SPACES_PER_TOKEN) + digits + punct + cjk + 1


def estimate_message(content: str) -> int:
    return estimate_tokens(content) + MESSAGE_OVERHEAD
