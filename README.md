# Learning DSPy

Backronym: "Declarative Self-improving Language Programs, pythonically." Basically the PyTorch of foundation models.

Just one of the things I'm learning. https://github.com/hchiam/learning

TODO: see how the DSPy framework solves the fragility problem in LLM-based applications by replacing prompting with programming and compiling.

https://github.com/stanfordnlp/dspy

https://colab.research.google.com/github/stanfordnlp/dspy/blob/main/intro.ipynb

https://dspy-docs.vercel.app/docs/quick-start/minimal-example

https://dspy-docs.vercel.app/docs/category/dspy-building-blocks

https://dspy-docs.vercel.app/docs/cheatsheet

https://dspy-docs.vercel.app/docs/category/deep-dive

## Workflow

1. data
2. pipeline (program of interacting declarative modules and prompts)
3. validation logic
4. compile (to optimize LM prompts to your (1) data, (2) pipeline, and (3) validation, to solve the task)
5. iterate

## Data

- `trainset`

## Pipeline

### Prediction

- signature `class` = docstring + input + output (yes, DSPy can see the docstring)
- `generate_answer = dspy.Predict(YourSignatureClassHere)`
  - or `generate_answer = dspy.ChainOfThought(YourSignatureClassHere)`
- `prediction = generate_answer(question=dev_example.question)`
- `print(dev_example.question)`
- `print(prediction.rationale.split('.', 1)[1].strip())` if you used `dspy.ChainOfThought` (ignore the "produce the answer." part of the `rationale`)
- `print(prediction.answer)`
- `turbo.inspect_history(n=1)`

```py
# question -> answer
class BasicQA(dspy.Signature):
    """Answer questions with short factoid answers."""
    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")

generate_answer = dspy.ChainOfThought(BasicQA) # <- BasicQA
prediction = generate_answer(question=dev_example.question)
print(f"Question: {dev_example.question}")
print(f"Thought: {prediction.rationale.split('.', 1)[1].strip()}")
print(f"Predicted Answer: {prediction.answer}")
```

### Retrieval

- `retrieve = dspy.Retrieve(k=3)`
- `topK_passages = retrieve(dev_example.question).passages`

```py
# context, question -> answer
class GenerateAnswer(dspy.Signature):
    """Answer questions with short factoid answers."""
    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")

class RAG(dspy.Module):
    # __init__ defines submodules it needs:
    def __init__(self, num_passages=3):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer) # <- GenerateAnswer

    # forward defines control flow: ((question -> context) -> RAG) -> final_prediction
    def forward(self, question):
        context = self.retrieve(question).passages
        rag_cot_answer = self.generate_answer(context=context, question=question)
        final_prediction = dspy.Prediction(context=context, answer=rag_cot_answer.answer)
        return final_prediction
```

## Validation and Compilation

- use a "teleprompter" AKA optimizer/compiler via "prompting at a distance"

```py
from dspy.teleprompt import BootstrapFewShot

# validation: check answer is correct, and check context contains that answer.
def validate_context_and_answer(example, pred, trace=None):
    is_answer_correct = dspy.evaluate.answer_exact_match(example, pred)
    is_answer_in_context = dspy.evaluate.answer_passage_match(example, pred)
    return is_answer_correct and is_answer_in_context

teleprompter = BootstrapFewShot(metric=validate_context_and_answer)
compiled_rag = teleprompter.compile(RAG(), trainset=trainset) # <- RAG from earlier
```

```py
question = "What castle did David Gregory inherit?"

prediction = compiled_rag(question)

print(f"Question: {question}")
print(f"Retrieved Contexts (truncated): {[c[:200] + '...' for c in prediction.context]}")
print(f"Predicted Answer: {prediction.answer}")
```

```py
# inspect the last prompt for the LM:
turbo.inspect_history(n=1)
```

```py
for name, parameter in compiled_rag.named_predictors():
    print(name)
    print(parameter.demos[0])
    print()
```

### Evaluation of final prediction

```py
from dspy.evaluate.evaluate import Evaluate

# exact match:
metric = dspy.evaluate.answer_exact_match

# create evaluator:
evaluate_on_hotpotqa = Evaluate(
  devset=devset,
  num_threads=1,
  display_progress=True,
  display_table=5
)

# use evaluator: (prints out metric result and table of inputs/outputs/metric values)
evaluate_on_hotpotqa(compiled_rag, metric=metric) # <- compiled_rag from earlier
```

### Evaluation of RAG retrievals

```py
def retrieved_gold_passages(example, pred, trace=None):
    gold_titles = set(map(dspy.evaluate.normalize_text, example['gold_titles']))
    found_titles = set(map(dspy.evaluate.normalize_text, [c.split(' | ')[0] for c in pred.context]))
    retrieved_some_gold_titles = gold_titles.issubset(found_titles)
    return retrieved_some_gold_titles

compiled_rag_retrieval_score = evaluate_on_hotpotqa(compiled_rag, metric=retrieved_gold_passages)
```

Tip: if (predictions correct > retrievals correct) then LM likely relying on memorized knowledge and not on retrievals.

## Iterate

Tip: if (one search query isn't enough) then "multi-hop" queries:

- (Baleen and GoldEn are examples of multi-hop search systems)

```py
# context, question -> query
class GenerateSearchQuery(dspy.Signature):
    """Write a simple search query that will help answer a complex question."""
    context = dspy.InputField(desc="may contain relevant facts")
    # or context = GenerateAnswer.signature.context # to avoid duplicating desc # <- GenerateAnswer
    question = dspy.InputField()
    query = dspy.OutputField()


from dsp.utils import deduplicate

class SimplifiedBaleen(dspy.Module):
    # __init__ defines submodules it needs:
    def __init__(self, passages_per_hop=3, max_hops=2):
        super().__init__()
        self.generate_query = [dspy.ChainOfThought(GenerateSearchQuery) for _ in range(max_hops)] # <- GenerateSearchQuery (c, q -> q)
        self.retrieve = dspy.Retrieve(k=passages_per_hop)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer) # <- GenerateAnswer (c, q -> a)
        self.max_hops = max_hops

    # forward defines control flow: ((context + question) -> query -> passages) -> context -> RAG) -> final_prediction
    def forward(self, question):
        context = []

        for hop in range(self.max_hops):
            query = self.generate_query[hop](
              context=context,
              question=question
            ).query

            passages = self.retrieve(query).passages

            context = deduplicate(context + passages)

        mhs_cot_answer = self.generate_answer(
          context=context,
          question=question
        )

        final_prediction = dspy.Prediction(
          context=context,
          answer=mhs_cot_answer.answer
        )
        return final_prediction
```
