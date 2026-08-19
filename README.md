

# Automatically Finding Reward Model Biases

## Getting started

1. Clone this repo.
2. Create a uv venv somewhere you like (Python 3.12 or newer), and activate it. Then, in this directory, run `uv pip install -e .`
3. Create a `.env` in the folder, with your `OPENROUTER_API_KEY`. Some parts of this pipeline by default make calls to OpenAI/Claude models, but you can always route them to OpenRouter by setting `force_caller="openrouter"` in the caller.
4. The entrypoint is `train.py`. Example:

```
python train.py \
--student_model skywork-qwen-0.6b \
--teacher_model gpt-5-mini \
--topic_ids 0 1 \
--planner_type list_reverse \
--direction plus \
--n_new 4 \
--n_pop_initial 8 \
--n_pop_targets 4 2 \
--train_batch_sizes 4 8 \
--m_var 1 \
--n_planner_requests 8 \
--val_split_size 0
```
