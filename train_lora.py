from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer, TrainerCallback
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
import os
import logging


save_dir = "./finetune_lora/t5-large-stats-v2"
os.makedirs(save_dir, exist_ok=True)

# log
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(save_dir, "training.log")),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# choose GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# 加载模型
model_name = "./model/t5-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# LoRA配置
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q", "v"],
    task_type=TaskType.SEQ_2_SEQ_LM,
)
model = get_peft_model(model, lora_config)

# 加载数据
dataset = load_dataset(
    "json",
    data_files="./data/train_stats_finetune.json",
)

# 数据预处理
def preprocess(example):
    source_text = example["instruction"]+example["input"]
    inputs = tokenizer(
        source_text,
        truncation=True,
        padding="max_length",
        max_length=32,
    )
    targets = tokenizer(
        example["output"],
        truncation=True,
        padding="max_length",
        max_length=32,
    )


    labels = targets["input_ids"]
    labels = [l if l != tokenizer.pad_token_id else -100 for l in labels]

    inputs["labels"] = labels

    return inputs

dataset = dataset.map(preprocess, remove_columns=["instruction", "input", "output"])

# 划分训练集和验证集
dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

logger.info(f"训练集大小: {len(train_dataset)}")
logger.info(f"验证集大小: {len(eval_dataset)}")


class LoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            if "eval_loss" in logs:
                logger.info(f"Validation - {logs}")
            else:
                logger.info(f"Train - {logs}")


# 参数
training_args = TrainingArguments(
    output_dir=save_dir,
    per_device_train_batch_size=1,
    num_train_epochs=10,
    logging_steps=100,
    save_steps=500,
    gradient_accumulation_steps=8,
    fp16=True,  
    evaluation_strategy="steps",
    eval_steps=500,
    report_to=[], 
    learning_rate=2e-5, 
)

# 训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    callbacks=[LoggingCallback],  
)

trainer.train()