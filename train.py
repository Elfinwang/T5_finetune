import yaml
import logging
import traceback
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, TrainingArguments, Trainer, TrainerCallback
from peft import LoraConfig, get_peft_model
import json
import os


# 日志配置
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class T5Classifier:
    def __init__(self, config_file='./config.yaml'):
        try:
            # 加载配置
            self.config = self.load_config(config_file)

            # 控制台日志
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(log_formatter)
            logger.addHandler(console_handler)

            # 文件日志（output_dir/train.log）
            log_file = os.path.join(self.config['output_dir'], 'train.log')
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setFormatter(log_formatter)
            logger.addHandler(file_handler)

            self.logger = logger  

            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.config['model_path'], device_map="auto")
            self.tokenizer = AutoTokenizer.from_pretrained(self.config['model_path'])
            self.logger.info(f"模型和分词器从 {self.config['model_path']} 加载成功")

        except Exception as e:
            try:
                self.logger.error(f"初始化过程中发生错误: {str(e)}")
                self.logger.error(traceback.format_exc())
            except Exception:
                print(f"初始化过程中发生错误: {str(e)}")
                print(traceback.format_exc())
            raise

    def load_config(self, config_file):
        try:
            with open(config_file, 'r') as file:
                config = yaml.safe_load(file)
            
            # 打印配置项的类型和值
            for key, value in config.items():
                logger.info(f"{key} - 类型: {type(value)}, 值: {value}")
            
            return config
        except Exception as e:
            logger.error(f"加载配置时发生错误: {str(e)}")
            logger.error(traceback.format_exc())
            raise



    def prepare_datasets(self):
        try:
            logger.info("正在加载本地 JSON 数据集...")
            with open(self.config['data_path'], 'r') as f:
                data = json.load(f)

            # 构建输入和输出
            inputs = [item["instruction"] + item["input"] for item in data]
            targets = [item["output"] for item in data]


            # 随机打乱索引并划分训练和验证集
            

            indices = np.arange(len(inputs))
            np.random.shuffle(indices)

            split_idx = int(len(inputs) * 0.99)
            train_indices = indices[:split_idx]
            val_indices = indices[split_idx:]

            train_inputs = [inputs[i] for i in train_indices]
            val_inputs = [inputs[i] for i in val_indices]
            train_targets = [targets[i] for i in train_indices]
            val_targets = [targets[i] for i in val_indices]

            # 划分训练和验证集
            # split_idx = int(len(inputs) * 0.99)
            # train_inputs, val_inputs = inputs[:split_idx], inputs[split_idx:]
            # train_targets, val_targets = targets[:split_idx], targets[split_idx:]

            # 构建datasets
            train_dataset = [{"input": inp, "output": tgt} for inp, tgt in zip(train_inputs, train_targets)]
            val_dataset = [{"input": inp, "output": tgt} for inp, tgt in zip(val_inputs, val_targets)]

            # 分词处理
            def preprocess(examples):
                model_inputs = self.tokenizer(
                    examples["input"],
                    max_length=self.config['max_length'],
                    padding="max_length",
                    truncation=True,
                )
                labels = self.tokenizer(
                    examples["output"],
                    max_length=16,
                    padding="max_length",
                    truncation=True,
                )
                model_inputs["labels"] = labels["input_ids"]
                return model_inputs

            # 转为HF Dataset格式
            from datasets import Dataset
            self.train_dataset = Dataset.from_list(train_dataset).map(preprocess, batched=True, remove_columns=["input", "output"])
            self.eval_dataset = Dataset.from_list(val_dataset).map(preprocess, batched=True, remove_columns=["input", "output"])

            logger.info("本地数据集已处理并分词。")
        except Exception as e:
            logger.error(f"准备数据集时发生错误: {str(e)}")
            logger.error(traceback.format_exc())
            raise

 

    def configure_lora(self):
        try:
            lora_config = LoraConfig(
                r=int(self.config['lora_r']),
                lora_alpha=int(self.config['lora_alpha']),
                target_modules=["q", "v"],
                lora_dropout=float(self.config['lora_dropout']),
                bias="none",
                task_type="SEQ_2_SEQ_LM"
            )
            self.model = get_peft_model(self.model, lora_config)
            logger.info("LoRA 配置已应用到模型。")
        except Exception as e:
            logger.error(f"配置 LoRA 时发生错误: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def train(self):
        try:
            # 检查关键参数的类型和值
            critical_params = ['learning_rate', 'num_train_epochs', 'per_device_train_batch_size', 
                               'per_device_eval_batch_size', 'gradient_accumulation_steps', 
                               'warmup_ratio', 'save_steps', 'logging_steps', 'eval_steps']
            for param in critical_params:
                logger.info(f"{param} - 类型: {type(self.config[param])}, 值: {self.config[param]}")

            training_args = TrainingArguments(
                output_dir=self.config['output_dir'],
                per_device_train_batch_size=int(self.config['per_device_train_batch_size']),
                per_device_eval_batch_size=int(self.config['per_device_eval_batch_size']),
                gradient_accumulation_steps=int(self.config['gradient_accumulation_steps']),
                learning_rate=float(self.config['learning_rate']),
                lr_scheduler_type=self.config['lr_scheduler_type'],
                warmup_ratio=float(self.config['warmup_ratio']),
                num_train_epochs=float(self.config['num_train_epochs']),
                save_strategy=self.config['save_strategy'],
                save_steps=int(self.config['save_steps']),
                logging_steps=int(self.config['logging_steps']),
                evaluation_strategy=self.config['evaluation_strategy'],
                eval_steps=int(self.config['eval_steps'])
            )

            class LoggingCallback(TrainerCallback):
                def on_evaluate(self, args, state, control, metrics, **kwargs):
                    logger.info(f"评估指标: {metrics}")

                def on_log(self, args, state, control, logs=None, **kwargs):
                    if logs:
                        logger.info(f"训练日志: {logs}")
            
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
                compute_metrics=self.compute_metrics,
                callbacks=[LoggingCallback()],
            )
            
            logger.info("开始训练...")
            trainer.train()
            logger.info("训练完成。")

            logger.info("执行最终评估...")
            eval_results = trainer.evaluate()
            logger.info(f"最终评估结果: {eval_results}")

        except Exception as e:
            logger.error(f"训练过程中发生错误: {str(e)}")
            logger.error(f"错误类型: {type(e)}")
            logger.error(f"错误参数: {e.args}")
            logger.error(f"错误追踪:\n{traceback.format_exc()}")
            raise

    def preprocess_function(self, examples):
        try:
            text_column = "sentence"
            label_column = "text_label"
            inputs = examples[text_column]
            targets = examples[label_column]
            
            model_inputs = self.tokenizer(inputs, max_length=self.config['max_length'], padding="max_length", truncation=True)
            labels = self.tokenizer(targets, max_length=3, padding="max_length", truncation=True)
            
            model_inputs["labels"] = labels["input_ids"]
            
            # 记录一些样本
            logger.info(f"样本输入: {inputs[0]}")
            logger.info(f"样本目标: {targets[0]}")
            logger.info(f"样本模型输入: {model_inputs['input_ids'][0]}")
            logger.info(f"样本标签: {model_inputs['labels'][0]}")
            
            return model_inputs
        except Exception as e:
            logger.error(f"预处理函数中发生错误: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def compute_metrics(self, eval_pred):
        try:
            predictions, labels = eval_pred
            logger.info(f"预测结果类型: {type(predictions)}")
            logger.info(f"标签类型: {type(labels)}")
            
            if isinstance(predictions, tuple):
                logger.info(f"预测结果是一个包含 {len(predictions)} 个元素的元组")
                # 假设第一个元素包含 logits
                predictions = predictions[0]
            
            logger.info(f"预测结果形状: {predictions.shape}")
            logger.info(f"标签形状: {labels.shape}")
            
            # 通过取 argmax 获取预测的类别
            predicted_classes = np.argmax(predictions, axis=-1)
            
            decoded_preds = self.tokenizer.batch_decode(predicted_classes, skip_special_tokens=True)
            # 将 -100 替换为 tokenizer.pad_token_id
            labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
            decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

            logger.info(f"样本预测: {decoded_preds[0]}")
            logger.info(f"样本标签: {decoded_labels[0]}")

            # 比较预测和标签
            accuracy = sum([pred.strip() == label.strip() for pred, label in zip(decoded_preds, decoded_labels)]) / len(decoded_preds)

            logger.info(f"计算得到的准确率: {accuracy}")

            return {"accuracy": accuracy}
        except Exception as e:
            logger.error(f"compute_metrics 中发生错误: {str(e)}")
            logger.error(traceback.format_exc())
            return {"accuracy": 0.0}  # 返回默认值

def main():
    try:
        classifier = T5Classifier()
        classifier.prepare_datasets()
        classifier.configure_lora()
        classifier.train()
    except Exception as e:
        logger.error(f"main 函数中发生错误: {str(e)}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()

