import os
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling, pipeline
from datasets import load_dataset
import torch
from peft import LoraConfig, get_peft_model
from transformers import BitsAndBytesConfig
from transformers.trainer_utils import get_last_checkpoint


def main():

    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Device: {device}')

    print(torch.cuda.is_available())  # Должно вернуть True
    print(torch.version.cuda)

    model_name = "ifable/gemma-2-Ifable-9B"  # или другая модель
    cache_dir = "E:\\ai\\ft\\cache"

    print('Quantization')
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16
    )

    device_map = "cuda:0"

    # Загружаем модель и токенизатор
    print('Load model and tokenizer')
    print(f'Model: {model_name}')
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        quantization_config=quantization_config,
        device_map=device_map,  # Распределение модели
        attn_implementation="eager",  # Используем eager attention
        use_cache=False
        )
    print(f'Tokenizer: {model_name}')
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

    # Настройка LoRA
    print('LoRA')
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding=True, max_length=512)

    print('Load dataset')
    dataset = load_dataset("text", data_files={"train": "d:\\ai\\learn\\ft\\*.*"})
    print('Tokenize dataset')
    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    # Добавление labels
    def add_labels(examples):
        examples["labels"] = examples["input_ids"]
        return examples
    print('Add labels')
    tokenized_datasets = tokenized_datasets.map(add_labels, batched=True)

    # Настройка аргументов обучения
    print('Training arguments')
    training_args = TrainingArguments(
        output_dir="./results",
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        save_steps=100,  # Сохранять чекпоинт каждые 100 шагов
        save_total_limit=2,  # Максимальное количество чекпоинтов
        logging_dir="./logs",
        logging_steps=10,  # Логировать каждые 100 шагов
        prediction_loss_only=False,
        fp16=True,
        report_to="tensorboard",
        dataloader_num_workers=4,
        learning_rate=5e-5,
        weight_decay=0.01
    )

    print('Data collator')
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    print(f'Number of training examples: {len(tokenized_datasets["train"])}')

    # Поиск последнего чекпоинта
    print('Get last checkpoint')
    latest_checkpoint = get_last_checkpoint(training_args.output_dir)
    print(f'Last checkpoint: {latest_checkpoint}')


    print('Optimizer')
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    print('Trainer')
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        data_collator=data_collator,
        optimizers=(optimizer, None),
    )

    # Очистка кэша CUDA перед обучением
    print('Clear CUDA cache')
    torch.cuda.empty_cache()

    # Обучение
    print('Training')
    if latest_checkpoint:
        print(f"Resuming training from checkpoint: {latest_checkpoint}")
        trainer.train(resume_from_checkpoint=latest_checkpoint)
    else:
        trainer.train()

    # Сохранение модели
    print('Saving model')
    model.save_pretrained("./fine_tuned_model")
    tokenizer.save_pretrained("./fine_tuned_model")

    # Генерация текста
    print('Generating text')
    generator = pipeline("text-generation", model="./fine_tuned_model")
    print(generator(
        "Once upon a time, in a far away land, there lived a one boy. ",
        max_length=500,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        num_return_sequences=3,  # Количество генерируемых последовательностей
        do_sample=True  # Включение семплирования
    ))


if __name__ == "__main__":
    main()