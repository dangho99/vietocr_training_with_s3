import traceback

if __name__ == "__main__":
    try:
        from src.model.trainer import Trainer
        trainer = Trainer()
        trainer.train()

    except Exception as e:
        print(f"Recheck at {traceback.print_exc()}")



