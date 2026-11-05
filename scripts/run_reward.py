############################################################################
#                              Run reward 
############################################################################

from src.models.reward_model import RewardModel
from src.trainers.train_reward import TrainReward
import torch 
from pathlib import Path
from scripts.prepare_experiments import (
    load_json, filter_compatible_examples, load_checkpoint_vocab,
    create_training_views, save_experiment_data,
)
from src.data.PreferenceDataSet import PreferenceDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt 
import csv
from datetime import datetime



def main(): 

  
    checkpoint_path = Path(
            "checkpoints/tinygpt_sft.pt"
        )

    train_data_path = Path("data/shp_processed/train_preferences.json")
    val_data_path = Path("data/shp_processed/validation_preferences.json")

    output_path_train = Path("data/shp_processed/train_reward_data.json")
    output_path_val = Path("data/shp_processed/validation_reward_data.json")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path("checkpoints/reward_model.pt")


    # Hyperparameters 
    
    learning_rate = 1e-4
    num_epochs = 2
    eval_every = 1
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    checkpoints = torch.load(checkpoint_path, 
                             map_location=device, 
                             weights_only=True)

    config = checkpoints["config"]

    itos, stoi, block_size = load_checkpoint_vocab(checkpoint_path, device)


    train_data_preferences = load_json(train_data_path)
    filtered_train_preferences = filter_compatible_examples(train_data_preferences, stoi, block_size, max_new_token=100)
    validation_data_preferences = load_json(val_data_path)
    filtered_val_preferences = filter_compatible_examples(validation_data_preferences, stoi, block_size, max_new_token=100)

    _, reward_data_train, _ = create_training_views(filtered_train_preferences)
    _, reward_data_val, _ = create_training_views(filtered_val_preferences)

    save_experiment_data(reward_data_train, output_path_train)
    save_experiment_data(reward_data_val, output_path_val)

    train_loader = DataLoader(PreferenceDataset(
        reward_data_train, block_size, itos, stoi
    ), batch_size= 2, shuffle=True)

    val_loader = DataLoader(PreferenceDataset(
        reward_data_val, block_size, itos, stoi
    ), batch_size= 2, shuffle=False)

    reward_model = RewardModel(config["n_embd"],
                               config["num_heads"], 
                               config["block_size"],
                               config["vocab_size"], 
                               config["dropout"], 
                               config.get("num_layer", config.get("num_layers"))).to(device)


    optimizer = torch.optim.AdamW(reward_model.parameters(), lr = learning_rate)

    trainer = TrainReward(reward_model,num_epochs, optimizer,device)

    History = trainer.full_train(train_loader, val_loader)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    output_dir = Path("experiments") / f"reward_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_losses = History["train_loss"]
    val_losses = History["val_loss"]
    accuracy = History.get("val_accuracy", [])
    
    epochs = History.get(
            "Epochs",
            list(range(1, len(train_losses) + 1))
        )
    
    eval_epochs = History.get(
            "eval_epochs",
            [
                epoch for epoch in epochs
                if epoch % eval_every == 0 or epoch == num_epochs
            ]
        )
    
    val_by_epoch = dict(zip(eval_epochs, val_losses))
    accuracy_by_epoch = dict(zip(eval_epochs, accuracy))
    
    with open(output_dir / "metrics.csv", "w", newline="") as f:
         writer = csv.writer(f)

         writer.writerow([
        "epoch",
        "train_loss",
        "val_loss",
        "val_accuracy"
    ])

         for epoch, train_loss in zip(epochs, train_losses):
              writer.writerow([
            epoch,
            train_loss,
            val_by_epoch.get(epoch, ""),
            accuracy_by_epoch.get(epoch, "")
            
        ])
    

    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(
            {
                "model_state_dict": reward_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": {**config, "num_layer": config.get("num_layer", config.get("num_layers"))},
                "stoi": stoi,
                "itos": itos,
                "epoch": num_epochs
            },
            output_path
        )
    
        
    
    torch.save(
            History,
            output_dir / "history.pt"
        )
    
        # -------------------------------------------------------
        # 13. Save learning curves
        # -------------------------------------------------------
    '''
    plt.figure(figsize=(10, 5))
    
    plt.plot(
            epochs,
            train_losses,
            label="Train"
        )
    
    plt.plot(
            eval_epochs,
            val_losses,
            label="Validation"
        )
    
    plt.xlabel("Epoch")
    plt.ylabel("Reward loss")
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.savefig(
            output_dir / "loss_curve.png",
            dpi=160,
            bbox_inches="tight"
        )
    
    plt.close()

    plt.figure(figsize=(10, 5))
        
    if accuracy:
        plt.plot(
            eval_epochs, accuracy, label="val accuracy"
        )
        
        
        
    plt.xlabel("Epoch")
    plt.ylabel("Validation accuracy")
    plt.legend()
    plt.grid(alpha=0.3)
        
    plt.savefig(
        output_dir / "accuracy_curve.png",
        dpi=160,
        bbox_inches="tight"
            )
        
    plt.close()
        
    
    #print("Reward checkpoint saved to:", output_path)
    #print("Experiment saved to:", output_dir)
    
    '''
    
if __name__ == "__main__":
        main()
    

















