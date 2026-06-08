import os
import argparse
import torch
from src.model import CAModel
from src.utils import load_target_image
from src.train import train_morphogenesis, train_chemotaxis, train_autonomous_life
from src.inference import run_inference_phase1, run_inference_chemotaxis, run_inference_ecosystem

def main():
    parser = argparse.ArgumentParser(description="NCA Project Manager")
    parser.add_argument("--action", choices=["train1", "train2", "train_all", "inference1", "inference2"], required=True)
    parser.add_argument("--experiment", choices=["chemotaxis", "chemotaxis_obs", "ecosystem"], required=True)
    parser.add_argument("--target", type=str, required=True, help="Path to the target image")
    parser.add_argument("--save_gif", action="store_true", help="Save inference as GIF instead of displaying it on screen")
    
    args = parser.parse_args()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    GRID_SIZE = 64
    
    target_filename = os.path.basename(args.target).lower()
    if "jelly" in target_filename:
        creature_name = "jelly"
    elif "salamander" in target_filename:
        creature_name = "salamandra"
    elif "giraffe" in target_filename:
        creature_name = "giraffa"
    else:
        creature_name = "base"

    PHASE1_WEIGHTS = f"weights/modello_fase1_forma_{creature_name}.pth"

    if args.experiment in ["chemotaxis", "chemotaxis_obs"]:
        PHASE2_WEIGHTS = f"weights/modello_chemiotassi_{creature_name}.pth"
    elif args.experiment == "ecosystem":
        PHASE2_WEIGHTS = f"weights/modello_vita_autonoma_{creature_name}.pth"

    use_circular = (args.experiment == "ecosystem")
    model = CAModel(channel_n=16, hidden_size=128, goal_dim=2, circular_padding=use_circular).to(DEVICE)
    
    print(f"=== NCA ORCHESTRATOR ===")
    print(f"Device: {DEVICE} | Experiment: {args.experiment} | Action: {args.action}")
    print(f"Detected creature: {creature_name}")
    print(f"Phase 1 Weights file: {PHASE1_WEIGHTS}")
    print(f"Phase 2 Weights file: {PHASE2_WEIGHTS}")
    print(f"Save GIF Mode: {'ENABLED' if args.save_gif else 'DISABLED'}")
    print("========================\n")
    
    if args.save_gif:
        os.makedirs("gifs", exist_ok=True)
    
    if args.action in ["train1", "train_all"]:
        base_target = load_target_image(args.target, size=GRID_SIZE, device=DEVICE)
        model = train_morphogenesis(model, base_target, epochs=8000, grid_size=GRID_SIZE, batch_size=8, device=DEVICE)
        os.makedirs("weights", exist_ok=True)
        torch.save(model.state_dict(), PHASE1_WEIGHTS)
        print(f"Phase 1 weights saved in {PHASE1_WEIGHTS}")

    if args.action in ["train2", "train_all"]:
        base_target = load_target_image(args.target, size=GRID_SIZE, device=DEVICE)
        
        if args.action == "train2":
            if os.path.exists(PHASE1_WEIGHTS):
                model.load_state_dict(torch.load(PHASE1_WEIGHTS, map_location=DEVICE, weights_only=True))
                print("Phase 1 weights loaded.")
            else:
                raise FileNotFoundError(f"Error: {PHASE1_WEIGHTS} not found! Run train1 first.")

        if args.experiment in ["chemotaxis", "chemotaxis_obs"]:
            model = train_chemotaxis(model, base_target, epochs=10000, grid_size=GRID_SIZE, batch_size=8, device=DEVICE)
        elif args.experiment == "ecosystem":
            model = train_autonomous_life(model, base_target, epochs=10000, grid_size=GRID_SIZE, batch_size=8, device=DEVICE)
            
        torch.save(model.state_dict(), PHASE2_WEIGHTS)
        print(f"Phase 2 weights saved in {PHASE2_WEIGHTS}")

    elif args.action == "inference1":
        if not os.path.exists(PHASE1_WEIGHTS):
            raise FileNotFoundError(f"Error: {PHASE1_WEIGHTS} not found!")
        model.load_state_dict(torch.load(PHASE1_WEIGHTS, map_location=DEVICE, weights_only=True))
        model.eval()
        
        gif_path = f"gifs/phase1_morphogenesis_{creature_name}.gif" if args.save_gif else None
        run_inference_phase1(model, DEVICE, grid_size=64, save_path=gif_path)

    elif args.action == "inference2":
        if not os.path.exists(PHASE2_WEIGHTS):
            raise FileNotFoundError(f"Error: {PHASE2_WEIGHTS} not found!")
        model.load_state_dict(torch.load(PHASE2_WEIGHTS, map_location=DEVICE, weights_only=True))
        model.eval()
        
        gif_path = f"gifs/phase2_{args.experiment}_{creature_name}.gif" if args.save_gif else None
        
        if args.experiment == "chemotaxis":
            run_inference_chemotaxis(model, DEVICE, grid_size=96, use_obstacles=False, save_path=gif_path)
        elif args.experiment == "chemotaxis_obs":
            run_inference_chemotaxis(model, DEVICE, grid_size=96, use_obstacles=True, save_path=gif_path)
        elif args.experiment == "ecosystem":
            run_inference_ecosystem(model, DEVICE, grid_size=128, save_path=gif_path)

if __name__ == "__main__":
    main()
