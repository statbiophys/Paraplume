from pathlib import Path
from abodybuilder3.lightning_module import LitABB3
from abodybuilder3.utils import add_atom37_to_output, output_to_pdb, string_to_input
import typer
app = typer.Typer(add_completion=False)
import pandas as pd
module = LitABB3.load_from_checkpoint("/home/athenes/old_gitlab/abodybuilder3_old/output/plddt-loss/best_second_stage_saved.ckpt")
import torch
from codecarbon import EmissionsTracker

@app.command()
def paragraph(
    file_path: Path = typer.Argument(
        ...,
        help="Path of the file.",
        show_default=False,
    ),
    gpu:bool=typer.Option(
        False,
        "--gpu",
    ),
    ):
    device = torch.device(f"cuda:0" if torch.cuda.is_available() and gpu else "cpu")
    model = module.model
    model=model.to(device)
    df = pd.read_csv(file_path)
    Path(f"/home/athenes/Paraplume/data_with_scripts/speed_tests/pdbs_size{len(df)}").mkdir(exist_ok=True)
    tracker = EmissionsTracker(project_name="ABB3_prediction_gpu", experiment_id=f"Size_{len(df)}_{str(gpu)}")
    tracker.start()
    for i, (heavy, light) in enumerate(df[["sequence_heavy","sequence_light"]].values):
        ab_input = string_to_input(heavy=heavy, light=light)
        ab_input_batch = {
            key: (value.unsqueeze(0).to(device) if key not in ["single", "pair"] else value.to(device))
            for key, value in ab_input.items()
        }  # add batch dim
        output = model(ab_input_batch, ab_input_batch["aatype"])
        output = add_atom37_to_output(output, ab_input["aatype"].to(device))
        pdb_string = output_to_pdb(output, ab_input)
        with open(f"/home/athenes/Paraplume/data_with_scripts/speed_tests/pdbs_size{len(df)}/{i}.pdb", "w") as file:
            file.write(pdb_string)
    tracker.stop()

if __name__ == "__main__":
    app()
