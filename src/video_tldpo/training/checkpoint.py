from pathlib import Path
import json, torch

def save_checkpoint(model, optimizer, scheduler, state, directory):
    d = Path(directory); d.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(d / "adapter")
    torch.save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict() if scheduler else None, "state": state}, d / "trainer.pt")
    (d / "config_snapshot.json").write_text(json.dumps(state, ensure_ascii=False, indent=2, default=str))
